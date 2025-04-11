"""
Predicting Compound Extreme Events With Graph-Based Deep Learning & Causal Inference:

How to run: "python gnn_compound_events.py --csv data/3974327.csv --device cuda"
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import BallTree
from torch_geometric.data import Data, DataLoader, InMemoryDataset
from torch_geometric.nn import SAGEConv, global_mean_pool

###############################################################################
# 0.  Configuration & CLI
###############################################################################

@dataclass
class Config:
    # I/O
    csv_path: str = "data/3974327.csv"  # raw NOAA station CSV
    processed_dir: str = "processed"  # where cached .pt files go

    # Graph construction
    k_neighbors: int = 8  # for k-NN graph

    # Temporal sampling
    window: int = 7  # days per graph sample

    # Model
    hidden_dim: int = 64
    num_layers: int = 3
    use_causal: bool = False

    # Training
    batch_size: int = 32
    lr: float = 1e-3
    num_epochs: int = 100
    patience: int = 10  # early stopping

    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


###############################################################################
# 1.  Utilities
###############################################################################

def set_seed(seed: int) -> None:
    """Ensure reproducibility (within limits of CUDA nondeterminism)."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6_371.0  # Earth radius [km]
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_knn_edges(lats: np.ndarray, lons: np.ndarray, k: int) -> torch.Tensor:
    """Return a *bidirectional* edge_index for k-NN graph."""
    coords = np.vstack([lats, lons]).T
    tree = BallTree(np.radians(coords), metric="haversine")
    _, idx = tree.query(np.radians(coords), k=k + 1)  # self included
    src = np.repeat(np.arange(len(coords)), k)
    dst = idx[:, 1:].reshape(-1)  # drop self-loop
    edge_index = np.vstack([src, dst])
    # make bidirectional
    edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)
    return torch.as_tensor(edge_index, dtype=torch.long)

###############################################################################
# 2.  Dataset
###############################################################################

class CompoundEventsDataset(InMemoryDataset):
    """Station-level daily data ➜ rolling-window graph snapshots."""

    def __init__(self, cfg: Config, transform=None, pre_transform=None):
        self.cfg = cfg
        super().__init__(cfg.processed_dir, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------
    @property
    def raw_file_names(self) -> List[str]:
        return [os.path.basename(self.cfg.csv_path)]

    @property
    def processed_file_names(self) -> List[str]:
        return ["compound_events.pt"]

    def download(self):
        # Expect local file; implement if remote download is needed
        pass

    def process(self):
        print("[Dataset] Processing raw CSV …")
        df = pd.read_csv(self.cfg.csv_path)
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

        # ------------------------------------------------------------------
        # Select numeric feature columns (ignore *_ATTRIBUTES)
        # ------------------------------------------------------------------
        feature_cols = [
            c
            for c in df.columns
            if c
            not in {
                "STATION",
                "NAME",
                "LATITUDE",
                "LONGITUDE",
                "ELEVATION",
                "DATE",
            }
            and not c.endswith("_ATTRIBUTES")
        ]
        df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

        # ------------------------------------------------------------------
        # Basic cleaning – drop rows w/ all-NaN features
        # ------------------------------------------------------------------
        df = df.dropna(subset=feature_cols, how="all")

        # ------------------------------------------------------------------
        # Build station lookup
        # ------------------------------------------------------------------
        stations = df[["STATION", "LATITUDE", "LONGITUDE"]].drop_duplicates()
        stations = stations.sort_values("STATION").reset_index(drop=True)
        lat_arr, lon_arr = stations["LATITUDE"].values, stations["LONGITUDE"].values
        edge_index = build_knn_edges(lat_arr, lon_arr, self.cfg.k_neighbors)

        # ------------------------------------------------------------------
        # Pre-compute per-station daily feature matrix
        # ------------------------------------------------------------------
        daily = (
            df.groupby(["STATION", "DATE"])[feature_cols]
            .mean()
            .reset_index()
            .pivot(index="DATE", columns="STATION", values=feature_cols)
        )
        # MultiIndex columns ➜ flatten:  (feature, station)  ➜  f#_s#
        daily.columns = [f"{feat}__{stn}" for feat, stn in daily.columns]
        daily = daily.sort_index()

        # ------------------------------------------------------------------
        # Rolling windows ➜ PyG Data objects
        # ------------------------------------------------------------------
        data_list: List[Data] = []
        win = self.cfg.window
        for start_idx in range(len(daily) - win + 1):
            window_slice = daily.iloc[start_idx : start_idx + win]
            # reshape to (num_stations, win * num_features)
            arr = (
                window_slice.to_numpy().reshape(win, len(feature_cols), len(stations))
            )  # (win, feat, station)
            arr = arr.transpose(2, 0, 1).reshape(len(stations), -1)
            x = torch.tensor(arr, dtype=torch.float32)

            # ------------------------------------------------------------------
            # TODO ❶ – Replace with real compound-event labels
            # For now, dummy zeros.
            # ------------------------------------------------------------------
            y = torch.zeros(1, dtype=torch.float32)

            data = Data(x=x, edge_index=edge_index, y=y)
            data.date_range = (
                window_slice.index[0],
                window_slice.index[-1],
            )  # metadata
            data_list.append(data)

        print(f"[Dataset] Generated {len(data_list)} graph snapshots.")
        data, slices = self.collate(data_list)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])

###############################################################################
# 3.  Model Components
###############################################################################

class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, batch):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
        x = self.convs[-1](x, edge_index)
        return global_mean_pool(x, batch)  # graph-level embedding


class CausalHead(nn.Module):
    """Simple TARNet-style head (binary treatment)."""

    def __init__(self, dim: int):
        super().__init__()
        self.t_logits = nn.Linear(dim, 1)
        self.y0 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))
        self.y1 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, z):
        t_logit = self.t_logits(z)
        t = torch.bernoulli(torch.sigmoid(t_logit))  # sampled treatment
        y0, y1 = self.y0(z), self.y1(z)
        y_factual = torch.where(t.bool(), y1, y0)
        return y_factual.squeeze(), t_logit.squeeze(), (y0.squeeze(), y1.squeeze())


class CompoundEventModel(nn.Module):
    def __init__(self, in_dim: int, cfg: Config):
        super().__init__()
        self.use_causal = cfg.use_causal
        self.encoder = GraphEncoder(in_dim, cfg.hidden_dim, cfg.num_layers)
        self.head = CausalHead(cfg.hidden_dim) if self.use_causal else nn.Linear(
            cfg.hidden_dim, 1
        )

    def forward(self, data: Data):
        z = self.encoder(data.x, data.edge_index, data.batch)
        if self.use_causal:
            return self.head(z)  # tuple
        return self.head(z).squeeze()

###############################################################################
# 4.  Training Helpers
###############################################################################

def step(model, data, criterion):
    out = model(data)
    if model.use_causal:
        out = out[0]  # y_factual
    loss = criterion(out, data.y)
    return loss


def run_epoch(model, loader, optimizer, criterion, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for data in loader:
            data = data.to(next(model.parameters()).device)
            if train:
                optimizer.zero_grad()
            loss = step(model, data, criterion)
            if train:
                loss.backward()
                optimizer.step()
            total += loss.item() * data.num_graphs
            n += data.num_graphs
    return total / n

###############################################################################
# 5.  Main Entrypoint
###############################################################################

def main(cfg: Config):
    set_seed(cfg.seed)

    # ------------------------------------------------------------------
    # Dataset & loaders
    # ------------------------------------------------------------------
    dataset = CompoundEventsDataset(cfg)
    split = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [split, len(dataset) - split], generator=torch.Generator().manual_seed(cfg.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    in_dim = dataset[0].x.size(1)
    model = CompoundEventModel(in_dim, cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val, patience_cnt = float("inf"), 0
    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, train=False)
        print(f"Epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                print("Early stopping.")
                break

    print("Training finished. Best val loss:", best_val)


###############################################################################
# 6.  CLI glue
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compound Event GNN Trainer")
    parser.add_argument("--csv", type=str, default=Config.csv_path, help="Path to raw NOAA CSV")
    parser.add_argument("--device", type=str, default=Config.device, help="cpu | cuda")
    args = parser.parse_args()

    cfg = Config(csv_path=args.csv, device=args.device)
    main(cfg)

###############################################################################
# 7.  TODOs / Next Steps (in-code reminders)
###############################################################################
# Implement real compound-event labelling logic in Dataset.process()
# Swap rolling mean features for something richer (e.g., quantiles)
# Add edge attributes (distance, watershed, tele-connection index)
# Handle severe class imbalance (focal loss or oversampling)
# Integrate causal metrics (ATE, PEHE) when cfg.use_causal = True
