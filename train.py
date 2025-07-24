from __future__ import annotations
import argparse, csv, random, re, warnings, yaml
from pathlib import Path
from typing import List

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import BallTree
from sklearn.exceptions import UndefinedMetricWarning
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv          # <- back to SAGEConv

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ───────────────────────── helpers ──────────────────────────
def set_seed(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

_float = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
def _coerce(v):
    if isinstance(v, dict):  return {k: _coerce(x) for k, x in v.items()}
    if isinstance(v, list):  return [_coerce(x) for x in v]
    if isinstance(v, str):
        if v.isdigit():        return int(v)
        if _float.match(v):    return float(v)
    return v

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dφ = np.radians(lat2-lat1); dλ = np.radians(lon2-lon1)
    a  = np.sin(dφ/2)**2 + np.cos(np.radians(lat1))*np.cos(np.radians(lat2))*np.sin(dλ/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def build_knn(lat, lon, k:int):
    """Return (edge_index, edge_weight_inv_dist) - weight is optional."""
    tree = BallTree(np.radians(np.c_[lat, lon]), metric="haversine")
    _, idx = tree.query(np.radians(np.c_[lat, lon]), k=k+1)
    src = np.repeat(np.arange(len(lat)), k); dst = idx[:,1:].reshape(-1)
    ei  = torch.as_tensor(
            np.concatenate([np.vstack([src,dst]), np.vstack([dst,src])], 1),
            dtype=torch.long)

    dist = haversine_km(lat[src], lon[src], lat[dst], lon[dst]) + 1e-6
    ew   = torch.tensor(1.0/dist, dtype=torch.float32)          # [E]
    ew   = torch.cat([ew, ew])                                  # mirror
    return ei, ew

def macro_auc(y: np.ndarray, p: np.ndarray) -> float:
    scores = [roc_auc_score(y[:,c], p[:,c])
              for c in range(y.shape[1]) if len(np.unique(y[:,c])) > 1]
    return float(np.mean(scores)) if scores else float("nan")

# ───────────────────────── dataset ──────────────────────────
class WeeklyDataset(InMemoryDataset):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        super().__init__(cfg["processed_dir"])
        self.data, self.slices = torch.load(self.processed_paths[0])

    # filenames
    def raw_file_names(self):       return [Path(self.cfg["csv_path"]).name]
    def processed_file_names(self): return ["graphs.pt"]
    def download(self):             pass

    def process(self):
        df = pd.read_csv(self.cfg["csv_path"])

        targets = [c for c in df.columns if c.startswith("COMPOUND_") and c.endswith("_next")]
        meta    = {"STATION","NAME","LATITUDE","LONGITUDE","ELEVATION","WEEK_START","SPLIT", *targets}
        feats   = [c for c in df.columns if c not in meta]      # unused > left as columns

        st = df[["STATION","LATITUDE","LONGITUDE"]].drop_duplicates().sort_values("STATION")
        ei, ew = build_knn(st.LATITUDE.values, st.LONGITUDE.values, k=self.cfg["k_neighbors"])

        train_st = set(df[df.SPLIT=="train"].STATION.unique())
        graphs: List[Data] = []
        for _, g in df.groupby("WEEK_START"):
            g = g.set_index("STATION").loc[st.STATION]          # align row order
            x = torch.tensor(g[feats].values, dtype=torch.float32)
            y = torch.tensor(g[targets].values, dtype=torch.float32)
            mask_tr = torch.tensor([s in train_st for s in st.STATION], dtype=torch.bool)

            graphs.append(Data(
                x=x, y=y, edge_index=ei,          # <- pass only edge_index
                train_mask=mask_tr, val_mask=~mask_tr))

        Path(self.cfg["processed_dir"]).mkdir(parents=True, exist_ok=True)
        torch.save(self.collate(graphs), self.processed_paths[0])

# ───────────────────────── model ──────────────────────────
class NodeSAGE(nn.Module):
    def __init__(self, in_d:int, cfg:dict, out_d:int):
        super().__init__()
        h, L, p = cfg["hidden_dim"], cfg["num_layers"], cfg.get("dropout",0.0)
        self.convs = nn.ModuleList([SAGEConv(in_d, h)])
        for _ in range(L-1):
            self.convs.append(SAGEConv(h, h))
        self.drop = p
        self.cls  = nn.Linear(h, out_d)

    def forward(self, d: Data):
        x = d.x
        for c in self.convs:
            x = F.relu(c(x, d.edge_index))        # <- **no edge_weight / edge_attr**
            if self.drop: x = F.dropout(x, self.drop, self.training)
        return self.cls(x)

# ─────────────────────── train / eval ──────────────────────
def _epoch(loader, model, crit, opt=None):
    train = opt is not None
    model.train(train)
    tot, outs, tgts = 0., [], []
    for batch in loader:
        batch = batch.to(next(model.parameters()).device, non_blocking=True)
        mask  = batch.train_mask if train else batch.val_mask
        if not mask.any(): continue

        logit = model(batch)
        loss  = crit(logit[mask], batch.y[mask])
        if train:
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

        tot += loss.item()
        outs.append(torch.sigmoid(logit[mask]).detach().cpu())
        tgts.append(batch.y[mask].cpu())

    if not outs: return float("nan"), float("nan")
    p = torch.cat(outs).numpy(); y = torch.cat(tgts).numpy()
    return tot/len(outs), macro_auc(y, p)

# ─────────────────────────── main ──────────────────────────
def main(cfg_path:str):
    cfg = _coerce(yaml.safe_load(Path(cfg_path).read_text()))
    set_seed(cfg["seed"])

    ds   = WeeklyDataset(cfg)
    dkw  = dict(batch_size=cfg["batch_size"], num_workers=0)
    trDL = DataLoader(ds, shuffle=True,  **dkw)
    vaDL = DataLoader(ds, shuffle=False, **dkw)

    model = NodeSAGE(ds[0].x.size(1), cfg, ds[0].y.size(1)).to(cfg["device"])
    opt   = torch.optim.AdamW(model.parameters(),
                              lr=float(cfg["lr"]),
                              weight_decay=float(cfg["weight_decay"]))

    sched = None
    if cfg.get("lr_scheduler") == "cosine_warm_restart":
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=max(10, int(cfg["num_epochs"])//10), eta_min=1e-5)

    crit = nn.BCEWithLogitsLoss()
    out_dir = Path(cfg["output_dir"]); (out_dir/"logs").mkdir(parents=True, exist_ok=True)

    best, wait = float("inf"), 0
    with (out_dir/"logs"/"train_log.csv").open("w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["epoch","train_bce","val_bce","val_auc","lr"])
        for ep in range(1, int(cfg["num_epochs"])+1):
            tr_bce,_      = _epoch(trDL, model, crit, opt)
            va_bce,va_auc = _epoch(vaDL, model, crit)
            if sched: sched.step()

            lr_now = opt.param_groups[0]["lr"]
            wr.writerow([ep,tr_bce,va_bce,va_auc,lr_now]); f.flush()
            print(f"Ep{ep:03d}  BCE {tr_bce:.4f}/{va_bce:.4f}  "
                  f"AUROC {va_auc:.3f}  lr {lr_now:.2e}")

            if np.isfinite(va_bce) and (best - va_bce) > cfg["min_delta"]:
                best, wait = va_bce, 0; torch.save(model.state_dict(), out_dir/"best.pt")
            else:
                wait += 1
                if wait >= cfg["patience"]:
                    print("Early stopping"); break

    print(f"Training done. Best val BCE {best:.4f} (weights @ {out_dir/'best.pt'})")

# ────────────────────── entry point ───────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True)
    main(p.parse_args().config)