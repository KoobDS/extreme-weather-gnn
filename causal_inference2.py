# imports
import os, itertools, datetime, random, math
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import kendalltau        # for discovery scan

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv

try:                     # causal check optional
    from dowhy import CausalModel
except ImportError:
    CausalModel = None

# constants
BASELINE_YEARS = (1995, 2014)
P_HIGH, P_LOW  = 0.95, 0.05
WINDOW = 7                    # ±days for simultaneity
DISCOVERY_PHI_MIN = 0.10      # tail‑dependence threshold
DISCOVERY_TOP_N   = 5         # how many pairs to keep
GRAPH_K = 5                   # k‑nearest neighbours
EPOCHS  = 10                  # GNN training epochs
LR      = 3e-4
HIDDEN  = 32


# extremes

PCTL_RULES: Dict[str, Dict] = {
    # cloud / sunshine
    "ACMH_CLOUDY": {"base": "ACMH", "tail": "high", "p": P_HIGH},
    "ACMH_CLEAR":  {"base": "ACMH", "tail": "low",  "p": P_LOW},
    "ACSH_CLOUDY": {"base": "ACSH", "tail": "high", "p": P_HIGH},
    "ACSH_CLEAR":  {"base": "ACSH", "tail": "low",  "p": P_LOW},
    "PSUN_DARK":   {"base": "PSUN", "tail": "low",  "p": P_LOW},
    "PSUN_BRIGHT": {"base": "PSUN", "tail": "high", "p": P_HIGH},
    "TSUN_DARK":   {"base": "TSUN", "tail": "low",  "p": P_LOW},
    "TSUN_BRIGHT": {"base": "TSUN", "tail": "high", "p": P_HIGH},
    # temperature
    "TMAX_HOT":    {"base": "TMAX", "tail": "high", "p": P_HIGH},
    "TMAX_COLD":   {"base": "TMAX", "tail": "low",  "p": P_LOW},
    "TMIN_TROP":   {"base": "TMIN", "tail": "high", "p": P_HIGH},
    "TMIN_FROST":  {"base": "TMIN", "tail": "low",  "p": P_LOW},
    # precip & snow
    "PRCP_P95":    {"base": "PRCP", "tail": "high", "p": P_HIGH},
    "SNOW_P95":    {"base": "SNOW", "tail": "high", "p": P_HIGH},
    "SNWD_P90":    {"base": "SNWD", "tail": "high", "p": 0.90},
    # wind
    "AWND_P95":    {"base": "AWND", "tail": "high", "p": P_HIGH},
}

ABS_RULES: Dict[str, Dict] = {
    "PRCP_20MM": {"base": "PRCP", "op": ">=", "value": 20.0},
    "SNOW_10CM": {"base": "SNOW", "op": ">=", "value": 100.0},
    "AWND_GALE": {"base": "AWND", "op": ">=", "value": 17.0},  # m/s
    "TMAX_GT35": {"base": "TMAX", "op": ">=", "value": 35.0},  # °C
    "TMIN_LT0":  {"base": "TMIN", "op": "<=", "value": 0.0},
}

def add_extreme_flags(df: pd.DataFrame,
                      baseline_years: Tuple[int, int] = BASELINE_YEARS,
                      verbose=False) -> pd.DataFrame:
    out = df.copy()
    cols_present = set(out.columns)

    # 2  percentile thresholds
    base = out[out.DATE.dt.year.between(*baseline_years)]
    thresholds = {}
    for tag, rule in PCTL_RULES.items():
        col = rule["base"]
        if col not in cols_present:
            if verbose: print(f"[skip pctl] {tag} (missing {col})")
            continue
        thresholds[tag] = base.groupby("STATION")[col].quantile(rule["p"])

    for tag, rule in PCTL_RULES.items():
        if tag not in thresholds:
            continue
        col, thr = rule["base"], thresholds[tag]
        cond = out[col] >= out.STATION.map(thr) if rule["tail"] == "high" \
               else out[col] <= out.STATION.map(thr)
        out[f"EXT_{tag}"] = cond.astype(int)

    # 2 absolute
    for tag, rule in ABS_RULES.items():
        col = rule["base"]
        if col not in cols_present:
            if verbose: print(f"[skip abs]  {tag} (missing {col})")
            continue
        op, val = rule["op"], rule["value"]
        cond = out[col] >= val if op == ">=" else out[col] <= val
        out[f"EXT_{tag}"] = cond.astype(int)

    # 2.3  weather‑type codes (already binary)
    for col in out.columns:
        if col.startswith(("WT", "WV")) and set(out[col].dropna().unique()) <= {0,1}:
            out[f"EXT_{col}"] = out[col].fillna(0).astype(int)

    return out

# compound event tagging
def rolling_or(series, window=WINDOW):
    return series.rolling(2*window+1, center=True, min_periods=1).max()

def tag_compound_rule_based(df: pd.DataFrame,
                            definitions: Dict[str, List[str]],
                            window: int = WINDOW) -> pd.DataFrame:
    """
    definitions = {
        "HEAT_RAIN": ["EXT_TMAX_HOT", "EXT_PRCP_P95"],
        "GALE_RAIN": ["EXT_AWND_GALE", "EXT_PRCP_P95"]
    }
    Produces COMPOUND_<name> columns.
    """
    out = df.copy()
    # expand each extreme flag through rolling OR
    expanded = {}
    for col in [c for c in out.columns if c.startswith("EXT_")]:
        expanded[col] = out.groupby("STATION")[col].transform(
                   lambda s: rolling_or(s, window)
               ).astype("int8")

    for name, flags in definitions.items():
        missing = [f for f in flags if f not in expanded]
        if missing:
            print(f"[compound‑skip] {name} missing {missing}")
            continue
        cond = np.logical_and.reduce([expanded[f] == 1 for f in flags])
        out[f"COMPOUND_{name}"] = cond.astype(int)
    return out

def discover_tail_pairs(df: pd.DataFrame,
                        min_phi: float = DISCOVERY_PHI_MIN,
                        top_n: int = DISCOVERY_TOP_N) -> List[Tuple[str,str,float]]:
    
    ext_cols = [c for c in df.columns if c.startswith("EXT_")]
    pairs, phi_vals = [], []
    for i, a in enumerate(ext_cols):
        for b in ext_cols[i+1:]:
            phi = np.corrcoef(df[a], df[b])[0,1]
            if phi >= min_phi:
                pairs.append((a, b, phi))
                phi_vals.append(phi)
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)[:top_n]
    return pairs_sorted

def add_discovered_compounds(df: pd.DataFrame,
                             pairs: List[Tuple[str,str,float]],
                             window:int = WINDOW) -> pd.DataFrame:
    out = df.copy()
    expanded = {col: out.groupby("STATION")[col].transform(
                    lambda s: rolling_or(s, window))
                for col in {c for pair in pairs for c in pair[:2]}}

    for a, b, phi in pairs:
        name = f"AUTO_{a[4:]}_{b[4:]}"          # strip "EXT_"
        cond = ((expanded[a] > 0) & (expanded[b] > 0)).astype(int)
        out[f"COMPOUND_{name}"] = cond
    return out

# GRAPH 
def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def build_spatial_edge_index(stations: pd.DataFrame, k: int = GRAPH_K):
    lon, lat = stations.LONGITUDE.values, stations.LATITUDE.values
    n = len(stations)
    edges = []
    for i in range(n):
        d = haversine(lon[i], lat[i], lon, lat)
        knn = np.argsort(d)[1:k+1]
        edges.extend([(i,j) for j in knn])
    return torch.tensor(edges, dtype=torch.long).t()

class GAT(nn.Module):
    def __init__(self, in_ch, hidden=HIDDEN):
        super().__init__()
        self.g1 = GATv2Conv(in_ch, hidden, heads=4, dropout=0.1)
        self.g2 = GATv2Conv(hidden*4, 1, heads=1)
    def forward(self, x, edge_index):
        x = F.elu(self.g1(x, edge_index))
        return self.g2(x, edge_index)          # logits

# main
def main(csv_path: str,
         compound_defs: Dict[str, List[str]],
         use_discovery: bool = True):

    # Load & add extreme flags
    df = pd.read_csv(csv_path, parse_dates=["DATE"])
    df = add_extreme_flags(df, verbose=True)

    # 5.2  Rule‑based compounds
    df = tag_compound_rule_based(df, compound_defs)

    # Discovery compounds
    if use_discovery:
        pairs = discover_tail_pairs(df)
        print("Top tail‑dependent pairs:", pairs)
        df = add_discovered_compounds(df, pairs)

    df.to_csv('merged_pca_11_cmp_7d.csv')
    # ------------------ choose one compound label ------------
    label_col = "COMPOUND_HEAT_RAIN"  # example; change as needed
    if label_col not in df.columns:
        raise ValueError(f"{label_col} not found – adjust compound_defs.")

    # build graph dataset
    stations = df[["STATION","LATITUDE","LONGITUDE"]].drop_duplicates()
    station2idx = dict(zip(stations.STATION, stations.index))
    df["NODE_ID"] = df.STATION.map(station2idx)
    edge_index = build_spatial_edge_index(stations)

    #  feature list = all numeric weather vars (non‑EXT / non‑COMPOUND)
    feat_cols = [c for c in df.columns
                 if c not in ("DATE","STATION","NODE_ID") and
                    not c.startswith(("EXT_","COMPOUND_"))]
    # quick standardise
    df[feat_cols] = (df[feat_cols] - df[feat_cols].mean()) / df[feat_cols].std()

    def make_day_graph(gdate):
        sub = df[df.DATE == gdate].sort_values("NODE_ID")
        x = torch.tensor(sub[feat_cols].values, dtype=torch.float)
        y = torch.tensor(sub[label_col].values, dtype=torch.float).unsqueeze(1)
        return Data(x=x, edge_index=edge_index, y=y)

    all_dates = np.sort(df.DATE.unique())
    split = np.datetime64("2016-01-01")
    train_dates = all_dates[all_dates < split]
    test_dates  = all_dates[all_dates >= split]
    train_DL = DataLoader([make_day_graph(d) for d in train_dates], shuffle=True)
    test_DL  = DataLoader([make_day_graph(d) for d in test_dates])

    # train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAT(len(feat_cols)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(1, EPOCHS+1):
        for phase, loader, train in [("train",train_DL,True), ("test",test_DL,False)]:
            model.train(train)
            tot, n = 0.0, 0
            for data in loader:
                data = data.to(device)
                if train: opt.zero_grad()
                loss = F.binary_cross_entropy_with_logits(
                    model(data.x, data.edge_index), data.y, reduction="sum")
                if train: loss.backward(); opt.step()
                tot += loss.item(); n += data.y.numel()
            print(f"Epoch {epoch:02d} {phase} BCE={tot/n:.4f}")

    # quick causal effect (if DoWhy available)
    if CausalModel and label_col in df.columns and "TMAX" in df.columns:
        causal_df = df[["TMAX","PRCP",label_col]].rename(
            columns={label_col:"compound","TMAX":"temp","PRCP":"rain"})
        graph = 'digraph{ temp->compound; rain->compound; rain->temp }'
        cm = CausalModel(causal_df, treatment="temp", outcome="compound",
                         graph=graph, common_causes=["rain"])
        est = cm.estimate_effect(cm.identify_effect(), method_name="backdoor.logistic_regression")
        print("Causal coef temp→compound:", est.value)

if __name__ == "__main__":
    PREDEF = {
    "HEAT_DRY":  ["EXT_TMAX_HOT",],  
    
    "HEAT_WIND": ["EXT_TMAX_HOT", "EXT_AWND_GALE"],
    
    "HOT_CLEAR_SKY": ["EXT_TMAX_HOT", "EXT_ACSH_CLEAR"],
    
    "STORM_RAIN_WIND": ["EXT_PRCP_P95", "EXT_WSFG_DAMG"],
   
    "BLIZZARD": ["EXT_SNOW_P95", "EXT_AWND_GALE"],
    
    "RAIN_ON_SNOW": ["EXT_PRCP_P95", "SNWD_P90_LAG1", "TMAX_ABOVE0"],
    
    "THAW_FREEZE": ["TMAX_HOT_T", "TMIN_FROST_TPLUS1"],
    
    "BACK2BACK_RAIN": ["EXT_PRCP_P95"],   
    
}
    main("merged_pca_11.csv", PREDEF, use_discovery=True)
