#!/usr/bin/env python3

import argparse, json, warnings, yaml
from pathlib import Path
from typing import List

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from dowhy import CausalModel
from tqdm import tqdm
warnings.filterwarnings("ignore", category=UserWarning)

# build k-nearest edge_index 
def build_knn(lat, lon, k:int=5):
    from sklearn.neighbors import BallTree
    tree = BallTree(np.radians(np.c_[lat, lon]), metric="haversine")
    _, idx = tree.query(np.radians(np.c_[lat, lon]), k=k+1)
    src = np.repeat(np.arange(len(lat)), k); dst = idx[:,1:].reshape(-1)
    ei  = torch.as_tensor(np.concatenate([np.vstack([src,dst]), np.vstack([dst,src])],1),
                          dtype=torch.long)
    return ei

# model skeleto
class NodeSAGE(nn.Module):
    def __init__(self, in_d:int, h:int, out_d:int, drop:float):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_d, h), SAGEConv(h, h)])
        self.cls = nn.Linear(h, out_d); self.drop = drop
    def forward(self, data:Data):
        x = data.x
        for c in self.convs:
            x = F.relu(c(x, data.edge_index))
            if self.drop: x = F.dropout(x, p=self.drop, training=self.training)
        return self.cls(x)

# build graphs from CSV 
def csv_to_graphs(csv_path:str, k:int)->List[Data]:
    df = pd.read_csv(csv_path)

    targets = [c for c in df.columns if c.startswith("COMPOUND_") and c.endswith("_next")]
    feats   = [c for c in df.columns
               if c not in {"STATION","NAME","LATITUDE","LONGITUDE","ELEVATION",
                             "WEEK_START","SPLIT",*targets}]

    st = df[["STATION","LATITUDE","LONGITUDE"]].drop_duplicates().sort_values("STATION")
    edge = build_knn(st.LATITUDE.values, st.LONGITUDE.values, k)

    graphs=[]
    for _, g in df.groupby("WEEK_START"):
        g = g.set_index("STATION").loc[st.STATION]   # align order
        x = torch.tensor(g[feats].values, dtype=torch.float32)
        y = torch.tensor(g[targets].values, dtype=torch.float32)
        graphs.append(Data(x=x, y=y, edge_index=edge, x_names=feats, y_names=targets))
    return graphs

def locate(col: str, names: list) -> int:
    col_clean = col.strip().lower()

    # exact (case-insensitive) hit
    for i, n in enumerate(names):
        if n.lower() == col_clean:
            return i

    # prefix match (unique)
    hits = [i for i, n in enumerate(names)
            if n.lower().startswith(col_clean + "_")]
    if len(hits) == 1:
        return hits[0]
    if len(hits) == 0:
        raise ValueError(f"No feature matches '{col}' in {names}")
    raise ValueError(f"Ambiguous feature name for '{col}': "
                     f"{[names[i] for i in hits]}\n"
                     f"Specify one explicitly via --drivers.")

#  main 
def main(args):
    # read hyper-params
    cfg = yaml.safe_load(Path(args.yaml).read_text())
    hdim, drop, k = cfg["hidden_dim"], cfg.get("dropout",0.0), cfg["k_neighbors"]

    # graphs + driver / label columns
    graphs = csv_to_graphs(args.csv, k)
    print("\nModel feature names:\n", graphs[0].x_names, "\n")
    dl     = DataLoader(graphs, batch_size=1, shuffle=False)

    default_drivers = ["TMAX","PRCP"]
    
    if args.drivers:
        driver_cols = [c.strip() for c in args.drivers.split(",")]
    else:
        driver_cols = [c for c in default_drivers if c in graphs[0].x_names]

    if not driver_cols:
        raise ValueError("None of the requested driver columns "
                        f"({args.drivers or default_drivers}) are in dataset.\n"
                        f"Available x_names: {graphs[0].x_names}")
    print("Using driver variables:", driver_cols)

    comp_cols = graphs[0].y_names                   

    # build & load model
    model = NodeSAGE(
        in_d  = graphs[0].x.shape[1],
        h     = hdim,
        out_d = len(comp_cols),
        drop  = drop
    )
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    # forward pass → DataFrame
    recs=[]
    with torch.no_grad():
        for g in tqdm(dl, desc="forward"):
            p = torch.sigmoid(model(g)).squeeze(0)
            
            names = g.x_names
            if names and isinstance(names[0], list):
                names = names[0]      # [N,L]
            idx = [locate(c, names) for c in driver_cols]
             #indices of drivers
            drv = g.x[:, idx]  
            for n in range(p.size(0)):
                r = {c: drv[n,i].item() for i,c in enumerate(driver_cols)}
                r.update({f"PRED_{c}": p[n,i].item() for i,c in enumerate(comp_cols)})
                recs.append(r)
    df = pd.DataFrame.from_records(recs)

    # DoWhy causal loop
    ace={}
    for comp in comp_cols:
        outcome = f"PRED_{comp}"
        ace[comp] = {}
        for treat in driver_cols:         # ← loop over *all* drivers
            common = [d for d in driver_cols if d != treat]

            sub = df[[treat] + common + [outcome]].dropna()
            if sub[treat].nunique() < 2 or sub[outcome].nunique() < 2:
                print(f"[skip] {comp} / {treat} – no variation")
                continue

            edges = ";".join(
                [f"{treat}->{outcome}"] +
                [f"{z}->{outcome}" for z in common] +
                [f"{z}->{treat}"   for z in common]
            ) + ";"

            cm = CausalModel(
                    data=sub,
                    treatment=treat,
                    outcome=outcome,
                    graph=f"digraph{{ {edges} }}",
                    common_causes=common
            )

            # optional visualisation
            if args.save_graphs:
                fname = Path(args.save_graphs)/f"{comp}__{treat}.png"
                cm.view_model(layout='dot', file_name=str(fname))

            est = cm.estimate_effect(
                    cm.identify_effect(),
                    method_name="backdoor.linear_regression")

            ace[comp][treat] = est.value
            print(f"{comp:<30s}  ACE({treat} → {comp}) = {est.value:+.4f}")
    # for c in comp_cols:
    #     treat = ("TMAX" if ("HEAT" in c or "THAW" in c) else
    #              "WSFG" if "WIND" in c else "PRCP")
    #     if treat not in driver_cols: continue
    #     outcome = f"PRED_{c}"; 
    #     common=[d for d in driver_cols if d!=treat]

    #     sub = df[[treat] + common + [outcome]].dropna()
    #     if sub[treat].nunique() < 2 or sub[outcome].nunique() < 2:
    #         print(f"[skip] {c} no variation in treatment/outcome")
    #         continue

    #     edges_list = (
    #         [f"{treat}->{outcome}"] +                     # 1st edge
    #         [f"{z}->{outcome}" for z in common] +         # conf → outcome
    #         [f"{z}->{treat}"   for z in common]           # conf → treat
    #     )
    #     edges = ";".join(edges_list) + ";" 

    #     cm = CausalModel(
    #             data=sub,
    #             treatment=treat,
    #             outcome=outcome,
    #             graph=f"digraph{{ {edges} }}",   # <- wrap here
    #             common_causes=common
    #     )

    #     if args.view:                              # --view flag on CLI
    #         cm.view_model(layout='dot')            # Jupyter/VSCode popup
    #     elif args.save_graphs:                     # --save-graphs dir_path
    #         fname = Path(args.save_graphs)/f"{c}_graph.png"
    #         cm.view_model(layout='dot', file_name=str(fname))

    #     est=cm.estimate_effect(cm.identify_effect(),
    #                            method_name="backdoor.linear_regression")
    #     ace[c]=est.value
    #     print(f"{c:<25s}  ACE({treat} → {c}) = {est.value:+.4f}")

    Path("ace_results.json").write_text(json.dumps(ace, indent=2))
    print("Saved → ace_results.json")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--yaml", required=True)
    p.add_argument("--drivers",
               help="Comma-separated list of driver columns present in x_names")
    p.add_argument("--view", action="store_true",
               help="Display each causal graph with Graphviz")
    p.add_argument("--save-graphs",
               help="Directory to dump <compound>_graph.png images")
    main(p.parse_args())
