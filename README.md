# Extreme‑Weather‑GNN
Compound extreme‑weather forecasting with Graph Convolutional Networks and causal analysis *(Great‑Lakes region, weekly lead time)*  

## Headline
A 2‑layer GraphSAGE model predicts eight compound climate hazards (heat–drought, rain‑on‑snow, etc.).  
Key result: **macro‑AUROC = 0.95** on the 1995-2025 7-station hold‑out set.

Full methodology & figures are in **[`DL_Final_Project_Report.pdf`](./DL_Final_Project_Report.pdf)**.

## Results
| Metric (hold‑out) | Score |
|-------------------|-------|
| Macro‑AUROC       | 0.9499 |
| Binary‑Cross-Ent  | 0.1775 |
| Top causal driver | ΔT<sub>max</sub> → Heat + Dry (+3.6 pp ACE) |

## Tech stack
- Python 3.10
- PyTorch 2.3 · Torch‑Geometric 2.5
- NumPy 2.2 · pandas 2.2 · scikit‑learn 1.5
- YAML‑driven config (PyYAML 6)

## Repo structure

| Path | Description |
|------|-------------|
| `data/` | Raw station CSVs (processed tensors reproducible locally) |
| `training_results/` | Saved checkpoints, logs, and metrics |
| `causal_results/` | Outputs from DoWhy causal analysis |
| `visualization/` | PCA plots and compound‑event visualizations |
| `Processing.ipynb` | Generates training tensors from raw data |
| `train.py` | GraphSAGE training loop (YAML‑configurable) |
| `metrics.py` | AUROC / BCE / confusion‑matrix utilities |
| `requirements.txt` | Package versions for replication |
| `DL_Final_Project_Report.pdf` | Complete written report |

This study's graph's map is shown below: **[`Great Lakes Graph`](./Graph.png)**

## Quick start
```bash
git clone https://github.com/KoobDS/extreme-weather-gnn.git
cd extreme-weather-gnn
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py      # reproduces best checkpoint
```

### Contributions
- Benjamin Koob: Data processing (station merge, PCA), graph construction, all training scripts, model training, mapping, and primary prediction analysis.
- Teammates (see commit history): Assisted with raw‑data collection and preprocessing, experimental design, and causal‑inference.
This README highlights my work; the full report credits all co‑authors.

