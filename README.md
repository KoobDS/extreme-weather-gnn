# Extreme‑Weather‑GNN
Compound extreme‑weather forecasting with Graph Convolutional Networks and causal analysis  
*(Great‑Lakes region, weekly lead time)*  

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

- data/ # raw station data (processed sets too large for GitHub, though replicable by running below)
- training_results/ # Results of model training
- causal_results/ # Results of causal analysis
- visualization/ # PCA plots, compound event relationship plots
- Processing.ipynb # Replicates training data from station data
- train.py # GraphSAGE training loop
- metrics.py # Produces metrics (AUROC)

DL_Final_Project_Report.pdf # Full written report
requirements.txt # For replication

This study's graph's map is shown below: **[`Graph`](./Graph.png)**

## Quick start
```bash
git clone https://github.com/KoobDS/extreme-weather-gnn.git
cd extreme-weather-gnn
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py      # reproduces best checkpoint
```

Contributions

    Benjamin Koob: Data processing (station merge, PCA), graph construction, all training scripts, model training, mapping, and primary prediction analysis.

    Team colleagues (see commit history): Assisted with raw‑data collection and preprocessing, experimental design, and causal‑inference.

This README highlights my work; the full report credits all co‑authors.

