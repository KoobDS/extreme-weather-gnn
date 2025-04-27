"""metrics.py - AUROC utilities (multi-label)"""
import torch
from sklearn.metrics import roc_auc_score

def macro_auroc(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Macro AUROC across classes. preds/targets are (batch, C) logits & labels."""
    p = preds.detach().cpu().numpy()
    t = targets.detach().cpu().numpy()
    try:
        return roc_auc_score(t, p, average="macro")
    except ValueError:
        return float("nan")
