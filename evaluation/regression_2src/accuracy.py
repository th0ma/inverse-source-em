"""
Accuracy evaluation for 2‑source regression model.

Handles permutation invariance:
For each sample, evaluates both assignments:
    (predA→trueA, predB→trueB)
    (predA→trueB, predB→trueA)
and keeps the one with smaller total error.

Returns:
{
    "module": "accuracy",
    "status": "passed" | "failed",
    "metrics": {
        "mean_distA": ...,
        "mean_distB": ...,
        "mean_dist_max": ...,
        "p99_distA": ...,
        "p99_distB": ...,
        "p99_dist_max": ...
    }
}
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


# ------------------------------------------------------------
# Utility: robust NaN/Inf checking
# ------------------------------------------------------------
def _is_bad(x):
    try:
        arr = np.asarray(x, dtype=float)
        return np.isnan(arr).any() or np.isinf(arr).any()
    except Exception:
        return True


# ------------------------------------------------------------
# Core evaluation
# ------------------------------------------------------------
def evaluate(model, X, Y, device="cpu", batch_size=256):
    model.eval()

    ds = TensorDataset(
        torch.from_numpy(X).to(torch.float32),
        torch.from_numpy(Y).to(torch.float32)
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    distA_all = []
    distB_all = []
    distMax_all = []

    with torch.no_grad():
        for Xb, Yb in loader:
            Xb = Xb.to(device, dtype=torch.float32)
            Yb = Yb.to(device, dtype=torch.float32)

            preds = model(Xb)  # shape (B, 4)
            preds = preds.cpu().numpy()
            Yb = Yb.cpu().numpy()

            # Split into A and B
            predA = preds[:, 0:2]
            predB = preds[:, 2:4]

            trueA = Yb[:, 0:2]
            trueB = Yb[:, 2:4]

            # Assignment 1: A→A, B→B
            dA1 = np.linalg.norm(predA - trueA, axis=1)
            dB1 = np.linalg.norm(predB - trueB, axis=1)
            total1 = dA1 + dB1

            # Assignment 2: A→B, B→A
            dA2 = np.linalg.norm(predA - trueB, axis=1)
            dB2 = np.linalg.norm(predB - trueA, axis=1)
            total2 = dA2 + dB2

            # Choose best assignment per sample
            mask = total1 <= total2

            dA = np.where(mask, dA1, dA2)
            dB = np.where(mask, dB1, dB2)

            distA_all.append(dA)
            distB_all.append(dB)
            distMax_all.append(np.maximum(dA, dB))

    distA_all = np.concatenate(distA_all)
    distB_all = np.concatenate(distB_all)
    distMax_all = np.concatenate(distMax_all)

    metrics = {
        "mean_distA": float(distA_all.mean()),
        "mean_distB": float(distB_all.mean()),
        "mean_dist_max": float(distMax_all.mean()),
        "p99_distA": float(np.percentile(distA_all, 99)),
        "p99_distB": float(np.percentile(distB_all, 99)),
        "p99_dist_max": float(np.percentile(distMax_all, 99)),
    }

    status = "passed"
    if any(_is_bad(v) for v in metrics.values()):
        status = "failed"

    return {
        "module": "accuracy",
        "status": status,
        "metrics": metrics
    }
