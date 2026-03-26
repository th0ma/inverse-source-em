"""
Error tables for 2‑source regression model.

Handles permutation invariance:
For each sample, evaluates both assignments:
    (predA→trueA, predB→trueB)
    (predA→trueB, predB→trueA)
and keeps the one with smaller total error.

Returns:
{
    "module": "error_tables",
    "status": "passed" | "failed",
    "metrics": {
        "distA": {...},
        "distB": {...},
        "dist_max": {...},
        "rhoA": {...},
        "rhoB": {...},
        "rho_max": {...},
        "phiA": {...},
        "phiB": {...},
        "phi_max": {...}
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
# Summary statistics
# ------------------------------------------------------------
def _summaries(arr):
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


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

    rhoA_all = []
    rhoB_all = []
    rhoMax_all = []

    phiA_all = []
    phiB_all = []
    phiMax_all = []

    with torch.no_grad():
        for Xb, Yb in loader:
            Xb = Xb.to(device, dtype=torch.float32)
            Yb = Yb.to(device, dtype=torch.float32)

            preds = model(Xb).cpu().numpy()
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

            # Choose best assignment
            mask = total1 <= total2

            dA = np.where(mask, dA1, dA2)
            dB = np.where(mask, dB1, dB2)

            # Radii
            rho_true_A = np.linalg.norm(trueA, axis=1)
            rho_true_B = np.linalg.norm(trueB, axis=1)

            rho_pred_A = np.linalg.norm(predA, axis=1)
            rho_pred_B = np.linalg.norm(predB, axis=1)

            rhoA = np.abs(rho_pred_A - rho_true_A)
            rhoB = np.abs(rho_pred_B - rho_true_B)

            # Angles
            phi_true_A = np.arctan2(trueA[:, 1], trueA[:, 0])
            phi_true_B = np.arctan2(trueB[:, 1], trueB[:, 0])

            phi_pred_A = np.arctan2(predA[:, 1], predA[:, 0])
            phi_pred_B = np.arctan2(predB[:, 1], predB[:, 0])

            phiA = np.abs(np.angle(np.exp(1j * (phi_pred_A - phi_true_A))))
            phiB = np.abs(np.angle(np.exp(1j * (phi_pred_B - phi_true_B))))

            # Store
            distA_all.append(dA)
            distB_all.append(dB)
            distMax_all.append(np.maximum(dA, dB))

            rhoA_all.append(rhoA)
            rhoB_all.append(rhoB)
            rhoMax_all.append(np.maximum(rhoA, rhoB))

            phiA_all.append(phiA)
            phiB_all.append(phiB)
            phiMax_all.append(np.maximum(phiA, phiB))

    # Concatenate
    distA_all = np.concatenate(distA_all)
    distB_all = np.concatenate(distB_all)
    distMax_all = np.concatenate(distMax_all)

    rhoA_all = np.concatenate(rhoA_all)
    rhoB_all = np.concatenate(rhoB_all)
    rhoMax_all = np.concatenate(rhoMax_all)

    phiA_all = np.concatenate(phiA_all)
    phiB_all = np.concatenate(phiB_all)
    phiMax_all = np.concatenate(phiMax_all)

    metrics = {
        "distA": _summaries(distA_all),
        "distB": _summaries(distB_all),
        "dist_max": _summaries(distMax_all),

        "rhoA": _summaries(rhoA_all),
        "rhoB": _summaries(rhoB_all),
        "rho_max": _summaries(rhoMax_all),

        "phiA": _summaries(phiA_all),
        "phiB": _summaries(phiB_all),
        "phi_max": _summaries(phiMax_all),
    }

    status = "passed"
    for v in metrics.values():
        if _is_bad(v):
            status = "failed"

    return {
        "module": "error_tables",
        "status": status,
        "metrics": metrics
    }
