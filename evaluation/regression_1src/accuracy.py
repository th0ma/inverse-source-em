"""
Accuracy evaluation for the 1‑source regression model.

This module computes absolute and relative errors for all predicted quantities:

- Cartesian localization error:
  $\left(x,\\,y\right)$

- Polar absolute errors:
  $\rho$, $\phi$

- Strength absolute error:
  $I$

- Relative polar error:
  $\rho_{\mathrm{rel}} = \dfrac{\rho_{\mathrm{abs}}}{\max\left(\rho_{\mathrm{true}},\,10^{-12}\right)}$

- Relative strength error:
  $I_{\mathrm{rel}} = \dfrac{I_{\mathrm{abs}}}{\max\left(I_{\mathrm{true}},\,10^{-12}\right)}$

Returned structure:
{
    "module": "accuracy",
    "status": "passed" | "failed",
    "metrics": {
        "xy_abs": {...},
        "rho_abs": {...},
        "phi_abs": {...},
        "I_abs": {...},
        "rho_rel": {...},
        "I_rel": {...}
    }
}
"""



import numpy as np
import torch
import math
from torch.utils.data import DataLoader, TensorDataset


# ------------------------------------------------------------
# Utility: robust NaN/Inf checking
# ------------------------------------------------------------
def _is_bad(x):
    if isinstance(x, dict):
        return any(_is_bad(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return any(_is_bad(v) for v in x)
    try:
        arr = np.asarray(x, dtype=float)
        return np.isnan(arr).any() or np.isinf(arr).any()
    except Exception:
        return False


# ------------------------------------------------------------
# Compute summary statistics
# ------------------------------------------------------------
def _summaries(arr):
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


# ------------------------------------------------------------
# Core evaluation
# ------------------------------------------------------------
def evaluate(model, X, Y, device="cpu", batch_size=256):
    """
    Compute all absolute and relative errors for the 1‑source regression model.
    """

    model.eval()

    xy_all = []
    rho_all = []
    phi_all = []
    I_all = []
    rho_true_all = []
    I_true_all = []

    ds = TensorDataset(
        torch.from_numpy(X).to(torch.float64),
        torch.from_numpy(Y).to(torch.float64)
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for Xb, Yb in loader:
            Xb = Xb.to(device, dtype=torch.float32)
            Yb = Yb.to(device, dtype=torch.float32)

            out = model(Xb)
            xy_pred = out["xy"]
            I_pred  = out["I"].view(-1)

            x_true = Yb[:, 0]
            y_true = Yb[:, 1]
            I_true = Yb[:, 2]

            x_pred = xy_pred[:, 0]
            y_pred = xy_pred[:, 1]

            # Absolute xy error
            xy_err = torch.sqrt((x_pred - x_true)**2 + (y_pred - y_true)**2)

            # True polar
            rho_true = torch.sqrt(x_true**2 + y_true**2)
            phi_true = torch.atan2(y_true, x_true)

            # Pred polar
            rho_pred = torch.sqrt(x_pred**2 + y_pred**2)
            phi_pred = torch.atan2(y_pred, x_pred)

            # Absolute errors
            rho_err = torch.abs(rho_pred - rho_true)

            dphi = phi_pred - phi_true
            dphi = (dphi + math.pi) % (2 * math.pi) - math.pi
            phi_err = torch.abs(dphi)

            I_err = torch.abs(I_pred - I_true)

            xy_all.append(xy_err.cpu())
            rho_all.append(rho_err.cpu())
            phi_all.append(phi_err.cpu())
            I_all.append(I_err.cpu())
            rho_true_all.append(rho_true.cpu())
            I_true_all.append(I_true.cpu())

    # Convert to numpy
    xy_all = torch.cat(xy_all).numpy()
    rho_all = torch.cat(rho_all).numpy()
    phi_all = torch.cat(phi_all).numpy()
    I_all   = torch.cat(I_all).numpy()
    rho_true_all = torch.cat(rho_true_all).numpy()
    I_true_all   = torch.cat(I_true_all).numpy()

    # Relative errors
    rho_rel = rho_all / np.maximum(rho_true_all, 1e-12)
    I_rel   = I_all   / np.maximum(I_true_all, 1e-12)

    metrics = {
        "xy_abs": _summaries(xy_all),
        "rho_abs": _summaries(rho_all),
        "phi_abs": _summaries(phi_all),
        "I_abs": _summaries(I_all),
        "rho_rel": _summaries(rho_rel),
        "I_rel": _summaries(I_rel),
    }

    # Status
    status = "passed"
    if any(_is_bad(v) for v in metrics.values()):
        status = "failed"

    return {
        "module": "accuracy",
        "status": status,
        "metrics": metrics
    }
