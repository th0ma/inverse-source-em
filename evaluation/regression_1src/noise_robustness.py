"""
Noise robustness evaluation for 1‑source regression model.

For each noise level:
- Adds Gaussian noise to X
- Computes absolute and relative errors:
    xy_abs, rho_abs, phi_abs, I_abs, rho_rel, I_rel
- Computes summary statistics (mean, p50, p90, p99)

Returned structure:
{
    "module": "noise_robustness",
    "status": "passed" | "failed",
    "metrics": {
        "noise_levels": [...],
        "summaries": {
            noise_level: {
                "xy_abs": {...},
                "rho_abs": {...},
                "phi_abs": {...},
                "I_abs": {...},
                "rho_rel": {...},
                "I_rel": {...}
            },
            ...
        }
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
# Summary statistics
# ------------------------------------------------------------
def _summaries(arr):
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


# ------------------------------------------------------------
# Add Gaussian noise
# ------------------------------------------------------------
def _add_noise(X, sigma, rng):
    if sigma == 0.0:
        return X.copy()
    noise = rng.normal(loc=0.0, scale=sigma, size=X.shape)
    return X + noise


# ------------------------------------------------------------
# Core evaluation
# ------------------------------------------------------------
def evaluate(
    model,
    X,
    Y,
    device="cpu",
    noise_levels=None,
    batch_size=256
):
    """
    Evaluate robustness of the 1‑source regression model under Gaussian noise.
    """

    if noise_levels is None:
        noise_levels = [0.0, 0.01, 0.03, 0.05, 0.10]

    model.eval()
    rng = np.random.default_rng(12345)

    results = {}

    for sigma in noise_levels:

        # Add noise
        X_noisy = _add_noise(X, sigma, rng)

        xy_all = []
        rho_all = []
        phi_all = []
        I_all = []
        rho_true_all = []
        I_true_all = []

        ds = TensorDataset(
            torch.from_numpy(X_noisy).to(torch.float64),
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

        # Store summaries
        results[sigma] = {
            "xy_abs": _summaries(xy_all),
            "rho_abs": _summaries(rho_all),
            "phi_abs": _summaries(phi_all),
            "I_abs": _summaries(I_all),
            "rho_rel": _summaries(rho_rel),
            "I_rel": _summaries(I_rel),
        }

    # Status
    status = "passed"
    for sigma in noise_levels:
        if any(_is_bad(v) for v in results[sigma].values()):
            status = "failed"

    return {
        "module": "noise_robustness",
        "status": status,
        "metrics": {
            "noise_levels": noise_levels,
            "summaries": results
        }
    }
