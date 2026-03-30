"""
Noise‑robustness evaluation for the 2‑source regression model.

This module handles permutation invariance by evaluating both possible
assignments between predicted and true sources:

1. $\left(\text{predA}\rightarrow\text{trueA},\\,\text{predB}\rightarrow\text{trueB}\right)$
2. $\left(\text{predA}\rightarrow\text{trueB},\\,\text{predB}\rightarrow\text{trueA}\right)$

For each sample, the assignment with the smaller total distance error is kept.

Gaussian noise is added to the input fields using feature‑wise scaling:
$$
X_{\mathrm{noisy}} = X + \sigma\,X_{\mathrm{std}}\,\varepsilon,
\qquad
\varepsilon \sim \mathcal{N}\left(0,\\,1\right)
$$

For each noise level $\sigma$, the module computes:
- mean of the per‑sample maximum distance error
- 99th‑percentile of the per‑sample maximum distance error

Returned structure:
{
    "module": "noise_robustness",
    "status": "passed" | "failed",
    "metrics": [
        {
            "noise_sigma": ...,
            "mean_dmax": ...,
            "p99_dmax": ...
        },
        ...
    ]
}
"""


import numpy as np
import torch


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
def evaluate(model, X, Y, noise_levels, device="cpu"):
    model.eval()

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    # Feature-wise std for noise scaling
    X_std = X.std(axis=0) + 1e-8

    results = []

    with torch.no_grad():
        for sigma in noise_levels:

            # Add Gaussian noise
            noise = np.random.randn(*X.shape).astype(np.float32) * (sigma * X_std)
            X_noisy = X + noise

            # Convert to tensor
            Xb = torch.from_numpy(X_noisy).to(device, dtype=torch.float32)
            Yb = torch.from_numpy(Y).to(device, dtype=torch.float32)

            # Predict
            preds = model(Xb).cpu().numpy()

            # Split into A and B
            predA = preds[:, 0:2]
            predB = preds[:, 2:4]

            trueA = Y[:, 0:2]
            trueB = Y[:, 2:4]

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

            dmax = np.maximum(dA, dB)

            results.append({
                "noise_sigma": float(sigma),
                "mean_dmax": float(dmax.mean()),
                "p99_dmax": float(np.percentile(dmax, 99)),
            })

    status = "passed"
    for row in results:
        if any(_is_bad(v) for v in row.values()):
            status = "failed"

    return {
        "module": "noise_robustness",
        "status": status,
        "metrics": results
    }
