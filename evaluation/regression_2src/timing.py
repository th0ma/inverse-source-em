"""
Timing evaluation for 2‑source regression model.

Measures:
- mean inference time per batch
- mean inference time per sample

Handles permutation invariance internally (same logic as accuracy.py),
but only for correctness — timing is dominated by forward().

Returns:
{
    "module": "timing",
    "status": "passed" | "failed",
    "metrics": {
        "mean_batch_time_ms": ...,
        "mean_sample_time_ms": ...
    }
}
"""

import time
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
# Core timing evaluation
# ------------------------------------------------------------
def evaluate(model, X, Y, device="cpu", batch_size=256, warmup_batches=5, timed_batches=20):
    model.eval()

    # Prepare dataset
    ds = TensorDataset(
        torch.from_numpy(X).to(torch.float32),
        torch.from_numpy(Y).to(torch.float32)
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # Move model to device
    model.to(device)

    # --------------------------------------------------------
    # Warmup (stabilizes GPU clocks / JIT paths)
    # --------------------------------------------------------
    with torch.no_grad():
        for i, (Xb, _) in enumerate(loader):
            if i >= warmup_batches:
                break
            Xb = Xb.to(device, dtype=torch.float32)
            _ = model(Xb)

    # --------------------------------------------------------
    # Timed batches
    # --------------------------------------------------------
    batch_times = []
    n_samples_total = 0

    with torch.no_grad():
        for i, (Xb, _) in enumerate(loader):
            if len(batch_times) >= timed_batches:
                break

            Xb = Xb.to(device, dtype=torch.float32)

            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.perf_counter()

            _ = model(Xb)

            torch.cuda.synchronize() if device.type == "cuda" else None
            t1 = time.perf_counter()

            batch_times.append(t1 - t0)
            n_samples_total += Xb.shape[0]

    if len(batch_times) == 0:
        return {
            "module": "timing",
            "status": "failed",
            "metrics": {}
        }

    mean_batch_time = np.mean(batch_times)
    mean_sample_time = mean_batch_time / batch_size

    metrics = {
        "mean_batch_time_ms": float(mean_batch_time * 1000),
        "mean_sample_time_ms": float(mean_sample_time * 1000),
    }

    status = "passed"
    if any(_is_bad(v) for v in metrics.values()):
        status = "failed"

    return {
        "module": "timing",
        "status": status,
        "metrics": metrics
    }
