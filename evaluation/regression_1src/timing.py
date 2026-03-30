"""
Timing evaluation for the 1‑source regression model.

This module measures inference speed by computing:

- time per sample (in milliseconds), using batch size 32 as reference:
  $$
  \text{time\_per\_sample\_ms} = 1000 \times \text{time\_per\_sample}
  $$

- throughput (samples per second):
  $$
  \text{throughput} = \dfrac{1}{\text{time\_per\_sample}}
  $$

- batch‑scaling performance for batch sizes
  $\left(1,\,8,\,32,\,128\right)$:
  time per sample as a function of batch size

Returned structure:
{
    "module": "timing",
    "status": "passed" | "failed",
    "metrics": {
        "time_per_sample_ms": ...,
        "throughput_samples_per_sec": ...,
        "batch_scaling": {
            1: ...,
            8: ...,
            32: ...,
            128: ...
        }
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
# Measure inference time over a dataloader
# ------------------------------------------------------------
def _measure_time(model, loader, device):
    model.eval()

    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    with torch.no_grad():
        for Xb, _ in loader:
            Xb = Xb.to(device, dtype=torch.float32)
            _ = model(Xb)

    if device == "cuda":
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    return t1 - t0


# ------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------
def evaluate(model, X, Y, device="cpu", batch_sizes=None):
    """
    Evaluate inference speed of the 1‑source regression model.
    """

    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 128]

    # Warmup
    warmup_ds = TensorDataset(
        torch.from_numpy(X[:64]).to(torch.float64),
        torch.from_numpy(Y[:64]).to(torch.float64)
    )
    warmup_loader = DataLoader(warmup_ds, batch_size=64, shuffle=False)

    with torch.no_grad():
        for Xb, _ in warmup_loader:
            Xb = Xb.to(device, dtype=torch.float32)
            _ = model(Xb)

    # Main timing
    N = len(X)
    batch_scaling = {}

    for bs in batch_sizes:
        ds = TensorDataset(
            torch.from_numpy(X).to(torch.float64),
            torch.from_numpy(Y).to(torch.float64)
        )
        loader = DataLoader(ds, batch_size=bs, shuffle=False)

        total_time = _measure_time(model, loader, device)
        time_per_sample = total_time / N

        batch_scaling[bs] = float(time_per_sample)

    # Use batch size 32 as reference
    ref_time = batch_scaling[32]
    time_per_sample_ms = ref_time * 1000.0
    throughput = 1.0 / ref_time

    metrics = {
        "time_per_sample_ms": float(time_per_sample_ms),
        "throughput_samples_per_sec": float(throughput),
        "batch_scaling": batch_scaling,
    }

    status = "passed"
    if any(_is_bad(v) for v in metrics.values()):
        status = "failed"

    return {
        "module": "timing",
        "status": status,
        "metrics": metrics
    }
