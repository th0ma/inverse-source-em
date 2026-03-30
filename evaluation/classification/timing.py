"""
Timing Evaluation for Source-Count Classification Model.

This module measures inference performance of the trained classifier under
different batch sizes. It computes:

    - inference time per sample (ms)
    - throughput (samples per second)
    - batch-size scaling (time per sample for batch sizes 1, 8, 32, 128)

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

Notes:
    - A warmup pass is performed before timing to stabilize GPU/CPU performance.
    - Timing is synchronized on CUDA devices to ensure accurate measurement.
    - Status is 'failed' only if NaN/Inf values appear in the metrics.
"""


import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _measure_time(model, loader, device):
    """Measure total inference time over a dataloader."""
    model.eval()

    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    with torch.no_grad():
        for X, _ in loader:
            X = X.to(device)
            _ = model(X)

    if device == "cuda":
        torch.cuda.synchronize()

    t1 = time.perf_counter()
    return t1 - t0


def _is_bad(x):
    """Recursively check for NaN/Inf in scalars, lists, arrays, or dicts."""
    if isinstance(x, dict):
        return any(_is_bad(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return any(_is_bad(v) for v in x)
    try:
        arr = np.asarray(x, dtype=float)
        return np.isnan(arr).any() or np.isinf(arr).any()
    except Exception:
        return False


def evaluate(model, X_test, y_test, device="cpu"):
    """
    Evaluate inference speed of the classifier.
    """
    # Warmup batch
    warmup_ds = TensorDataset(
        torch.from_numpy(X_test[:64]).to(torch.float64),
        torch.from_numpy(y_test[:64]).to(torch.long)
    )
    warmup_loader = DataLoader(warmup_ds, batch_size=64, shuffle=False)

    # Warmup pass
    with torch.no_grad():
        for X, _ in warmup_loader:
            X = X.to(device)
            _ = model(X)

    # Main timing
    batch_sizes = [1, 8, 32, 128]
    batch_scaling = {}

    N = len(X_test)

    for bs in batch_sizes:
        ds = TensorDataset(
            torch.from_numpy(X_test).to(torch.float64),
            torch.from_numpy(y_test).to(torch.long)
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

    # Status check
    status = "passed"
    for v in metrics.values():
        if _is_bad(v):
            status = "failed"

    return {
        "module": "timing",
        "status": status,
        "metrics": metrics
    }
