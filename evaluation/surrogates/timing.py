"""
Timing Benchmark Evaluation for Surrogate Models

This module compares the execution time of:
- PhysicsTM
- SurrogateEM
- SurrogateWrapper

We measure:
- average runtime per call (ms)
- speedup factors relative to PhysicsTM

Returned metrics:
{
    "module": "timing",
    "status": "passed" | "failed",
    "metrics": {
        "phys_time_ms": ...,
        "sur_time_ms": ...,
        "wrap_time_ms": ...,
        "sur_speedup": ...,
        "wrap_speedup": ...
    }
}
"""

import numpy as np
import time


def _benchmark_model(model, rho_s, phi_s, theta, N=2000):
    """
    Benchmark a model by running N forward evaluations and measuring time.

    Parameters
    ----------
    model : PhysicsTM or SurrogateEM or SurrogateWrapper
    rho_s : float
    phi_s : float
    theta : float
    N : int
        Number of repeated calls.

    Returns
    -------
    float
        Average time per call in milliseconds.
    """
    start = time.perf_counter()

    for _ in range(N):
        model.Esurf(rho_s, phi_s, theta)

    end = time.perf_counter()
    avg_time_ms = (end - start) * 1000.0 / N
    return float(avg_time_ms)


def evaluate(phys, sur, wrap):
    """
    Run timing benchmarks for all three models.

    Parameters
    ----------
    phys : PhysicsTM
    sur  : SurrogateEM
    wrap : SurrogateWrapper

    Returns
    -------
    dict
        Standardized evaluation result.
    """
    rho_s = 0.6 * phys.R
    phi_s = 1.0
    theta = 0.7

    # Benchmark each model
    phys_time = _benchmark_model(phys, rho_s, phi_s, theta)
    sur_time  = _benchmark_model(sur,  rho_s, phi_s, theta)
    wrap_time = _benchmark_model(wrap, rho_s, phi_s, theta)

    # Speedup factors
    sur_speedup  = float(phys_time / (sur_time + 1e-12))
    wrap_speedup = float(phys_time / (wrap_time + 1e-12))

    metrics = {
        "phys_time_ms": phys_time,
        "sur_time_ms":  sur_time,
        "wrap_time_ms": wrap_time,
        "sur_speedup":  sur_speedup,
        "wrap_speedup": wrap_speedup
    }

    # Status: fail only if NaN or inf
    status = "passed"
    if any(np.isnan(v) or np.isinf(v) for v in metrics.values()):
        status = "failed"

    return {
        "module": "timing",
        "status": status,
        "metrics": metrics
    }
