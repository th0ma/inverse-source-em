"""
Interpolation Error Evaluation for Surrogate Models

This module evaluates the surrogate's accuracy across the full 3D input space:
    ρ_s ∈ [0.1R, 0.9R]
    φ_s ∈ [0, 2π]
    θ   ∈ [0, 2π]

We sample N random points and compute:
    ΔE = E_phys - E_sur
    ΔH = H_phys - H_sur

Relative errors:
    ERR_E = |ΔE| / (|E_phys| + 1e-12)
    ERR_H = |ΔH| / (|H_phys| + 1e-12)

Returned metrics:
- mean / median / max relative error
- 95th percentile error
- correlation of error with θ (optional)

Standardized output:
{
    "module": "interpolation",
    "status": "passed" | "failed",
    "metrics": {
        "E_rel_mean": ...,
        "E_rel_median": ...,
        "E_rel_max": ...,
        "E_rel_p95": ...,
        "H_rel_mean": ...,
        "H_rel_median": ...,
        "H_rel_max": ...,
        "H_rel_p95": ...
    }
}
"""

import numpy as np


def evaluate(phys, sur, wrap):
    """
    Run interpolation error test for SurrogateEM vs PhysicsTM.

    Parameters
    ----------
    phys : PhysicsTM
    sur  : SurrogateEM
    wrap : SurrogateWrapper (ignored here)

    Returns
    -------
    dict
        Standardized evaluation result.
    """
    N = 5000  # number of random test points

    # Random sampling
    rho_rand   = np.random.uniform(0.1 * phys.R, 0.9 * phys.R, N)
    phi_rand   = np.random.uniform(0, 2*np.pi, N)
    theta_rand = np.random.uniform(0, 2*np.pi, N)

    # Surrogate evaluation (vectorized)
    E_sur = sur.Esurf(rho_rand, phi_rand, theta_rand)
    H_sur = sur.Hsurf(rho_rand, phi_rand, theta_rand)

    # PhysicsTM evaluation (not vectorized)
    E_phys = np.array([
        phys.Esurf(rho_rand[i], phi_rand[i], theta_rand[i])
        for i in range(N)
    ])
    H_phys = np.array([
        phys.Hsurf(rho_rand[i], phi_rand[i], theta_rand[i])
        for i in range(N)
    ])

    # Absolute errors
    E_abs = np.abs(E_phys - E_sur)
    H_abs = np.abs(H_phys - H_sur)

    # Relative errors
    E_rel = E_abs / (np.abs(E_phys) + 1e-12)
    H_rel = H_abs / (np.abs(H_phys) + 1e-12)

    # Summary metrics
    metrics = {
        "E_rel_mean":   float(np.mean(E_rel)),
        "E_rel_median": float(np.median(E_rel)),
        "E_rel_max":    float(np.max(E_rel)),
        "E_rel_p95":    float(np.percentile(E_rel, 95)),

        "H_rel_mean":   float(np.mean(H_rel)),
        "H_rel_median": float(np.median(H_rel)),
        "H_rel_max":    float(np.max(H_rel)),
        "H_rel_p95":    float(np.percentile(H_rel, 95)),
    }

    # Status: fail only if NaN or inf
    status = "passed"
    if any(np.isnan(v) or np.isinf(v) for v in metrics.values()):
        status = "failed"

    return {
        "module": "interpolation",
        "status": status,
        "metrics": metrics
    }
