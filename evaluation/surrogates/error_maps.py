"""
Error Map Evaluation for Surrogate Models.

This module computes ρ–θ error maps for Esurf and Hsurf on a fixed 2D grid,
but returns only summary numerical metrics (no plots, no full error arrays).

We evaluate PhysicsTM (reference) and SurrogateEM on the domain:

    ρ ∈ [0.1R, 0.9R]
    θ ∈ [0, 2π)

For each grid point we compute absolute and relative errors:

    ERR_E = |E_phys - E_sur| / (|E_phys| + 1e-12)
    ERR_H = |H_phys - H_sur| / (|H_phys| + 1e-12)

Returned metrics:
    - max absolute error
    - mean absolute error
    - max relative error
    - mean relative error

Notes:
    - SurrogateWrapper is ignored because it is an interface layer.
    - This module evaluates numerical accuracy, unlike API-based tests.
    - Grid resolution is fixed (Nr=80, Nt=200) for reproducibility.
    - The returned dictionary follows the standardized evaluation format
      used by evaluation/surrogates/run_all.py.

Returned structure:
{
    "module": "error_maps",
    "status": "passed" | "failed",
    "metrics": {
        "E_abs_max": ...,
        "E_abs_mean": ...,
        "E_rel_max": ...,
        "E_rel_mean": ...,
        "H_abs_max": ...,
        "H_abs_mean": ...,
        "H_rel_max": ...,
        "H_rel_mean": ...
    }
}
"""


import numpy as np


def evaluate(phys, sur, wrap):
    """
    Compute ρ–θ error maps and return summary metrics.

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
    # Grid resolution
    Nr = 80
    Nt = 200

    rho_grid = np.linspace(0.1 * phys.R, 0.9 * phys.R, Nr)
    theta_grid = np.linspace(0, 2*np.pi, Nt, endpoint=False)

    # Allocate error arrays
    E_abs = np.zeros((Nr, Nt))
    H_abs = np.zeros((Nr, Nt))
    E_rel = np.zeros((Nr, Nt))
    H_rel = np.zeros((Nr, Nt))

    # Compute error maps
    for i, r in enumerate(rho_grid):
        for j, th in enumerate(theta_grid):
            # PhysicsTM (reference)
            E_phys = phys.Esurf(r, 1.0, th)
            H_phys = phys.Hsurf(r, 1.0, th)

            # SurrogateEM
            E_sur = sur.Esurf(r, 1.0, th)
            H_sur = sur.Hsurf(r, 1.0, th)

            # Absolute errors
            E_abs[i, j] = np.abs(E_phys - E_sur)
            H_abs[i, j] = np.abs(H_phys - H_sur)

            # Relative errors
            E_rel[i, j] = E_abs[i, j] / (np.abs(E_phys) + 1e-12)
            H_rel[i, j] = H_abs[i, j] / (np.abs(H_phys) + 1e-12)

    # Summary metrics
    metrics = {
        "E_abs_max": float(np.max(E_abs)),
        "E_abs_mean": float(np.mean(E_abs)),
        "E_rel_max": float(np.max(E_rel)),
        "E_rel_mean": float(np.mean(E_rel)),

        "H_abs_max": float(np.max(H_abs)),
        "H_abs_mean": float(np.mean(H_abs)),
        "H_rel_max": float(np.max(H_rel)),
        "H_rel_mean": float(np.mean(H_rel)),
    }

    # Status: fail only if something is NaN or inf
    status = "passed"
    if any(np.isnan(v) or np.isinf(v) for v in metrics.values()):
        status = "failed"

    return {
        "module": "error_maps",
        "status": status,
        "metrics": metrics
    }
