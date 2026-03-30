"""
Periodicity Evaluation for Surrogate Models.

A physically correct TM forward model must satisfy 2π‑periodicity with respect
to the source azimuth φ_s:

    Esurf(rho, φ_s, θ) = Esurf(rho, φ_s + 2π, θ)
    Hsurf(rho, φ_s, θ) = Hsurf(rho, φ_s + 2π, θ)

This module evaluates whether the following models satisfy this periodicity:
    - PhysicsTM
    - SurrogateEM
    - SurrogateWrapper

For each model we compute:
    - E_abs: max absolute periodicity error for Esurf
    - H_abs: max absolute periodicity error for Hsurf

Notes:
    - Periodicity is tested only with respect to φ_s, not θ.
    - This test checks physical consistency, not numerical accuracy.
    - Evaluation is performed over θ ∈ [0, 2π) with fixed resolution
      (num_angles = 300).
    - The returned dictionary follows the standardized evaluation format
      used by evaluation/surrogates/run_all.py.

Returned structure:
{
    "module": "periodicity",
    "status": "passed" | "failed",
    "metrics": {
        "phys": {"E_abs": ..., "H_abs": ..., "model_passed": ...},
        "sur":  {"E_abs": ..., "H_abs": ..., "model_passed": ...},
        "wrap": {"E_abs": ..., "H_abs": ..., "model_passed": ...}
    }
}
"""


import numpy as np


def _evaluate_single_model(model, rho_s, phi_s, num_angles=300):
    """Compute periodicity errors for a single model."""
    theta = np.linspace(0, 2*np.pi, num_angles, endpoint=False)

    try:
        # Evaluate at φ_s
        E1 = model.Esurf(rho_s, phi_s, theta)
        H1 = model.Hsurf(rho_s, phi_s, theta)

        # Evaluate at φ_s + 2π
        E2 = model.Esurf(rho_s, phi_s + 2*np.pi, theta)
        H2 = model.Hsurf(rho_s, phi_s + 2*np.pi, theta)

        # Absolute errors
        E_abs = float(np.max(np.abs(E1 - E2)))
        H_abs = float(np.max(np.abs(H1 - H2)))

        return {
            "E_abs": E_abs,
            "H_abs": H_abs,
            "model_passed": True
        }

    except Exception:
        return {
            "E_abs": None,
            "H_abs": None,
            "model_passed": False
        }


def evaluate(phys, sur, wrap):
    """
    Run periodicity tests for all three models.

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

    results = {
        "module": "periodicity",
        "status": "passed",
        "metrics": {}
    }

    # Evaluate each model
    results["metrics"]["phys"] = _evaluate_single_model(phys, rho_s, phi_s)
    results["metrics"]["sur"]  = _evaluate_single_model(sur,  rho_s, phi_s)
    results["metrics"]["wrap"] = _evaluate_single_model(wrap, rho_s, phi_s)

    # Global status
    if not (
        results["metrics"]["phys"]["model_passed"]
        and results["metrics"]["sur"]["model_passed"]
        and results["metrics"]["wrap"]["model_passed"]
    ):
        results["status"] = "failed"

    return results
