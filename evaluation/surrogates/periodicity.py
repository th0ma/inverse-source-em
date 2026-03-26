"""
Periodicity Evaluation for Surrogate Models

A physically correct TM forward model must satisfy 2π-periodicity:

    Esurf(rho, phi_s, θ) = Esurf(rho, phi_s + 2π, θ)
    Hsurf(rho, phi_s, θ) = Hsurf(rho, phi_s + 2π, θ)

This module computes the maximum absolute periodicity error for:
- PhysicsTM
- SurrogateEM
- SurrogateWrapper

Returns standardized metrics:
{
    "module": "periodicity",
    "status": "passed" | "failed",
    "metrics": {
        "phys": {...},
        "sur": {...},
        "wrap": {...}
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
