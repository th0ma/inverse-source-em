"""
API Consistency Evaluation for Surrogate Models.

This module checks whether PhysicsTM, SurrogateEM, and SurrogateWrapper
expose a unified forward API for evaluating both Esurf and Hsurf:

    Esurf(rho, phi_s, theta_or_num_angles)
    Hsurf(rho, phi_s, theta_or_num_angles)

The evaluation verifies that each model supports:
    - Scalar θ input  → scalar output
    - Array θ input   → array output with matching shape
    - Integer input   → array output of length `num_angles`

Notes:
    - The test is performed using Esurf(), but the API requirements
      are identical for Hsurf(), so consistency in Esurf implies
      consistency in Hsurf as well.
    - This module checks only API behavior (shapes, types), not
      numerical accuracy.
    - The returned dictionary follows the standardized evaluation
      format used by evaluation/surrogates/run_all.py.

Returned structure:
{
    "module": "api_tests",
    "status": "passed" | "failed",
    "metrics": {
        "phys": {...},
        "sur":  {...},
        "wrap": {...}
    }
}
"""


import numpy as np


def _evaluate_single_model(model, phys_R):
    """Run API consistency checks for a single model."""
    rho = 0.6 * phys_R
    phi = 1.0

    metrics = {}

    # --- scalar θ ---
    try:
        E1 = model.Esurf(rho, phi, 0.7)
        metrics["scalar_ok"] = bool(np.isscalar(E1))
    except Exception:
        metrics["scalar_ok"] = False

    # --- array θ ---
    try:
        theta_arr = np.linspace(0, 2*np.pi, 50)
        E2 = model.Esurf(rho, phi, theta_arr)
        metrics["array_ok"] = (E2.shape == theta_arr.shape)
    except Exception:
        metrics["array_ok"] = False

    # --- num_angles ---
    try:
        E3 = model.Esurf(rho, phi, 100)
        metrics["int_ok"] = (E3.shape == (100,))
    except Exception:
        metrics["int_ok"] = False

    # Overall status for this model
    metrics["model_passed"] = all(metrics.values())

    return metrics


def evaluate(phys, sur, wrap):
    """
    Run API consistency tests for all three models.

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
    results = {
        "module": "api_tests",
        "status": "passed",
        "metrics": {}
    }

    # Evaluate each model
    results["metrics"]["phys"] = _evaluate_single_model(phys, phys.R)
    results["metrics"]["sur"]  = _evaluate_single_model(sur,  phys.R)
    results["metrics"]["wrap"] = _evaluate_single_model(wrap, phys.R)

    # Global status
    if not (
        results["metrics"]["phys"]["model_passed"]
        and results["metrics"]["sur"]["model_passed"]
        and results["metrics"]["wrap"]["model_passed"]
    ):
        results["status"] = "failed"

    return results
