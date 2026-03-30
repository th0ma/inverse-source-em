"""
Broadcasting Evaluation for Surrogate Models.

PhysicsTM does NOT support broadcasting over (rho, phi, theta).
Broadcasting is required only for:
    - SurrogateEM
    - SurrogateWrapper

This module validates that the surrogate models correctly broadcast
vectorized inputs of equal length:

    rho:   shape (N,)
    phi_s: shape (N,)
    theta: shape (N,)

Expected output:
    Esurf(rho, phi_s, theta) → shape (N,)
    Hsurf(rho, phi_s, theta) → shape (N,)

The evaluation checks that both Esurf and Hsurf return outputs with
the correct shape when given vectorized inputs.

Notes:
    - PhysicsTM is excluded because it does not implement broadcasting.
    - This test checks only API behavior (shapes), not numerical accuracy.
    - The returned dictionary follows the standardized evaluation format
      used by evaluation/surrogates/run_all.py.

Returned structure:
{
    "module": "broadcasting",
    "status": "passed" | "failed",
    "metrics": {
        "sur":  {...},
        "wrap": {...}
    }
}
"""


import numpy as np


def _evaluate_single_model(model, phys_R):
    """Run broadcasting test for a single surrogate model."""
    rho = np.array([0.3 * phys_R, 0.5 * phys_R, 0.8 * phys_R])
    phi = np.array([0.0, 1.0, 2.0])
    theta = np.array([0.1, 0.5, 1.0])

    metrics = {}

    try:
        E = model.Esurf(rho, phi, theta)
        H = model.Hsurf(rho, phi, theta)

        metrics["E_shape_ok"] = (E.shape == (3,))
        metrics["H_shape_ok"] = (H.shape == (3,))
    except Exception:
        metrics["E_shape_ok"] = False
        metrics["H_shape_ok"] = False

    metrics["model_passed"] = metrics["E_shape_ok"] and metrics["H_shape_ok"]
    return metrics


def evaluate(phys, sur, wrap):
    """
    Run broadcasting tests for surrogate models.

    Parameters
    ----------
    phys : PhysicsTM (ignored)
    sur  : SurrogateEM
    wrap : SurrogateWrapper

    Returns
    -------
    dict
        Standardized evaluation result.
    """
    results = {
        "module": "broadcasting",
        "status": "passed",
        "metrics": {}
    }

    # Evaluate surrogate models only
    results["metrics"]["sur"]  = _evaluate_single_model(sur,  phys.R)
    results["metrics"]["wrap"] = _evaluate_single_model(wrap, phys.R)

    # Global status
    if not (
        results["metrics"]["sur"]["model_passed"]
        and results["metrics"]["wrap"]["model_passed"]
    ):
        results["status"] = "failed"

    return results
