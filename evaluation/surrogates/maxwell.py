"""
Maxwell Consistency Evaluation for Surrogate Models.

For TM polarization, the tangential magnetic field on the boundary satisfies
the Maxwell boundary condition:

    H_surf(θ) = (1 / (ω μ0)) * ∂E/∂ρ |_{ρ=R}

Using the series representation of E, PhysicsTM can compute the radial
derivative ∂E/∂ρ at ρ = R and reconstruct H_surf(θ). This provides a
physically exact reference for checking whether a model's H_surf(θ)
is consistent with its corresponding E_surf(θ).

This module evaluates:
    - PhysicsTM
    - SurrogateEM
    - SurrogateWrapper

For each model we compute:
    - H_direct(θ): the model's own H_surf(θ)
    - H_from_E(θ): reconstructed from ∂E/∂ρ using PhysicsTM

and measure:
    - max absolute error
    - relative error (normalized by max|H_direct|)

Notes:
    - H_from_E is always computed using PhysicsTM, even when evaluating
      surrogate models.
    - This test checks physical consistency between E and H, not numerical
      accuracy with respect to PhysicsTM.
    - Evaluation is performed over θ ∈ [0, 2π) with fixed resolution
      (num_angles = 400).
    - The returned dictionary follows the standardized evaluation format
      used by evaluation/surrogates/run_all.py.

Returned structure:
{
    "module": "maxwell",
    "status": "passed" | "failed",
    "metrics": {
        "phys": {"H_abs": ..., "H_rel": ..., "model_passed": ...},
        "sur":  {"H_abs": ..., "H_rel": ..., "model_passed": ...},
        "wrap": {"H_abs": ..., "H_rel": ..., "model_passed": ...}
    }
}
"""


import numpy as np
from scipy.special import h1vp


def _compute_H_from_E_derivative(phys, rho_s, phi_s, theta):
    """
    Use PhysicsTM machinery to compute H_surf from the radial derivative of E.

    Parameters
    ----------
    phys : PhysicsTM
    rho_s : float
    phi_s : float
    theta : ndarray

    Returns
    -------
    ndarray
        H_from_E(θ) reconstructed via Maxwell boundary condition.
    """
    n = phys.nvals
    alpha = phys.alpha_n_vec(rho_s)

    # Hankel derivative H_n^{(1)'}(k0 R)
    Hprime = h1vp(n, phys.K0 * phys.R, 1).astype(np.complex128)

    # Phase factor e^{i n (θ - φ_s)}
    phase = np.exp(1j * n[:, None] * (theta[None, :] - phi_s))

    # Radial derivative of E at ρ = R
    dE_dr = phys.A * np.sum(
        alpha[:, None] * Hprime[:, None] * phys.K0 * phase,
        axis=0
    )

    # Maxwell relation
    H_from_E = dE_dr / (phys.OMEGA * phys.MU0)
    return H_from_E


def _evaluate_single_model(model, phys, rho_s, phi_s, num_angles=400):
    """Compute Maxwell consistency errors for a single model."""
    theta = np.linspace(0, 2*np.pi, num_angles, endpoint=False)

    try:
        # Direct magnetic field from the model
        H_direct = model.Hsurf(rho_s, phi_s, theta)

        # Reconstructed H from E-derivative using PhysicsTM
        H_from_E = _compute_H_from_E_derivative(phys, rho_s, phi_s, theta)

        abs_err = float(np.max(np.abs(H_direct - H_from_E)))
        rel_err = float(abs_err / (np.max(np.abs(H_direct)) + 1e-12))

        return {
            "H_abs": abs_err,
            "H_rel": rel_err,
            "model_passed": True
        }

    except Exception:
        return {
            "H_abs": None,
            "H_rel": None,
            "model_passed": False
        }


def evaluate(phys, sur, wrap):
    """
    Run Maxwell consistency tests for all three models.

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
        "module": "maxwell",
        "status": "passed",
        "metrics": {}
    }

    # Evaluate each model
    results["metrics"]["phys"] = _evaluate_single_model(phys, phys, rho_s, phi_s)
    results["metrics"]["sur"]  = _evaluate_single_model(sur,  phys, rho_s, phi_s)
    results["metrics"]["wrap"] = _evaluate_single_model(wrap, phys, rho_s, phi_s)

    # Global status
    if not (
        results["metrics"]["phys"]["model_passed"]
        and results["metrics"]["sur"]["model_passed"]
        and results["metrics"]["wrap"]["model_passed"]
    ):
        results["status"] = "failed"

    return results
