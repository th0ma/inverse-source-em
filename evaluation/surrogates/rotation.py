"""
Rotation Invariance Evaluation for Surrogate Models

A physically correct TM forward model must satisfy rotational symmetry:

    E(θ; φ_s + α)  ≈  E(θ - α; φ_s)
    H(θ; φ_s + α)  ≈  H(θ - α; φ_s)

In Fourier domain this corresponds to:
    F_rot[n] = F[n] * exp(-i n α)

This module computes:
- absolute rotation error
- relative rotation error

for:
- PhysicsTM
- SurrogateEM
- SurrogateWrapper

Returns standardized metrics:
{
    "module": "rotation",
    "status": "passed" | "failed",
    "metrics": {
        "phys": {...},
        "sur": {...},
        "wrap": {...}
    }
}
"""

import numpy as np


def _evaluate_single_model(model, rho_s, phi_s, alpha=0.7, num_angles=1024):
    """Compute rotation invariance errors for a single model."""
    theta = np.linspace(0, 2*np.pi, num_angles, endpoint=False)

    try:
        # Base and rotated fields
        E_base = model.Esurf(rho_s, phi_s, theta)
        E_rot  = model.Esurf(rho_s, phi_s + alpha, theta)

        # FFT of base field
        F = np.fft.fft(E_base)

        # Mode indices
        k = np.fft.fftfreq(num_angles, d=1/num_angles)

        # Apply rotation in Fourier domain
        phase = np.exp(-1j * k * alpha)
        F_rot = F * phase

        # Expected rotated field
        E_ref = np.fft.ifft(F_rot)

        # Errors
        E_abs = float(np.max(np.abs(E_rot - E_ref)))
        E_rel = float(E_abs / (np.max(np.abs(E_ref)) + 1e-12))

        return {
            "E_abs": E_abs,
            "E_rel": E_rel,
            "model_passed": True
        }

    except Exception:
        return {
            "E_abs": None,
            "E_rel": None,
            "model_passed": False
        }


def evaluate(phys, sur, wrap):
    """
    Run rotation invariance tests for all three models.

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
        "module": "rotation",
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
