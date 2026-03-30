"""
Spectral Evaluation for Surrogate Models.

This module compares the Fourier spectra of Esurf(θ) and Hsurf(θ) between
PhysicsTM (reference) and SurrogateEM. The comparison is performed on a
fixed θ‑slice:

    θ ∈ [0, 2π), sampled at 1024 points
    ρ_s = 0.6R
    φ_s = 1.0

For each field we compute:

    - L2 norm of FFT difference
    - relative spectral error
    - spectral centroid difference (frequency‑weighted average)
    - high‑frequency energy ratio (surrogate vs physics)

These metrics quantify how well the surrogate reproduces the spectral
content of the true fields, including modal amplitudes and distribution
of energy across frequencies.

Notes:
    - SurrogateWrapper is ignored because it is an interface layer.
    - This test evaluates spectral fidelity, not pointwise accuracy.
    - FFTs are complex‑valued; all metrics use magnitudes |·|.
    - The returned dictionary follows the standardized evaluation format
      used by evaluation/surrogates/run_all.py.

Returned structure:
{
    "module": "spectral",
    "status": "passed" | "failed",
    "metrics": {
        "E_fft_l2": ...,
        "E_fft_rel": ...,
        "E_centroid_diff": ...,
        "E_highfreq_ratio": ...,
        "H_fft_l2": ...,
        "H_fft_rel": ...,
        "H_centroid_diff": ...,
        "H_highfreq_ratio": ...
    }
}
"""


import numpy as np


def _spectral_metrics(E_phys, E_sur):
    """Compute spectral metrics for a single field (E or H)."""
    # FFTs
    F_phys = np.fft.fft(E_phys)
    F_sur  = np.fft.fft(E_sur)

    # L2 norm of FFT difference
    fft_l2 = float(np.linalg.norm(F_phys - F_sur))

    # Relative spectral error
    fft_rel = float(
        fft_l2 / (np.linalg.norm(F_phys) + 1e-12)
    )

    # Spectral centroid (weighted average frequency)
    k = np.fft.fftfreq(len(E_phys), d=1/len(E_phys))
    mag_phys = np.abs(F_phys)
    mag_sur  = np.abs(F_sur)

    centroid_phys = float(np.sum(np.abs(k) * mag_phys) / (np.sum(mag_phys) + 1e-12))
    centroid_sur  = float(np.sum(np.abs(k) * mag_sur)  / (np.sum(mag_sur)  + 1e-12))

    centroid_diff = abs(centroid_phys - centroid_sur)

    # High-frequency energy ratio (|k| > 0.25 * k_max)
    k_abs = np.abs(k)
    high_mask = k_abs > 0.25 * np.max(k_abs)

    hf_phys = np.sum(mag_phys[high_mask])
    hf_sur  = np.sum(mag_sur[high_mask])

    highfreq_ratio = float(hf_sur / (hf_phys + 1e-12))

    return {
        "fft_l2": fft_l2,
        "fft_rel": fft_rel,
        "centroid_diff": centroid_diff,
        "highfreq_ratio": highfreq_ratio
    }


def evaluate(phys, sur, wrap):
    """
    Run spectral comparison tests for Esurf and Hsurf.

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
    # θ-slice
    theta = np.linspace(0, 2*np.pi, 1024)
    rho0 = 0.6 * phys.R
    phi0 = 1.0

    # PhysicsTM fields
    E_phys = phys.Esurf(rho0, phi0, theta)
    H_phys = phys.Hsurf(rho0, phi0, theta)

    # Surrogate fields
    E_sur = sur.Esurf(rho0, phi0, theta)
    H_sur = sur.Hsurf(rho0, phi0, theta)

    # Compute spectral metrics
    E_metrics = _spectral_metrics(E_phys, E_sur)
    H_metrics = _spectral_metrics(H_phys, H_sur)

    # Merge metrics
    metrics = {
        "E_fft_l2":          E_metrics["fft_l2"],
        "E_fft_rel":         E_metrics["fft_rel"],
        "E_centroid_diff":   E_metrics["centroid_diff"],
        "E_highfreq_ratio":  E_metrics["highfreq_ratio"],

        "H_fft_l2":          H_metrics["fft_l2"],
        "H_fft_rel":         H_metrics["fft_rel"],
        "H_centroid_diff":   H_metrics["centroid_diff"],
        "H_highfreq_ratio":  H_metrics["highfreq_ratio"],
    }

    # Status: fail only if NaN or inf
    status = "passed"
    if any(np.isnan(v) or np.isinf(v) for v in metrics.values()):
        status = "failed"

    return {
        "module": "spectral",
        "status": status,
        "metrics": metrics
    }
