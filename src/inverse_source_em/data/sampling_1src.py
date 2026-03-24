"""
Sampling utilities for Regression Problem I (single-source localization).

This module provides:
- Uniform sampling of a single source in the disk (rho, phi → x, y)
- Uniform sampling of observation angles
- Reproducible random generation

The sampling logic is intentionally simple and physics‑consistent.
"""

import numpy as np


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Set numpy random seed for reproducible sampling.
    """
    np.random.seed(seed)


# ------------------------------------------------------------
# Source sampling
# ------------------------------------------------------------

def sample_sources_1src(
    N: int,
    R: float = 1.0,
    rho_min: float = 0.05,
    rho_max: float = 0.95,
):
    """
    Sample N single sources uniformly in the disk (annulus).

    Parameters
    ----------
    N : int
        Number of samples.
    R : float
        Disk radius.
    rho_min, rho_max : float
        Minimum and maximum radial distance as fractions of R.

    Returns
    -------
    rho : (N,) ndarray
    phi : (N,) ndarray
    x   : (N,) ndarray
    y   : (N,) ndarray
    """
    # Uniform in area: sample r^2 uniformly
    r2_min = (rho_min / R) ** 2
    r2_max = (rho_max / R) ** 2

    rho = np.sqrt(np.random.uniform(r2_min, r2_max, size=N)) * R
    phi = np.random.uniform(0.0, 2 * np.pi, size=N)

    # Convert to Cartesian
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    return rho, phi, x, y


# ------------------------------------------------------------
# Observation angles
# ------------------------------------------------------------

def sample_angles(M: int):
    """
    Uniform sampling of M observation angles in [0, 2π).

    Parameters
    ----------
    M : int
        Number of observation angles.

    Returns
    -------
    theta_obs : (M,) ndarray
    """
    return np.linspace(0.0, 2 * np.pi, M, endpoint=False)
