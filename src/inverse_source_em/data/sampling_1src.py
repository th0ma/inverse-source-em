"""
Sampling utilities for Regression Problem I (single-source localization).

This module provides the sampling routines used to generate training data
for the 1‑source localization regression problem. It includes:

- Uniform sampling of a single source inside a disk (annulus)
- Conversion from polar (rho, phi) to Cartesian (x, y)
- Uniform sampling of observation angles
- Optional reproducible seeding

The sampling logic is intentionally simple, physics‑consistent, and matches
the assumptions used in the analytical and surrogate forward models.
"""

import numpy as np


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Set NumPy's global random seed for reproducible sampling.

    Parameters
    ----------
    seed : int
        Seed value for NumPy's RNG.

    Notes
    -----
    - This affects all subsequent NumPy random operations.
    - For per-function reproducibility, prefer using np.random.Generator.
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
    Sample N single sources uniformly in an annular region of a disk.

    The sampling domain is:
        rho ∈ [rho_min * R, rho_max * R]
        phi ∈ [0, 2π)

    The radial coordinate is sampled using sqrt() so that the resulting
    distribution is uniform with respect to area (i.e., uniform density
    in the disk annulus).

    Parameters
    ----------
    N : int
        Number of samples to generate.
    R : float, optional
        Disk radius. Default is 1.0.
    rho_min, rho_max : float, optional
        Minimum and maximum normalized radial distance (fractions of R).

    Returns
    -------
    rho : ndarray of shape (N,)
        Radial coordinates of sampled sources.
    phi : ndarray of shape (N,)
        Angular coordinates in radians.
    x : ndarray of shape (N,)
        Cartesian x-coordinates of sampled sources.
    y : ndarray of shape (N,)
        Cartesian y-coordinates of sampled sources.

    Notes
    -----
    - The sampling is rotationally symmetric and area-uniform.
    - Cartesian coordinates are computed via:
          x = rho * cos(phi)
          y = rho * sin(phi)
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
    Uniformly sample M observation angles in [0, 2π).

    Parameters
    ----------
    M : int
        Number of observation angles.

    Returns
    -------
    theta_obs : ndarray of shape (M,)
        Uniformly spaced angles in radians.

    Notes
    -----
    - The angles are returned in ascending order.
    - endpoint=False ensures periodicity without duplication at 2π.
    """
    return np.linspace(0.0, 2 * np.pi, M, endpoint=False)
