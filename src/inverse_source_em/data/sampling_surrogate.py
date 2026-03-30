"""
Sampling utilities for surrogate dataset generation.

This module provides helper functions used by the surrogate dataset generator.
It defines two sampling routines:

- ``sample_sources(N, R, rho_min, rho_max)``  
  Samples N source locations uniformly in the annulus:
      rho ∈ [rho_min * R, rho_max * R]
      phi ∈ [0, 2π)
  The radial coordinate is sampled using sqrt() to ensure uniform area density.

- ``sample_angles(M)``  
  Returns M uniformly spaced observation angles in [0, 2π).

These functions are used by SurrogateDataGenerator to construct the training
dataset for the surrogate MLP models.
"""

import numpy as np


def sample_sources(N, R=1.0, rho_min=0.05, rho_max=0.95):
    """
    Sample N source positions uniformly in an annular region.

    The sampling domain is:
        rho ∈ [rho_min * R, rho_max * R]
        phi ∈ [0, 2π)

    The radial coordinate is sampled using sqrt() so that the resulting
    distribution is uniform with respect to area (i.e., uniform density
    in the disk annulus).

    Parameters
    ----------
    N : int
        Number of source positions to sample.
    R : float, optional
        Cylinder radius. Default is 1.0.
    rho_min : float, optional
        Minimum normalized radial coordinate (relative to R).
    rho_max : float, optional
        Maximum normalized radial coordinate (relative to R).

    Returns
    -------
    rho : ndarray of shape (N,)
        Radial coordinates of sampled sources.
    phi : ndarray of shape (N,)
        Angular coordinates of sampled sources in radians.

    Notes
    -----
    - The returned rho values lie in [rho_min * R, rho_max * R].
    - The sampling is rotationally symmetric and area-uniform.
    """
    r2_min = (rho_min / R)**2
    r2_max = (rho_max / R)**2

    rho = np.sqrt(np.random.uniform(r2_min, r2_max, size=N)) * R
    phi = np.random.uniform(0.0, 2*np.pi, size=N)
    return rho, phi


def sample_angles(M):
    """
    Uniformly sample M observation angles in [0, 2π).

    Parameters
    ----------
    M : int
        Number of observation angles.

    Returns
    -------
    ndarray of shape (M,)
        Uniformly spaced angles in radians.
    """
    return np.linspace(0.0, 2*np.pi, M, endpoint=False)
