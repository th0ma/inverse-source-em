"""
Feature construction utilities for the two-source inverse EM problem.

This module provides the core routines used to generate training samples
for the 2‑source localization regression dataset. It includes:

1. Loading the surrogate forward model (Esurf, Hsurf)
2. Constructing the 120‑dimensional feature vector:
       X = [Re(E), Im(E), Re(H), Im(H)] over all observation angles
3. Generating a single training example (X, Y), where:
       Y = [x1/R, y1/R, x2/R, y2/R]

Dependencies
------------
- sampling_2src.py
- surrogate.surrogate (SurrogateEM)
- surrogate.surrogate_wrapper (SurrogateWrapper)
"""

import numpy as np
from inverse_source_em.data.sampling_2src import (
    sample_two_sources,
    polar_to_cart_normalized
)

from inverse_source_em.surrogate.surrogate import SurrogateEM
    # Provides Esurf() and Hsurf()
from inverse_source_em.surrogate.surrogate_wrapper import SurrogateWrapper
    # PhysicsTM-compatible interface


# ------------------------------------------------------------
# 1. Surrogate loader
# ------------------------------------------------------------
def load_surrogate(path_E, path_H):
    """
    Load the surrogate forward model (Esurf, Hsurf) and wrap it in a
    PhysicsTM-compatible interface.

    Parameters
    ----------
    path_E : str
        Path to the trained surrogate model for Esurf.
    path_H : str
        Path to the trained surrogate model for Hsurf.

    Returns
    -------
    sur_wrap : SurrogateWrapper
        Wrapper providing Esurf() and Hsurf() with broadcasting support.
    R : float
        Physical radius extracted from surrogate metadata.

    Notes
    -----
    - SurrogateEM loads the trained MLPs.
    - SurrogateWrapper provides the unified API used by all dataset builders.
    """
    sur = SurrogateEM(path_E=path_E, path_H=path_H)
    sur_wrap = SurrogateWrapper(sur)
    R = sur.R
    return sur_wrap, R


# ------------------------------------------------------------
# 2. Build 120-dimensional feature vector
# ------------------------------------------------------------
def build_features_two_sources(
    rho1_norm, phi1,
    rho2_norm, phi2,
    theta,
    sur_wrap,
    R
):
    """
    Construct the 120-dimensional feature vector for two sources.

    Parameters
    ----------
    rho1_norm, rho2_norm : float
        Normalized radii τ = ρ/R in [0, 1].
    phi1, phi2 : float
        Angular coordinates of the two sources (radians).
    theta : ndarray of shape (num_angles,)
        Observation angles.
    sur_wrap : SurrogateWrapper
        Unified surrogate forward model.
    R : float
        Physical radius.

    Returns
    -------
    X : ndarray of shape (4 * num_angles,)
        Concatenated feature vector:
            [Re(E_tot), Im(E_tot), Re(H_tot), Im(H_tot)]

    Notes
    -----
    - The surrogate forward model is evaluated separately for each source.
    - The fields are summed according to the superposition principle.
    - The output is flattened into a 1D feature vector.
    """

    # Convert normalized radii to physical units
    rho1_phys = rho1_norm * R
    rho2_phys = rho2_norm * R

    # Compute fields for each source
    E1 = sur_wrap.Esurf(rho1_phys, phi1, theta)
    E2 = sur_wrap.Esurf(rho2_phys, phi2, theta)
    H1 = sur_wrap.Hsurf(rho1_phys, phi1, theta)
    H2 = sur_wrap.Hsurf(rho2_phys, phi2, theta)

    # Total fields
    E_tot = E1 + E2
    H_tot = H1 + H2

    # Extract real/imag components
    Ere = E_tot.real
    Eim = E_tot.imag
    Hre = H_tot.real
    Him = H_tot.imag

    # Concatenate into a single feature vector
    X = np.concatenate([Ere, Eim, Hre, Him], axis=0)

    return X


# ------------------------------------------------------------
# 3. High-level sample builder (one training example)
# ------------------------------------------------------------
def generate_sample(theta, sur_wrap, R):
    """
    Generate one training example (X, Y) for the two-source inverse EM problem.

    Parameters
    ----------
    theta : ndarray of shape (num_angles,)
        Observation angles.
    sur_wrap : SurrogateWrapper
        Unified surrogate forward model.
    R : float
        Physical radius.

    Returns
    -------
    X : ndarray of shape (4 * num_angles,)
        Feature vector containing:
            [Re(E_tot), Im(E_tot), Re(H_tot), Im(H_tot)]
    Y : ndarray of shape (4,)
        Normalized Cartesian coordinates:
            [x1/R, y1/R, x2/R, y2/R]

    Notes
    -----
    - Source positions are sampled using sample_two_sources().
    - Labels are normalized Cartesian coordinates for stable training.
    """

    # 1. Sample geometry
    rho1, phi1, rho2, phi2 = sample_two_sources()

    # 2. Build features
    X = build_features_two_sources(
        rho1, phi1, rho2, phi2,
        theta, sur_wrap, R
    )

    # 3. Build normalized Cartesian labels
    x1, y1 = polar_to_cart_normalized(rho1, phi1)
    x2, y2 = polar_to_cart_normalized(rho2, phi2)

    Y = np.array([x1, y1, x2, y2], dtype=np.float64)

    return X, Y
