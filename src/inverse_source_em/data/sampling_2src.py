"""
Sampling utilities for the two-source inverse EM problem.

This module provides the sampling routines used to generate training data
for the 2‑source localization regression problem. It includes:

- Canonical ordering of the two sources
- Geometric sampling with minimum radial and angular separation
- Conversion between normalized polar and Cartesian coordinates

The sampling logic enforces physically meaningful configurations and ensures
that the two sources are not too close to each other, improving dataset
diversity and numerical stability.
"""

import numpy as np


# ------------------------------------------------------------
# 1. Canonical ordering
# ------------------------------------------------------------
def canonical_order(rho1, phi1, rho2, phi2):
    """
    Enforce a consistent ordering of the two sources.

    The canonical ordering rule is:
        - Sort primarily by rho (ascending)
        - If rho values are equal (within tolerance), sort by phi (ascending)

    This ensures that the dataset has a unique representation for each
    two-source configuration, avoiding label permutation ambiguity.

    Parameters
    ----------
    rho1, rho2 : float
        Normalized radii τ = ρ/R in [0, 1].
    phi1, phi2 : float
        Angular coordinates in radians.

    Returns
    -------
    rho1_c, phi1_c, rho2_c, phi2_c : floats
        Canonically ordered polar coordinates.

    Notes
    -----
    - np.isclose is used to handle floating-point equality of radii.
    """
    if rho1 > rho2 or (np.isclose(rho1, rho2) and phi1 > phi2):
        return rho2, phi2, rho1, phi1
    return rho1, phi1, rho2, phi2


# ------------------------------------------------------------
# 2. Sampling two sources with geometric priors
# ------------------------------------------------------------
def sample_two_sources(
    rho_min=0.05,
    rho_max=0.95,
    min_delta_rho=0.05,
    min_delta_phi_deg=5.0
):
    """
    Sample two sources in normalized polar coordinates with geometric priors.

    The sampling domain is:
        τ ∈ [rho_min, rho_max]
        φ ∈ [-π, π]

    Two constraints are enforced:
        - Minimum radial separation |τ1 - τ2| ≥ min_delta_rho
        - Minimum angular separation Δφ ≥ min_delta_phi_deg

    Parameters
    ----------
    rho_min, rho_max : float
        Allowed range for normalized radius τ.
    min_delta_rho : float
        Minimum allowed radial separation.
    min_delta_phi_deg : float
        Minimum allowed angular separation in degrees.

    Returns
    -------
    rho1, phi1, rho2, phi2 : floats
        Canonically ordered normalized polar coordinates.

    Notes
    -----
    - Angular separation is computed using wrapped difference in [-π, π].
    - Sampling repeats until both geometric constraints are satisfied.
    """
    min_delta_phi = np.deg2rad(min_delta_phi_deg)

    while True:
        rho1 = np.random.uniform(rho_min, rho_max)
        rho2 = np.random.uniform(rho_min, rho_max)
        phi1 = np.random.uniform(-np.pi, np.pi)
        phi2 = np.random.uniform(-np.pi, np.pi)

        # Radial separation
        delta_rho = abs(rho1 - rho2)

        # Angular separation (wrapped to [-π, π])
        delta_phi = abs(((phi1 - phi2 + np.pi) % (2*np.pi)) - np.pi)

        if delta_rho < min_delta_rho or delta_phi < min_delta_phi:
            continue

        # Enforce canonical ordering
        return canonical_order(rho1, phi1, rho2, phi2)


# ------------------------------------------------------------
# 3. Polar → Cartesian (normalized)
# ------------------------------------------------------------
def polar_to_cart_normalized(rho_norm, phi):
    """
    Convert normalized polar coordinates (τ = ρ/R, φ)
    into normalized Cartesian coordinates (x/R, y/R).

    Parameters
    ----------
    rho_norm : float
        Normalized radius τ in [0, 1].
    phi : float
        Angle in radians.

    Returns
    -------
    x_norm, y_norm : floats
        Normalized Cartesian coordinates.

    Notes
    -----
    - This conversion is used to produce normalized labels for training.
    - No physical radius R is needed since τ = ρ/R is already normalized.
    """
    x = rho_norm * np.cos(phi)
    y = rho_norm * np.sin(phi)
    return x, y
