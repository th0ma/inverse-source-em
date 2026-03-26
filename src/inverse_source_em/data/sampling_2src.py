"""
sampling_2src.py

Sampling utilities for the two-source inverse EM problem.
Provides:
    - canonical ordering of two sources
    - geometric sampling with minimum separation
    - polar <-> cartesian conversions (normalized)
"""

import numpy as np


# ------------------------------------------------------------
# 1. Canonical ordering
# ------------------------------------------------------------
def canonical_order(rho1, phi1, rho2, phi2):
    """
    Enforce a consistent ordering of the two sources.

    Rule:
        - Sort primarily by rho (ascending)
        - If rho is equal, sort by phi (ascending)

    All coordinates are in normalized units (rho in [0,1]).
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
    Sample two sources in normalized radius τ = ρ/R with geometric priors.

    Parameters
    ----------
    rho_min, rho_max : float
        Allowed range for normalized radius τ.
    min_delta_rho : float
        Minimum allowed radial separation |τ1 - τ2|.
    min_delta_phi_deg : float
        Minimum allowed angular separation in degrees.

    Returns
    -------
    rho1, phi1, rho2, phi2 : floats
        Normalized polar coordinates of the two sources.
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
        Normalized radius τ in [0,1].
    phi : float
        Angle in radians.

    Returns
    -------
    x_norm, y_norm : floats
        Normalized Cartesian coordinates.
    """
    x = rho_norm * np.cos(phi)
    y = rho_norm * np.sin(phi)
    return x, y
