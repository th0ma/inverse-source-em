"""
generator_2src.py

Feature construction for the two-source inverse EM problem.
Builds the 120-dimensional feature vector using the surrogate forward model.

Dependencies:
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
from inverse_source_em.surrogate.surrogate_wrapper import SurrogateWrapper


# ------------------------------------------------------------
# 1. Surrogate loader
# ------------------------------------------------------------
def load_surrogate(path_E, path_H):
    """
    Load the surrogate forward model (Esurf, Hsurf).

    Returns
    -------
    sur_wrap : SurrogateWrapper
        Wrapper providing Esurf() and Hsurf() methods.
    R : float
        Physical radius (cm) extracted from surrogate metadata.
    """
    sur = SurrogateEM(path_E=path_E, path_H=path_H)
    sur_wrap = SurrogateWrapper(sur)
    R = sur.R  # physical radius from surrogate metadata
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
        Normalized radii τ = ρ/R in [0,1].
    phi1, phi2 : float
        Angles in radians.
    theta : ndarray
        Observation angles (num_angles,)
    sur_wrap : SurrogateWrapper
        Provides Esurf() and Hsurf() methods.
    R : float
        Physical radius (cm).

    Returns
    -------
    X : ndarray of shape (4 * num_angles,)
        Concatenation of:
            [Ere(θ), Eim(θ), Hre(θ), Him(θ)]
    """

    # Convert normalized radii to physical units (cm)
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
    Generate one training example (X, Y) for the two-source problem.

    Returns
    -------
    X : ndarray, shape (120,)
    Y : ndarray, shape (4,)
        Normalized Cartesian coordinates:
            [x1/R, y1/R, x2/R, y2/R]
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
