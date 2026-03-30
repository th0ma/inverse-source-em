"""
Sampling utilities for the 3-source regression pipeline.

This module provides the sampling logic used to generate training data
for the three-source inverse EM problem. It includes:

- Geometry curriculum levels (progressively relaxing constraints)
- Canonical ordering for 3 sources
- Angular difference helper with wrapping
- Sampling function with geometric constraints per stage

The sampling strategy ensures:
- Physically meaningful configurations
- Minimum radial and angular separation between sources
- Deterministic ordering to avoid permutation ambiguity
"""

import numpy as np


# ============================================================
# 1. Geometry curriculum levels (same as successful experiment)
# ============================================================

GEOMETRY_LEVELS = {
    1: {"dr_min": 0.10, "dphi_min_deg": 40},
    2: {"dr_min": 0.08, "dphi_min_deg": 25},
    3: {"dr_min": 0.08, "dphi_min_deg": 20},
    4: {"dr_min": 0.07, "dphi_min_deg": 15},
    5: {"dr_min": 0.06, "dphi_min_deg": 10},
    6: {"dr_min": 0.05, "dphi_min_deg": 8},
    7: {"dr_min": 0.05, "dphi_min_deg": 6},
    8: {"dr_min": 0.05, "dphi_min_deg": 5},
}
"""
Dictionary defining the geometry curriculum.

Each stage specifies:
- dr_min : minimum radial separation between any two sources
- dphi_min_deg : minimum angular separation in degrees

These constraints gradually relax from stage 1 → stage 8.
"""


# ============================================================
# 2. Angular difference helper (wrapped)
# ============================================================

def ang_diff(a, b):
    """
    Compute wrapped angular difference |a - b| in [-π, π].

    Parameters
    ----------
    a, b : float
        Angles in radians.

    Returns
    -------
    float
        Wrapped absolute angular difference.

    Notes
    -----
    - Uses modulo arithmetic to ensure periodicity.
    """
    return np.abs(((a - b + np.pi) % (2 * np.pi)) - np.pi)


# ============================================================
# 3. Canonical ordering for 3 sources
# ============================================================

def canonical_order_three(rho1, phi1, rho2, phi2, rho3, phi3):
    """
    Canonical ordering for 3 sources.

    The three sources are sorted lexicographically by (rho, phi):

        (rho1, phi1), (rho2, phi2), (rho3, phi3)
        → sorted by increasing rho, then increasing phi

    This removes permutation ambiguity in the dataset.

    Parameters
    ----------
    rho1, rho2, rho3 : float
        Normalized radii τ = ρ/R.
    phi1, phi2, phi3 : float
        Angular coordinates in radians.

    Returns
    -------
    rho1_c, phi1_c, rho2_c, phi2_c, rho3_c, phi3_c : floats
        Canonically ordered coordinates.
    """
    pts = [(rho1, phi1), (rho2, phi2), (rho3, phi3)]
    pts_sorted = sorted(pts, key=lambda t: (t[0], t[1]))
    (rho1, phi1), (rho2, phi2), (rho3, phi3) = pts_sorted
    return rho1, phi1, rho2, phi2, rho3, phi3


# ============================================================
# 4. Sampling function for 3 sources
# ============================================================

def sample_three_sources(geom_level):
    """
    Sample 3 sources satisfying the geometry constraints of a given stage.

    Parameters
    ----------
    geom_level : int
        Geometry stage index (1–8). Determines:
            - minimum radial separation
            - minimum angular separation

    Returns
    -------
    rho1, phi1, rho2, phi2, rho3, phi3 : floats
        Canonically ordered normalized polar coordinates.

    Notes
    -----
    - Radii are sampled in [0.1, 0.9] to avoid degenerate near-center cases.
    - Angular coordinates are sampled in [-π, π].
    - Sampling repeats until all pairwise constraints are satisfied.
    """
    params = GEOMETRY_LEVELS[geom_level]
    dr_min = params["dr_min"]
    dphi_min = np.deg2rad(params["dphi_min_deg"])

    while True:
        # Sample 3 random sources
        rho = np.random.uniform(0.1, 0.9, size=3)
        phi = np.random.uniform(-np.pi, np.pi, size=3)

        # Geometry constraints
        ok = True
        for i in range(3):
            for j in range(i + 1, 3):
                if abs(rho[i] - rho[j]) < dr_min:
                    ok = False
                if ang_diff(phi[i], phi[j]) < dphi_min:
                    ok = False

        if not ok:
            continue

        # Canonical ordering
        return canonical_order_three(
            rho[0], phi[0],
            rho[1], phi[1],
            rho[2], phi[2]
        )
