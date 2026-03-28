"""
Sampling utilities for the 3-source regression pipeline.

This module provides:
- Geometry curriculum levels
- Canonical ordering for 3 sources
- Angular difference helper
- Sampling function with geometric constraints
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


# ============================================================
# 2. Angular difference helper (wrapped)
# ============================================================

def ang_diff(a, b):
    """
    Compute wrapped angular difference |a - b| in [-pi, pi].
    """
    return np.abs(((a - b + np.pi) % (2 * np.pi)) - np.pi)


# ============================================================
# 3. Canonical ordering for 3 sources
# ============================================================

def canonical_order_three(rho1, phi1, rho2, phi2, rho3, phi3):
    """
    Canonical ordering for 3 sources:
    Sort by (rho, phi) lexicographically.
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

    Returns:
        (rho1, phi1, rho2, phi2, rho3, phi3)
        in canonical order.
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
