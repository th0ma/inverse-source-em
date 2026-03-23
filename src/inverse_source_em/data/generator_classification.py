"""
Forward-model utilities for the source-count classification dataset.

This module provides:
    - forward_fields: multi-source superposition using SurrogateWrapper

Given a list of sources (each with rho, phi, I), the unified surrogate
forward model is used to compute:
    E_total(theta), H_total(theta)

The output is a feature tensor of shape (4, num_angles):
    [Re(E), Im(E), Re(H), Im(H)]

Public API:
    forward_fields(...)
"""

import numpy as np


def forward_fields(sources, theta, sur_wrap):
    """
    Compute multi-source Esurf and Hsurf using SurrogateWrapper.

    Parameters
    ----------
    sources : list of dict
        Each dict has keys {"rho", "phi", "I"}.
    theta : ndarray, shape (num_angles,)
        Observation angles.
    sur_wrap : SurrogateWrapper
        Unified surrogate forward model.

    Returns
    -------
    feat : ndarray, shape (4, num_angles)
        Feature tensor:
            [E_real, E_imag, H_real, H_imag]
    """

    # Initialize accumulators
    E_total = np.zeros_like(theta, dtype=np.complex128)
    H_total = np.zeros_like(theta, dtype=np.complex128)

    # Superposition of fields
    for src in sources:
        rho = src["rho"]
        phi = src["phi"]
        I   = src["I"]

        # SurrogateWrapper returns complex arrays
        E = sur_wrap.Esurf(rho, phi, theta) * I
        H = sur_wrap.Hsurf(rho, phi, theta) * I

        E_total += E
        H_total += H

    # Build feature tensor (4, num_angles)
    feat = np.stack([
        E_total.real,
        E_total.imag,
        H_total.real,
        H_total.imag
    ], axis=0)

    return feat.astype(np.float32)
