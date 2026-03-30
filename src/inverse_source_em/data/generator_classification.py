"""
Forward-model utilities for the source-count classification dataset.

This module provides the function ``forward_fields`` which computes the
multi-source boundary fields (Esurf, Hsurf) using the unified surrogate
forward model (SurrogateWrapper). The fields from all sources are summed
according to the superposition principle.

Given a list of sources, each defined by:
    {"rho": ..., "phi": ..., "I": ...}

the function computes:
    E_total(theta), H_total(theta)

and returns a feature tensor of shape (4, num_angles):

    [Re(E_total), Im(E_total), Re(H_total), Im(H_total)]

This feature representation is used as input to the source-count
classification model.
"""

import numpy as np


def forward_fields(sources, theta, sur_wrap):
    """
    Compute multi-source Esurf and Hsurf using SurrogateWrapper.

    This function evaluates the surrogate forward model for each source
    in the list and sums the resulting fields to obtain the total boundary
    fields. The output is formatted as a 4-channel feature tensor suitable
    for classification tasks.

    Parameters
    ----------
    sources : list of dict
        List of source dictionaries, each with keys:
            {"rho", "phi", "I"}
        where:
            rho : float
                Radial coordinate of the source.
            phi : float
                Angular coordinate of the source.
            I : float
                Source strength.
    theta : ndarray of shape (num_angles,)
        Observation angles in radians.
    sur_wrap : SurrogateWrapper
        Unified surrogate forward model providing:
            Esurf(rho, phi, theta)
            Hsurf(rho, phi, theta)

    Returns
    -------
    feat : ndarray of shape (4, num_angles)
        Feature tensor containing:
            [Re(E_total), Im(E_total), Re(H_total), Im(H_total)]

    Notes
    -----
    - All computations are performed in complex128 for numerical stability.
    - The final feature tensor is returned as float32 for compact storage.
    - This function is used by the classification dataset generator.
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
