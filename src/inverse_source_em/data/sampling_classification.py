"""
Sampling utilities for the source-count classification dataset.

This module provides:
    - sample_single_source
    - sample_sources

These functions generate random source configurations inside a disk
of radius R, with optional intensity ranges.

Public API:
    sample_single_source(...)
    sample_sources(...)
"""

import numpy as np


def sample_single_source(R, rng, I_range=(1.0, 1.0)):
    """
    Sample a single source inside a disk of radius R.

    Parameters
    ----------
    R : float
        Physical radius.
    rng : np.random.Generator
        Random number generator.
    I_range : tuple (I_min, I_max)
        Range for source intensity.

    Returns
    -------
    rho : float
        Radial coordinate (0 ≤ rho ≤ R).
    phi : float
        Angular coordinate in [0, 2π).
    I : float
        Source intensity.
    """
    rho = R * np.sqrt(rng.uniform())
    phi = 2 * np.pi * rng.uniform()
    I   = rng.uniform(*I_range)
    return rho, phi, I


def sample_sources(S, R, rng, I_range=(1.0, 1.0)):
    """
    Sample S independent sources.

    Parameters
    ----------
    S : int
        Number of sources.
    R : float
        Physical radius.
    rng : np.random.Generator
        Random number generator.
    I_range : tuple
        Range for source intensity.

    Returns
    -------
    sources : list of dict
        Each dict has keys: {"rho", "phi", "I"}.
    """
    sources = []
    for _ in range(S):
        rho, phi, I = sample_single_source(R, rng, I_range)
        sources.append({"rho": rho, "phi": phi, "I": I})
    return sources
