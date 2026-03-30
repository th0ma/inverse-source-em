"""
Sampling utilities for the source-count classification dataset.

This module provides helper functions for generating random source
configurations inside a circular domain of radius R. These sampled
sources are used by the classification dataset generator to create
multi-source boundary field examples.

Two sampling routines are provided:

- ``sample_single_source(R, rng, I_range)``  
  Samples a single source uniformly in the disk, with optional intensity range.

- ``sample_sources(S, R, rng, I_range)``  
  Samples S independent sources and returns them as a list of dictionaries.

Public API
----------
sample_single_source(...)
sample_sources(...)
"""

import numpy as np


def sample_single_source(R, rng, I_range=(1.0, 1.0)):
    """
    Sample a single source uniformly inside a disk of radius R.

    The sampling distribution is:
        rho = R * sqrt(U),   U ~ Uniform(0, 1)
        phi ~ Uniform(0, 2π)
    which ensures uniform density over the disk area.

    Parameters
    ----------
    R : float
        Physical radius of the domain.
    rng : np.random.Generator
        Random number generator.
    I_range : tuple (I_min, I_max), optional
        Range for source intensity. Default is (1.0, 1.0), i.e. fixed intensity.

    Returns
    -------
    rho : float
        Radial coordinate of the sampled source.
    phi : float
        Angular coordinate in radians.
    I : float
        Source intensity sampled from the given range.

    Notes
    -----
    - The returned rho satisfies 0 ≤ rho ≤ R.
    - The sampling is rotationally symmetric and area-uniform.
    """
    rho = R * np.sqrt(rng.uniform())
    phi = 2 * np.pi * rng.uniform()
    I   = rng.uniform(*I_range)
    return rho, phi, I


def sample_sources(S, R, rng, I_range=(1.0, 1.0)):
    """
    Sample S independent sources inside a disk of radius R.

    Parameters
    ----------
    S : int
        Number of sources to sample.
    R : float
        Physical radius of the domain.
    rng : np.random.Generator
        Random number generator.
    I_range : tuple (I_min, I_max), optional
        Range for source intensity.

    Returns
    -------
    sources : list of dict
        A list of S dictionaries, each with keys:
            {"rho", "phi", "I"}

    Notes
    -----
    - Each source is sampled independently using ``sample_single_source``.
    - Intensities may be fixed or variable depending on I_range.
    """
    sources = []
    for _ in range(S):
        rho, phi, I = sample_single_source(R, rng, I_range)
        sources.append({"rho": rho, "phi": phi, "I": I})
    return sources
