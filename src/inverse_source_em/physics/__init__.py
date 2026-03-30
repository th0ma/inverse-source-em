"""
Physics-based forward models for electromagnetic scattering.

This subpackage contains analytical forward solvers used throughout the
inverse-source-em framework. These solvers provide ground-truth boundary
fields for dataset generation, surrogate training, and evaluation of
inverse models.

Currently included:

- ``PhysicsTM``  
  Analytical TM (transverse-magnetic) forward solver for a dielectric
  cylinder excited by a single internal line source. Computes the
  complex-valued boundary fields (E_z, H_φ) on the cylinder surface
  using a truncated cylindrical-harmonic expansion.

Public API
----------
The following classes are intended for external use and are re-exported
at the package level:

- :class:`PhysicsTM`

Notes
-----
This subpackage is imported by the surrogate models
(`mlp_surrogate.py`, `surrogate.py`, `surrogate_wrapper.py`) and forms
the canonical physics reference for the entire project.
"""

from .physics_tm import PhysicsTM

__all__ = [
    "PhysicsTM",
]

