"""
Surrogate models and unified forward API.

This subpackage provides all surrogate-based forward models used throughout
the inverse-source-em framework. These models approximate the analytical
PhysicsTM solver and enable fast, differentiable forward evaluations for:

- dataset generation
- inverse-model training
- large-scale experiments where the analytical solver is too slow

Included components
-------------------
SurrogateMLP
    Base multilayer perceptron architecture used to approximate the
    complex-valued TM-mode boundary fields.

SurrogateEM
    Unified surrogate forward model that loads trained MLPs for Esurf and Hsurf
    and exposes the same API as the analytical PhysicsTM solver.

SurrogateWrapper
    Compatibility layer that adapts a SurrogateEM instance to a fully
    PhysicsTM-compatible interface, including broadcasting and batch utilities.

Public API
----------
The following classes are re-exported at the package level:

- :class:`SurrogateMLP`
- :class:`SurrogateEM`
- :class:`SurrogateWrapper`

Notes
-----
This subpackage is used by all regression pipelines and by the dataset
generators that rely on surrogate-based forward evaluations.
"""

from .mlp_surrogate import SurrogateMLP
from .surrogate import SurrogateEM
from .surrogate_wrapper import SurrogateWrapper

__all__ = [
    "SurrogateMLP",
    "SurrogateEM",
    "SurrogateWrapper",
]
