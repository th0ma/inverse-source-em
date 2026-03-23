"""
Surrogate models and unified API.

Public exports:
- SurrogateMLP: the base MLP architecture
- SurrogateEM: unified surrogate forward model (Esurf, Hsurf)
- SurrogateWrapper: PhysicsTM-compatible interface layer
"""

from .mlp_surrogate import SurrogateMLP
from .surrogate import SurrogateEM
from .surrogate_wrapper import SurrogateWrapper

__all__ = [
    "SurrogateMLP",
    "SurrogateEM",
    "SurrogateWrapper",
]
