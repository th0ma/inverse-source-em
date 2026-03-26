"""
Surrogate model evaluation suite.

This package contains modular evaluation components for validating:
- PhysicsTM (analytical model)
- SurrogateEM (neural surrogate)
- SurrogateWrapper (API adapter)

Each module exposes a function:

    evaluate(phys, sur, wrap) -> dict

which returns standardized metrics in the form:

{
    "module": "<module_name>",
    "status": "passed" | "failed",
    "metrics": { ... }
}

No plots are generated here — only numerical metrics.
"""

from .api_tests import evaluate as evaluate_api
from .broadcasting import evaluate as evaluate_broadcasting
from .periodicity import evaluate as evaluate_periodicity
from .rotation import evaluate as evaluate_rotation
from .maxwell import evaluate as evaluate_maxwell
from .error_maps import evaluate as evaluate_error_maps
from .interpolation import evaluate as evaluate_interpolation
from .spectral import evaluate as evaluate_spectral
from .timing import evaluate as evaluate_timing

__all__ = [
    "evaluate_api",
    "evaluate_broadcasting",
    "evaluate_periodicity",
    "evaluate_rotation",
    "evaluate_maxwell",
    "evaluate_error_maps",
    "evaluate_interpolation",
    "evaluate_spectral",
    "evaluate_timing",
]
