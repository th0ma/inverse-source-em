"""
Unified Evaluation Runner for Surrogate Models

This script loads:
- PhysicsTM (analytical model)
- SurrogateEM (neural surrogate)
- SurrogateWrapper (API adapter)

and runs all evaluation modules in this package.

Each module returns:
{
    "module": "<name>",
    "status": "passed" | "failed",
    "metrics": {...}
}

The runner aggregates all results into a single dictionary:
{
    "overall_status": "passed" | "failed",
    "results": {
        "api_tests": {...},
        "broadcasting": {...},
        "periodicity": {...},
        "rotation": {...},
        "maxwell": {...},
        "error_maps": {...},
        "interpolation": {...},
        "spectral": {...},
        "timing": {...}
    }
}
"""

import os
import numpy as np

from inverse_source_em.physics import PhysicsTM
from inverse_source_em.surrogate import SurrogateEM, SurrogateWrapper

# Import evaluation modules
from . import (
    evaluate_api,
    evaluate_broadcasting,
    evaluate_periodicity,
    evaluate_rotation,
    evaluate_maxwell,
    evaluate_error_maps,
    evaluate_interpolation,
    evaluate_spectral,
    evaluate_timing,
)


def run_all(models_path="models"):

    """
    Run all surrogate evaluation modules.

    Parameters
    ----------
    models_path : str
        Path to directory containing surrogate model .pth files.

    Returns
    -------
    dict
        Aggregated evaluation results.
    """
    # ------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------
    phys = PhysicsTM()

    sur = SurrogateEM(
        path_E=os.path.join(models_path, "surrogate_Esurf.pth"),
        path_H=os.path.join(models_path, "surrogate_Hsurf.pth"),
        R=phys.R
    )

    wrap = SurrogateWrapper(sur)

    # ------------------------------------------------------------
    # Run all evaluation modules
    # ------------------------------------------------------------
    results = {}

    modules = [
        ("api_tests",       evaluate_api),
        ("broadcasting",    evaluate_broadcasting),
        ("periodicity",     evaluate_periodicity),
        ("rotation",        evaluate_rotation),
        ("maxwell",         evaluate_maxwell),
        ("error_maps",      evaluate_error_maps),
        ("interpolation",   evaluate_interpolation),
        ("spectral",        evaluate_spectral),
        ("timing",          evaluate_timing),
    ]

    overall_status = "passed"

    for name, func in modules:
        res = func(phys, sur, wrap)
        results[name] = res

        if res["status"] != "passed":
            overall_status = "failed"

    # ------------------------------------------------------------
    # Return aggregated results
    # ------------------------------------------------------------
    return {
        "overall_status": overall_status,
        "results": results
    }


if __name__ == "__main__":
    out = run_all()
    print("\n=== Surrogate Evaluation Summary ===\n")
    print(f"Overall status: {out['overall_status']}\n")

    for name, res in out["results"].items():
        print(f"[{name}] status: {res['status']}")
    print()
