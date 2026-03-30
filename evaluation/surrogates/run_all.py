"""
Unified Evaluation Runner for Surrogate Models.

This script loads the three forward models:
    - PhysicsTM (analytical reference model)
    - SurrogateEM (neural surrogate for Esurf and Hsurf)
    - SurrogateWrapper (API‑compatible adapter around SurrogateEM)

and executes all evaluation modules in this package.

Each evaluation module returns a standardized dictionary:

    {
        "module": "<name>",
        "status": "passed" | "failed",
        "metrics": {...}
    }

The unified runner:
    - loads all models,
    - executes each evaluation module in sequence,
    - collects their outputs,
    - and aggregates everything into a single result dictionary.

The final output has the form:

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

Notes:
    - overall_status is "failed" if any module reports failure.
    - SurrogateEM weights are loaded from the directory specified by
      `models_path` (default: "models").
    - This runner is the recommended entry point for full surrogate
      evaluation and regression testing.
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
