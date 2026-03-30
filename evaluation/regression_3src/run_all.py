"""
run_all.py — Unified evaluation suite for 3‑source regression.

This script runs the complete evaluation pipeline for the 3‑source regression
model, including:

1. accuracy evaluation  
2. detailed error tables  
3. noise‑robustness sweep  
4. timing benchmarks  

Each evaluation sample contains three point sources with polar coordinates
$\left(\rho_1,\\,\phi_1\right)$,
$\left(\rho_2,\\,\phi_2\right)$,
$\left(\rho_3,\\,\phi_3\right)$
and strengths $I_1, I_2, I_3$, together with the boundary fields
$E_r,\\,E_i,\\,H_r,\\,H_i$ computed by the canonical surrogate models.

The script prints a clean summary of all metrics and returns a dictionary
containing the outputs of all evaluation modules.

Run as a standalone script:
    python run_all.py
"""


# ============================================================
# run_all.py — Run full evaluation suite for 3-source regression
# ============================================================

import os

from .accuracy import evaluate_accuracy
from .error_tables import evaluate_error_tables
from .noise_robustness import evaluate_noise_robustness
from .timing import evaluate_timing


def run_all():
    print("\n===============================================")
    print("   3-SOURCE REGRESSION — FULL EVALUATION SUITE")
    print("===============================================\n")

    print("1) Running accuracy evaluation...")
    acc = evaluate_accuracy()

    print("\n2) Running error tables...")
    err = evaluate_error_tables()

    print("\n3) Running noise robustness sweep...")
    noise = evaluate_noise_robustness()

    print("\n4) Running timing benchmarks...")
    timing = evaluate_timing()

    print("\n==================== SUMMARY ====================")
    print("Accuracy:")
    print(acc)

    print("\nError Tables:")
    print({
        "mean_err_A": float(err["err_A"].mean()),
        "mean_err_B": float(err["err_B"].mean()),
        "mean_err_C": float(err["err_C"].mean()),
        "mean_err_max_triplet": float(err["err_max_triplet"].mean()),
    })

    print("\nNoise Robustness:")
    for entry in noise:
        print(entry)

    print("\nTiming:")
    print(timing)

    print("\n================= EVALUATION DONE =================\n")


if __name__ == "__main__":
    run_all()
