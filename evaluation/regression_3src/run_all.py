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
