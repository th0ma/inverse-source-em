"""
Generate classification datasets (1–5 sources) for the inverse_source_em package.

Usage:
    python make_classification_dataset.py
        -> uses defaults: 30 angles, balanced classes

    python make_classification_dataset.py -a 64 -c 8000 8000 8000 16000 16000
        -> custom angles + custom samples per class

This script:
    1. Instantiates the surrogate forward model
    2. Generates multi-source Esurf/Hsurf feature tensors
    3. Saves train/val/test splits + scalers
    4. Prints a validation report
"""

import argparse
import numpy as np
import os

from inverse_source_em.physics.physics_tm import PhysicsTM
from inverse_source_em.surrogate.surrogate import SurrogateEM
from inverse_source_em.surrogate.surrogate_wrapper import SurrogateWrapper

from inverse_source_em.data import generate_classification_dataset


# ------------------------------------------------------------
# Validation report
# ------------------------------------------------------------
def report_dataset(path: str):
    print(f"\n[Report] {path}")
    data = np.load(path)

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]

    X_test = data["X_test"]
    y_test = data["y_test"]

    print("  X_train:", X_train.shape, X_train.dtype)
    print("  y_train:", y_train.shape, y_train.dtype)

    print("  X_val:  ", X_val.shape, X_val.dtype)
    print("  y_val:  ", y_val.shape, y_val.dtype)

    print("  X_test: ", X_test.shape, X_test.dtype)
    print("  y_test: ", y_test.shape, y_test.dtype)

    # Basic sanity checks
    assert X_train.ndim == 3 and X_train.shape[1] == 4, "X must be (N, 4, num_angles)"
    assert np.isfinite(X_train).all(), "X_train contains NaN or Inf"
    assert np.isfinite(X_val).all(), "X_val contains NaN or Inf"
    assert np.isfinite(X_test).all(), "X_test contains NaN or Inf"

    print("  ✔ Dataset OK")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate classification dataset (1–5 sources).")

    parser.add_argument(
        "-a", "--angles",
        type=int,
        default=30,
        help="Number of observation angles (default: 30)"
    )

    parser.add_argument(
        "-c", "--counts",
        nargs=5,
        type=int,
        default=[10000, 10000, 10000, 20000, 20000],
        help="Samples per class S=1..5 (default: 10000 10000 10000 20000 20000)"
    )

    parser.add_argument(
        "-o", "--out",
        type=str,
        default="data/classification",
        help="Output directory (default: data/classification)"
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Instantiate surrogate forward model
    # ------------------------------------------------------------
    phys = PhysicsTM()
    R = phys.R

    # Surrogate paths (package-internal)
    # The SurrogateEM constructor already knows how to load its models
    sur = SurrogateEM()
    sur_wrap = SurrogateWrapper(sur)

    # Observation angles
    theta = np.linspace(0, 2*np.pi, args.angles, endpoint=False)

    print("\n=== Classification Dataset Generation ===")
    print("Angles:", args.angles)
    print("Samples per class:", args.counts)
    print("Output directory:", args.out)
    print("R =", R)

    # ------------------------------------------------------------
    # Generate dataset
    # ------------------------------------------------------------
    out_file = generate_classification_dataset(
        out_dir=args.out,
        sur_wrap=sur_wrap,
        R=R,
        theta=theta,
        samples_per_class=args.counts,
        seed=2025
    )

    print("\nDataset saved to:", out_file)

    # ------------------------------------------------------------
    # Validation report
    # ------------------------------------------------------------
    report_dataset(out_file)


if __name__ == "__main__":
    main()
