"""
Generate surrogate datasets (Esurf, Hsurf) for the inverse_source_em package.

Usage:
    python make_surrogate_dataset.py
        -> uses defaults: 10000 sources, 72 angles

    python make_surrogate_dataset.py -s 20000 -a 128
        -> custom number of sources and angles

This script:
    1. Instantiates the PhysicsTM solver
    2. Generates Esurf and Hsurf surrogate datasets
    3. Saves them into data/surrogate/ as .npz files
    4. Each .npz file contains:
         - X: array of shape (N*M, 5)
         - Y: array of shape (N*M, 2)
    5. Prints a validation report (shapes, dtypes, min/max)

Notes:
    - Must be executed from the project root directory.
    - Requires the inverse_source_em package to be installed (pip install -e .).
"""


import argparse
import numpy as np

from inverse_source_em.physics.physics_tm import PhysicsTM
from inverse_source_em.data.generator_surrogate import SurrogateDataGenerator


# ------------------------------------------------------------
# Validation report
# ------------------------------------------------------------
def report_dataset(path: str):
    print(f"\n[Report] {path}")
    data = np.load(path)

    X = data["X"]
    Y = data["Y"]

    print("  X shape:", X.shape)
    print("  Y shape:", Y.shape)

    print("  X dtype:", X.dtype)
    print("  Y dtype:", Y.dtype)

    print("  X min/max:", float(X.min()), float(X.max()))
    print("  Y min/max:", float(Y.min()), float(Y.max()))

    # Basic sanity checks
    assert X.ndim == 2 and X.shape[1] == 5, "X must be (N*M, 3)"
    assert Y.ndim == 2 and Y.shape[1] == 2, "Y must be (N*M, 2)"
    assert np.isfinite(X).all(), "X contains NaN or Inf"
    assert np.isfinite(Y).all(), "Y contains NaN or Inf"

    print("  ✔ Dataset OK")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate surrogate datasets (Esurf, Hsurf).")

    parser.add_argument(
        "-s", "--sources",
        type=int,
        default=10000,
        help="Number of sources to sample (default: 10000)"
    )

    parser.add_argument(
        "-a", "--angles",
        type=int,
        default=72,
        help="Number of observation angles (default: 72)"
    )

    args = parser.parse_args()

    # Instantiate physics solver
    phys = PhysicsTM()

    # Instantiate generator
    gen = SurrogateDataGenerator(
        physics=phys,
        output_dir="data/surrogate",
        num_angles=args.angles,
        rho_min=0.01,
        rho_max=0.99
    )

    # Generate Esurf
    X_E, Y_E = gen.generate(kind="Esurf", n_sources=args.sources)
    gen.save_npz(X_E, Y_E, "Esurf.npz")

    # Generate Hsurf
    X_H, Y_H = gen.generate(kind="Hsurf", n_sources=args.sources)
    gen.save_npz(X_H, Y_H, "Hsurf.npz")

    print("\nAll surrogate datasets saved in data/surrogate/")

    # Validation reports
    report_dataset("data/surrogate/Esurf.npz")
    report_dataset("data/surrogate/Hsurf.npz")


if __name__ == "__main__":
    main()
