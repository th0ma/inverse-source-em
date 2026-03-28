"""
Script: make_3src_dataset.py
Generates the full 3-source regression dataset (stages 1-8).

This script is a thin wrapper around:
    inverse_source_em.data.generator_3src.create_3src_datasets
"""

import os
from inverse_source_em.data.generator_3src import create_3src_datasets


def main():

    # Output directory for 3-source datasets
    OUT_DIR = os.path.join("data", "regression_3src")

    # Surrogate model paths (same pattern as 1src/2src)
    PATH_E = os.path.join("models", "surrogate_Esurf.pth")
    PATH_H = os.path.join("models", "surrogate_Hsurf.pth")

    # Optional normalized surrogate data (if available)
    DATA_E = None
    DATA_H = None

    # Geometry stages
    STAGES = [1, 2, 3, 4, 5, 6, 7, 8]

    # Samples per stage (same as 2src)
    N_SAMPLES = 70_000

    print("\n===============================================")
    print("Generating 3-source regression datasets")
    print("===============================================\n")

    create_3src_datasets(
        out_dir=OUT_DIR,
        path_E=PATH_E,
        path_H=PATH_H,
        data_E=DATA_E,
        data_H=DATA_H,
        stages=STAGES,
        num_samples_per_stage=N_SAMPLES,
        num_angles=30
    )

    print("\nAll 3-source datasets generated successfully.\n")


if __name__ == "__main__":
    main()
