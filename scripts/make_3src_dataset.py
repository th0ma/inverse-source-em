"""
make_3src_dataset.py

Top‑level script for generating the full 3‑source regression datasets
(stages 1–8). This script is a thin wrapper around the modular pipeline:

    inverse_source_em.data.generator_3src.create_3src_datasets

Each dataset contains three point sources with polar coordinates
$\left(\rho_1,\\,\phi_1\right)$,
$\left(\rho_2,\\,\phi_2\right)$,
$\left(\rho_3,\\,\phi_3\right)$
and strengths $I_1, I_2, I_3$, together with the corresponding boundary
fields $E_r,\\,E_i,\\,H_r,\\,H_i$ computed by the canonical surrogate models.

Run from the project root:
    python make_3src_dataset.py
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
