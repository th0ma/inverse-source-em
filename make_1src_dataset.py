#!/usr/bin/env python
# coding: utf-8

"""
Generate dataset_1src.npz for Regression Problem I (single-source localization).

This script:
- loads the canonical surrogate models
- generates N samples using the surrogate forward model
- performs 2-pass normalization
- saves the dataset in data/regression/dataset_1src.npz

Run:
    python make_1src_dataset.py
"""

import os
from inverse_source_em.data.generator_1src import build_dataset_1src, save_dataset_1src


# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------

N_SOURCES = 20_000
NUM_ANGLES = 30

# Correct paths for surrogate models
PATH_E = "models/surrogate_Esurf.pth"
PATH_H = "models/surrogate_Hsurf.pth"

# Output dataset path
SAVE_PATH = "data/regression/dataset_1src.npz"


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("\n=== Regression I: Generating dataset_1src ===")
    print(f"N_SOURCES   = {N_SOURCES}")
    print(f"NUM_ANGLES  = {NUM_ANGLES}")
    print(f"Saving to   = {SAVE_PATH}")

    # Build dataset
    X, Y, theta_obs, ymax_E, ymax_H, R = build_dataset_1src(
        n_sources=N_SOURCES,
        num_angles=NUM_ANGLES,
        path_E=PATH_E,
        path_H=PATH_H,
    )

    # Save
    save_dataset_1src(
        SAVE_PATH,
        X,
        Y,
        theta_obs,
        ymax_E,
        ymax_H,
        R,
    )

    print("\nDataset saved successfully.")
    print("Shapes:")
    print("  X:", X.shape)
    print("  Y:", Y.shape)
    print("  theta_obs:", theta_obs.shape)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
