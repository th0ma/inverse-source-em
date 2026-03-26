#!/usr/bin/env python
# coding: utf-8

"""
make_2src_dataset.py

Top-level script to generate the two-source dataset using the
modular pipeline in inverse_source_em.data.dataset_2src.

Run from project root:
    python make_2src_dataset.py
"""

import numpy as np
import os

from inverse_source_em.data.dataset_2src import build_dataset_2src


# ------------------------------------------------------------
# 1. Dataset configuration
# ------------------------------------------------------------
NUM_ANGLES = 30
N_SAMPLES = 20_000

theta = np.linspace(0, 2*np.pi, NUM_ANGLES, endpoint=False)

OUT_DIR = "./data/regression_2src"

# Surrogate paths (relative to project root)
PATH_E = "models/surrogate_Esurf.pth"
PATH_H = "models/surrogate_Hsurf.pth"

DATASET_NAME = f"dataset_2src_fullfield_{NUM_ANGLES}obs_{N_SAMPLES//1000}k.npz"


# ------------------------------------------------------------
# 2. Generate dataset
# ------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== Two-Source Dataset Generation ===\n")
    print("NUM_ANGLES:", NUM_ANGLES)
    print("N_SAMPLES :", N_SAMPLES)
    print("OUT_DIR   :", OUT_DIR)
    print("Dataset   :", DATASET_NAME)

    dataset_path = build_dataset_2src(
        N_samples=N_SAMPLES,
        theta=theta,
        path_E=PATH_E,
        path_H=PATH_H,
        test_size=0.30,
        out_dir=OUT_DIR,
        dataset_name=DATASET_NAME
    )

    print("\nDataset saved to:")
    print(dataset_path)
    print("\nDone.")
