"""
Fast test for 3-source dataset generation.
Creates only 5 samples per stage.
"""

import os
import shutil
import numpy as np
from inverse_source_em.data.generator_3src import create_3src_datasets


def test_make_3src_dataset():

    OUT_DIR = os.path.join("data", "regression_3src_test")

    # Clean old test directory
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    # Run dataset creation with tiny sample count
    create_3src_datasets(
        out_dir=OUT_DIR,
        path_E=os.path.join("models", "surrogate_Esurf.pth"),
        path_H=os.path.join("models", "surrogate_Hsurf.pth"),
        stages=[1],
        num_samples_per_stage=5,
        num_angles=30
    )

    # Check files
    prefix = os.path.join(OUT_DIR, "stage_1")

    expected_files = [
        prefix + "_X_train.npy",
        prefix + "_X_test.npy",
        prefix + "_y_train.npy",
        prefix + "_y_test.npy",
        prefix + "_scaler_X.pkl",
        prefix + "_scaler_y.pkl",
    ]

    for f in expected_files:
        assert os.path.isfile(f), f"Missing file: {f}"

    X_train = np.load(prefix + "_X_train.npy")
    y_train = np.load(prefix + "_y_train.npy")

    assert X_train.shape[1] == 120
    assert y_train.shape[1] == 6
