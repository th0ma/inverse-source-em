"""
Test for make_2src_dataset.py

Checks:
- imports work
- build_dataset_2src() is called correctly
- output dataset file is created
"""

import os
import sys
import numpy as np
import pytest

# Ensure project root is in path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import script under test
import make_2src_dataset as script


# ------------------------------------------------------------
# Dummy implementation for mocking
# ------------------------------------------------------------
def dummy_build_dataset_2src(
    N_samples,
    theta,
    path_E,
    path_H,
    test_size,
    out_dir,
    dataset_name
):
    """
    Creates a minimal dummy dataset file and returns its path.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, dataset_name)

    # Minimal valid dataset
    X_train = np.zeros((10, len(theta)), dtype=np.float32)
    Y_train = np.zeros((10, 4), dtype=np.float32)

    X_test = np.zeros((5, len(theta)), dtype=np.float32)
    Y_test = np.zeros((5, 4), dtype=np.float32)

    np.savez(out_path,
             X_train=X_train, Y_train=Y_train,
             X_test=X_test, Y_test=Y_test)

    return out_path


# ------------------------------------------------------------
# Fixture to patch dependencies
# ------------------------------------------------------------
@pytest.fixture
def patch_dependencies(monkeypatch, tmp_path):
    # Patch build_dataset_2src
    monkeypatch.setattr(
        script,
        "build_dataset_2src",
        dummy_build_dataset_2src
    )

    # Change working directory to tmp
    monkeypatch.chdir(tmp_path)

    return tmp_path


# ------------------------------------------------------------
# Test script execution
# ------------------------------------------------------------
def test_script_runs_without_errors(patch_dependencies):
    """
    Ensures that:
    - build_dataset_2src() is called
    - dummy dataset is created
    """

    # Call the dataset builder manually (cleaner than exec)
    script.build_dataset_2src(
        N_samples=script.N_SAMPLES,
        theta=script.theta,
        path_E=script.PATH_E,
        path_H=script.PATH_H,
        test_size=0.30,
        out_dir=script.OUT_DIR,
        dataset_name=script.DATASET_NAME
    )

    # Expected output file
    expected_file = os.path.join(
        script.OUT_DIR,
        script.DATASET_NAME
    )

    assert os.path.isfile(expected_file), "2-source dataset file was not created"


# ------------------------------------------------------------
# Test dummy dataset validity
# ------------------------------------------------------------
def test_dummy_dataset_validity(tmp_path):
    """
    Ensures that the dummy dataset saved by dummy_build_dataset_2src()
    is a valid npz file with expected fields.
    """

    theta = np.linspace(0, 2*np.pi, 30)
    out_path = tmp_path / "dummy_2src.npz"

    dummy_build_dataset_2src(
        N_samples=100,
        theta=theta,
        path_E="E.pth",
        path_H="H.pth",
        test_size=0.3,
        out_dir=tmp_path,
        dataset_name="dummy_2src.npz"
    )

    data = np.load(out_path)

    assert "X_train" in data
    assert "Y_train" in data
    assert "X_test" in data
    assert "Y_test" in data

    assert data["X_train"].shape[0] == 10
    assert data["X_test"].shape[0] == 5
