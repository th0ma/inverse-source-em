"""
Test for make_1src_dataset.py

Checks:
- imports work
- main() runs without crashing
- build_dataset_1src() is called correctly
- save_dataset_1src() is called correctly
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
import make_1src_dataset as script


# ------------------------------------------------------------
# Dummy implementations for mocking
# ------------------------------------------------------------
def dummy_build_dataset_1src(n_sources, num_angles, path_E, path_H):
    """
    Returns minimal valid dummy dataset for Regression I.
    """
    X = np.zeros((10, num_angles), dtype=np.float32)
    Y = np.zeros((10, 2), dtype=np.float32)
    theta_obs = np.linspace(0, 2*np.pi, num_angles, dtype=np.float32)
    ymax_E = 1.0
    ymax_H = 1.0
    R = 1.234
    return X, Y, theta_obs, ymax_E, ymax_H, R


def dummy_save_dataset_1src(path, X, Y, theta_obs, ymax_E, ymax_H, R):
    """
    Saves a minimal dummy npz file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        X=X,
        Y=Y,
        theta_obs=theta_obs,
        ymax_E=ymax_E,
        ymax_H=ymax_H,
        R=R,
    )


# ------------------------------------------------------------
# Fixture to patch dependencies
# ------------------------------------------------------------
@pytest.fixture
def patch_dependencies(monkeypatch, tmp_path):
    # Patch build_dataset_1src
    monkeypatch.setattr(
        script,
        "build_dataset_1src",
        dummy_build_dataset_1src
    )

    # Patch save_dataset_1src
    monkeypatch.setattr(
        script,
        "save_dataset_1src",
        dummy_save_dataset_1src
    )

    # Change working directory to tmp
    monkeypatch.chdir(tmp_path)

    return tmp_path


# ------------------------------------------------------------
# Test main()
# ------------------------------------------------------------
def test_main_runs_without_errors(patch_dependencies):
    """
    Ensures that:
    - main() executes
    - output file is created
    """

    script.main()

    # Check that output file exists
    out_file = "data/regression/dataset_1src.npz"
    assert os.path.isfile(out_file), "dataset_1src.npz was not created"


# ------------------------------------------------------------
# Test that saved dataset loads correctly
# ------------------------------------------------------------
def test_saved_dataset_is_valid(tmp_path):
    """
    Ensures that the dummy save function produces a valid npz file.
    """

    X = np.zeros((10, 30), dtype=np.float32)
    Y = np.zeros((10, 2), dtype=np.float32)
    theta_obs = np.linspace(0, 2*np.pi, 30, dtype=np.float32)

    path = tmp_path / "dummy_1src.npz"

    dummy_save_dataset_1src(
        path,
        X,
        Y,
        theta_obs,
        ymax_E=1.0,
        ymax_H=1.0,
        R=1.234,
    )

    data = np.load(path)

    assert "X" in data
    assert "Y" in data
    assert "theta_obs" in data
    assert "ymax_E" in data
    assert "ymax_H" in data
    assert "R" in data

    assert data["X"].shape == (10, 30)
    assert data["Y"].shape == (10, 2)
