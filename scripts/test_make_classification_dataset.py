"""
Test for make_classification_dataset.py

Checks:
- imports work
- main() runs without crashing
- generate_classification_dataset() is called correctly
- report_dataset() loads npz files correctly
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
import make_classification_dataset as script


# ------------------------------------------------------------
# Dummy classes for mocking
# ------------------------------------------------------------
class DummyPhysics:
    def __init__(self):
        self.R = 1.234  # arbitrary constant


class DummySurrogate:
    def __init__(self):
        pass


class DummyWrapper:
    def __init__(self, sur):
        self.sur = sur


# ------------------------------------------------------------
# Dummy dataset generator
# ------------------------------------------------------------
def dummy_generate_classification_dataset(out_dir, sur_wrap, R, theta, samples_per_class, seed):
    """
    Returns a path to a dummy npz file.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dummy_classification.npz")

    # Create minimal valid dataset
    X_train = np.zeros((10, 4, len(theta)), dtype=np.float32)
    y_train = np.zeros((10,), dtype=np.int64)

    X_val = np.zeros((5, 4, len(theta)), dtype=np.float32)
    y_val = np.zeros((5,), dtype=np.int64)

    X_test = np.zeros((5, 4, len(theta)), dtype=np.float32)
    y_test = np.zeros((5,), dtype=np.int64)

    np.savez(out_path,
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)

    return out_path


# ------------------------------------------------------------
# Fixture to patch dependencies
# ------------------------------------------------------------
@pytest.fixture
def patch_dependencies(monkeypatch, tmp_path):
    # Patch PhysicsTM
    monkeypatch.setattr(script, "PhysicsTM", lambda: DummyPhysics())

    # Patch SurrogateEM
    monkeypatch.setattr(script, "SurrogateEM", lambda: DummySurrogate())

    # Patch SurrogateWrapper
    monkeypatch.setattr(script, "SurrogateWrapper", lambda sur: DummyWrapper(sur))

    # Patch generate_classification_dataset
    monkeypatch.setattr(
        script,
        "generate_classification_dataset",
        dummy_generate_classification_dataset
    )

    # Patch report_dataset to avoid printing
    monkeypatch.setattr(script, "report_dataset", lambda path: None)

    # Change working directory to tmp
    monkeypatch.chdir(tmp_path)

    return tmp_path


# ------------------------------------------------------------
# Test main()
# ------------------------------------------------------------
def test_main_runs_without_errors(patch_dependencies, monkeypatch):
    """
    Ensures that:
    - main() executes
    - generate_classification_dataset() is called
    - output file is created
    """

    # Fake argv so argparse does NOT see pytest args
    monkeypatch.setattr(sys, "argv", ["make_classification_dataset.py"])

    # Run main()
    script.main()

    # Check that output directory exists
    assert os.path.isdir("data/classification")

    # Check that dummy output file exists
    out_file = os.path.join("data/classification", "dummy_classification.npz")
    assert os.path.isfile(out_file)


# ------------------------------------------------------------
# Test report_dataset()
# ------------------------------------------------------------
def test_report_dataset_loads_npz_correctly(tmp_path):
    """
    Creates a dummy classification npz and checks that report_dataset()
    loads X_train/X_val/X_test without errors.
    """

    X_train = np.zeros((10, 4, 30), dtype=np.float32)
    y_train = np.zeros((10,), dtype=np.int64)

    X_val = np.zeros((5, 4, 30), dtype=np.float32)
    y_val = np.zeros((5,), dtype=np.int64)

    X_test = np.zeros((5, 4, 30), dtype=np.float32)
    y_test = np.zeros((5,), dtype=np.int64)

    path = tmp_path / "dummy.npz"
    np.savez(path,
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)

    # Should not raise
    script.report_dataset(str(path))
