"""
Test for train_2src_model.py (refactored version)

Checks:
- main() runs without crashing
- dataset loading is mocked
- model building is mocked
- training loop is mocked
- output model + logs files are created
"""

import os
import sys
import numpy as np
import pytest
import torch
import pandas as pd

# Ensure project root is in path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import script under test
import train_2src_model as script


# ------------------------------------------------------------
# Dummy components for mocking
# ------------------------------------------------------------
class DummyModel:
    def __init__(self, *args, **kwargs):
        pass

    def to(self, *args, **kwargs):
        return self

    def state_dict(self):
        return {"dummy": True}


def dummy_build_model(input_size, hidden_dims, output_size):
    return DummyModel()


def dummy_train_model(model, train_loader, test_loader, config, device):
    logs_df = pd.DataFrame({"epoch": [1, 2], "loss": [0.5, 0.4]})
    return model, logs_df


def dummy_np_load(path):
    return {
        "X_train": np.zeros((10, 120)),
        "Y_train": np.zeros((10, 4)),
        "X_test": np.zeros((5, 120)),
        "Y_test": np.zeros((5, 4)),
    }


def dummy_joblib_load(path):
    return lambda x: x  # identity scaler


def dummy_torch_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"DUMMY")


def dummy_to_csv(self, path, index):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("epoch,loss\n1,0.5\n2,0.4\n")


# ------------------------------------------------------------
# Fixture to patch dependencies
# ------------------------------------------------------------
@pytest.fixture
def patch_dependencies(monkeypatch, tmp_path):

    # Patch dataset loading
    monkeypatch.setattr(script.np, "load", lambda path: dummy_np_load(path))
    monkeypatch.setattr(script.joblib, "load", lambda path: dummy_joblib_load(path))

    # Patch model building + training
    monkeypatch.setattr(script, "build_model", dummy_build_model)
    monkeypatch.setattr(script, "train_model", dummy_train_model)

    # Patch saving
    monkeypatch.setattr(script.torch, "save", dummy_torch_save)
    monkeypatch.setattr(pd.DataFrame, "to_csv", dummy_to_csv)

    # Patch DataLoader to avoid real torch ops
    monkeypatch.setattr(script, "DataLoader", lambda data, batch_size, shuffle: data)

    # Patch torch.tensor to avoid GPU ops
    monkeypatch.setattr(script.torch, "tensor", lambda arr, dtype, device: arr)

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
    - model + logs files are created
    """

    script.main()

    # Expected output files
    model_path = os.path.join("models", "best_model_2src.pth")
    logs_path = os.path.join("models", "training_logs_2src.csv")

    assert os.path.isfile(model_path), "Model file was not created"
    assert os.path.isfile(logs_path), "Logs file was not created"
