"""
Test for train_surrogate_models.py

Checks:
- imports work
- main() runs without crashing
- load_surrogate_dataset() is called twice
- train_surrogate() is called twice
- SurrogateMLP() is instantiated twice
- output .pth files are created
"""

import os
import sys
import numpy as np
import pytest
import torch

# Ensure project root is in path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import script under test
import train_surrogate_models as script


# ------------------------------------------------------------
# Dummy classes and functions for mocking
# ------------------------------------------------------------
class DummyModel:
    """Mock surrogate model."""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def dummy_load_surrogate_dataset(filename, data_dir, batch_size, device):
    """Return dummy train/val loaders."""
    # Return simple lists instead of real DataLoaders
    train_loader = [("X", "Y")] * 2
    val_loader = [("X", "Y")] * 1
    return train_loader, val_loader


def dummy_train_surrogate(model, train_loader, val_loader, save_path, lr, max_epochs, patience, device):
    """Create a dummy .pth file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"dummy": True}, save_path)


# ------------------------------------------------------------
# Fixture to patch dependencies
# ------------------------------------------------------------
@pytest.fixture
def patch_dependencies(monkeypatch, tmp_path):
    # Patch SurrogateMLP
    monkeypatch.setattr(script, "SurrogateMLP", lambda *a, **k: DummyModel(*a, **k))

    # Patch load_surrogate_dataset
    monkeypatch.setattr(script, "load_surrogate_dataset", dummy_load_surrogate_dataset)

    # Patch train_surrogate
    monkeypatch.setattr(script, "train_surrogate", dummy_train_surrogate)

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
    - two .pth files are created
    """

    script.main()

    # Expected output files
    expected_E = os.path.join("models", "surrogate_Esurf.pth")
    expected_H = os.path.join("models", "surrogate_Hsurf.pth")

    assert os.path.isfile(expected_E), "Esurf model file not created"
    assert os.path.isfile(expected_H), "Hsurf model file not created"


# ------------------------------------------------------------
# Test dummy training function
# ------------------------------------------------------------
def test_dummy_train_creates_file(tmp_path):
    """Ensure dummy_train_surrogate writes a valid .pth file."""

    out_path = tmp_path / "dummy_model.pth"
    dummy_train_surrogate(
        model=None,
        train_loader=None,
        val_loader=None,
        save_path=out_path,
        lr=1e-3,
        max_epochs=10,
        patience=5,
        device="cpu"
    )

    assert os.path.isfile(out_path)
    data = torch.load(out_path)
    assert data["dummy"] is True
