"""
Run all evaluation modules for the 1‑source regression model.

This script loads:
- the dataset: data/regression_1src/dataset_1src.npz
- the latest model checkpoint: models/regression_1src/best_epoch_xxxx.pth

It then executes the full evaluation suite:
- accuracy (absolute and relative errors)
- error tables (full error arrays + summaries)
- noise robustness (performance under Gaussian noise)
- timing (inference speed and batch‑scaling)

Returned structure:
{
    "overall_status": "passed" | "failed",
    "results": {
        "accuracy": {...},
        "error_tables": {...},
        "noise_robustness": {...},
        "timing": {...}
    }
}
"""


import os
import numpy as np
import torch

from inverse_source_em.training.model_1src import MultitaskNet
from inverse_source_em.data.dataset_1src import Regression1SrcDataset

from . import (
    evaluate_accuracy,
    evaluate_error_tables,
    evaluate_noise,
    evaluate_timing,
)


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
def _load_data(project_root, data_relpath):
    data_path = os.path.join(project_root, data_relpath)
    dataset = Regression1SrcDataset(data_path, device="cpu")

    X = dataset.X.cpu().numpy()
    Y = dataset.Y.cpu().numpy()

    return X, Y


# ------------------------------------------------------------
# Load latest checkpoint
# ------------------------------------------------------------
def _load_latest_checkpoint(ckpt_dir, device):
    ckpt_files = [
        f for f in os.listdir(ckpt_dir)
        if f.startswith("best_epoch_") and f.endswith(".pth")
    ]

    if len(ckpt_files) == 0:
        raise RuntimeError(f"No checkpoints found in {ckpt_dir}")

    def extract_epoch(fname):
        return int(fname.split("_")[-1].split(".")[0])

    epochs = [extract_epoch(f) for f in ckpt_files]
    latest_epoch = max(epochs)

    ckpt_path = os.path.join(ckpt_dir, f"best_epoch_{latest_epoch:04d}.pth")
    return ckpt_path


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------
def _load_model(ckpt_path, device):
    model = MultitaskNet(input_dim=120, hidden_dim=256).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# ------------------------------------------------------------
# Run all evaluations
# ------------------------------------------------------------
def run_all(
    project_root=".",
    data_relpath="data/regression_1src/dataset_1src.npz",
    ckpt_relpath="models/regression_1src",
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    X, Y = _load_data(project_root, data_relpath)

    # Load model
    ckpt_dir = os.path.join(project_root, ckpt_relpath)
    ckpt_path = _load_latest_checkpoint(ckpt_dir, device)
    model = _load_model(ckpt_path, device)

    results = {}

    # Accuracy (zero noise)
    results["accuracy"] = evaluate_accuracy(model, X, Y, device=device)

    # Full error tables
    results["error_tables"] = evaluate_error_tables(model, X, Y, device=device)

    # Noise robustness
    results["noise_robustness"] = evaluate_noise(model, X, Y, device=device)

    # Timing
    results["timing"] = evaluate_timing(model, X, Y, device=device)

    # Overall status
    overall_status = "passed"
    for res in results.values():
        if res["status"] != "passed":
            overall_status = "failed"
            break

    return {
        "overall_status": overall_status,
        "results": results,
    }


# ------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out = run_all(project_root=project_root)

    print("=== Regression 1‑Source Evaluation Summary ===\n")
    print(f"Overall status: {out['overall_status']}\n")

    for name, res in out["results"].items():
        print(f"[{name}] status: {res['status']}")
