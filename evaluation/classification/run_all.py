"""
Run all classification evaluation modules.

Loads:
- dataset: data/classification/dataset_classification.npz
- model:   models/classifier_1_to_5_resnet1d.pt

Runs:
- accuracy
- confusion
- noise_robustness
- timing
"""

import os
import numpy as np
import torch

from inverse_source_em.training.classification_model import SourceCountResNet1D
from . import (
    evaluate_accuracy,
    evaluate_confusion,
    evaluate_noise,
    evaluate_timing,
)


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------

def _load_data(project_root, data_relpath):
    data_path = os.path.join(project_root, data_relpath)
    data = np.load(data_path)

    return {
        "X_train": data["X_train"],
        "y_train": data["y_train"],
        "X_val":   data["X_val"],
        "y_val":   data["y_val"],
        "X_test":  data["X_test"],
        "y_test":  data["y_test"],
    }


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------

def load_model_with_num_angles(project_root, models_relpath, num_angles, device):
    model_path = os.path.join(project_root, models_relpath, "classifier_1_to_5_resnet1d.pt")

    model = SourceCountResNet1D(
        in_channels=4,
        num_angles=num_angles,
        num_classes=5,
        base_channels=64,
        num_blocks=4,
        dropout=0.1,
    ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# ------------------------------------------------------------
# Run all evaluations
# ------------------------------------------------------------

def run_all(
    project_root=".",
    data_relpath="data/classification/dataset_classification.npz",
    models_relpath="models",
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    data = _load_data(project_root, data_relpath)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]

    num_angles = X_train.shape[2]

    # Load model
    model = load_model_with_num_angles(project_root, models_relpath, num_angles, device)

    results = {}

    # Accuracy
    results["accuracy"] = evaluate_accuracy(model, X_test, y_test, device=device)

    # Confusion
    results["confusion"] = evaluate_confusion(model, X_test, y_test, device=device)

    # Noise robustness
    results["noise_robustness"] = evaluate_noise(
        model,
        X_test,
        y_test,
        X_train,
        device=device,
    )

    # Timing
    results["timing"] = evaluate_timing(model, X_test, y_test, device=device)

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

    print("=== Classification Evaluation Summary ===\n")
    print(f"Overall status: {out['overall_status']}\n")

    for name, res in out["results"].items():
        print(f"[{name}] status: {res['status']}")
