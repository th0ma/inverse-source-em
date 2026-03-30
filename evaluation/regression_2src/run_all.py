"""
Unified evaluation runner for the 2‑source regression model.

This script loads the trained model and the test split of the two‑source
dataset, and executes the full evaluation suite:

- accuracy (permutation‑invariant)
- error tables (detailed geometric errors)
- noise robustness (Gaussian perturbations of the inputs)
- timing (inference latency per batch and per sample)

Permutation invariance is handled internally by each evaluation module using
the two possible assignments:
1. $\left(\text{predA}\rightarrow\text{trueA},\\,\text{predB}\rightarrow\text{trueB}\right)$
2. $\left(\text{predA}\rightarrow\text{trueB},\\,\text{predB}\rightarrow\text{trueA}\right)$

Returned structure:
{
    "module": "run_all",
    "status": "passed" | "failed",
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

from .accuracy import evaluate as eval_accuracy
from .error_tables import evaluate as eval_error_tables
from .noise_robustness import evaluate as eval_noise
from .timing import evaluate as eval_timing


# ------------------------------------------------------------
# Load dataset + scalers
# ------------------------------------------------------------
def load_dataset(data_dir):
    dataset_file = os.path.join(data_dir, "dataset_2src_fullfield_30obs_20k.npz")

    data = np.load(dataset_file)
    X_test = data["X_test"]
    Y_test = data["Y_test"]

    return X_test, Y_test


# ------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------
def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    # Reconstruct architecture
    class TwoSourcePredictor(torch.nn.Module):
        def __init__(self, input_size, hidden_dims, output_size):
            super().__init__()
            layers = []
            prev = input_size

            for i, h in enumerate(hidden_dims):
                layers.append(torch.nn.Linear(prev, h))
                layers.append(torch.nn.ReLU())
                if i < 2:
                    layers.append(torch.nn.Dropout(0.05))
                prev = h

            layers.append(torch.nn.Linear(prev, output_size))
            self.model = torch.nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    model = TwoSourcePredictor(
        input_size=checkpoint["input_size"],
        hidden_dims=checkpoint["hidden_dims"],
        output_size=checkpoint["output_size"]
    ).to(torch.float32).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


# ------------------------------------------------------------
# Main evaluation pipeline
# ------------------------------------------------------------
def run_all(project_root="."):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join(project_root, "data", "regression_2src")
    model_path = os.path.join(project_root, "models", "best_model_2src.pth")

    # Load dataset
    X, Y = load_dataset(data_dir)

    # Load model
    model = load_model(model_path, device)

    # Run modules
    results = {}
    results["accuracy"] = eval_accuracy(model, X, Y, device=device)
    results["error_tables"] = eval_error_tables(model, X, Y, device=device)
    results["noise_robustness"] = eval_noise(
        model, X, Y,
        noise_levels=[0.0, 0.01, 0.03, 0.05, 0.10],
        device=device
    )
    results["timing"] = eval_timing(model, X, Y, device=device)

    # Determine overall status
    status = "passed"
    for module in results.values():
        if module["status"] != "passed":
            status = "failed"

    # Print clean summary
    print("\n=== Regression 2‑Source Evaluation Summary ===\n")
    print("Overall status:", status)
    for name, module in results.items():
        print(f"[{name}] status:", module["status"])

    return {
        "module": "run_all",
        "status": status,
        "results": results
    }


# ------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    run_all(project_root=project_root)
