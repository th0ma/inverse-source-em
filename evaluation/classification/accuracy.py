"""
Accuracy Evaluation for Source-Count Classification Model

This module evaluates:
- overall accuracy
- per-class accuracy
- macro accuracy
- weighted accuracy

Returned structure:
{
    "module": "accuracy",
    "status": "passed" | "failed",
    "metrics": {
        "accuracy": ...,
        "per_class": [...],
        "macro_accuracy": ...,
        "weighted_accuracy": ...
    }
}
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _compute_accuracy(model, loader, device):
    """Compute overall accuracy and collect predictions."""
    model.eval()
    correct = 0
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            preds_all.append(preds.cpu().numpy())
            targets_all.append(y.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    accuracy = correct / len(loader.dataset)
    return accuracy, preds_all, targets_all


def _per_class_accuracy(preds, targets, num_classes):
    """Compute per-class accuracy."""
    acc = []
    for c in range(num_classes):
        mask = (targets == c)
        if mask.sum() == 0:
            acc.append(0.0)
        else:
            acc.append((preds[mask] == c).mean())
    return acc


def _is_bad(x):
    """Check for NaN/Inf in scalars, lists, or arrays."""
    x = np.asarray(x)
    return np.isnan(x).any() or np.isinf(x).any()


def evaluate(model, X_test, y_test, device="cpu"):
    """
    Evaluate classification accuracy metrics.
    """
    num_classes = int(np.max(y_test)) + 1

    # Build dataloader
    test_ds = TensorDataset(
        torch.from_numpy(X_test).to(torch.float64),
        torch.from_numpy(y_test).to(torch.long)
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Compute accuracy
    accuracy, preds, targets = _compute_accuracy(model, test_loader, device)

    # Per-class accuracy
    per_class = _per_class_accuracy(preds, targets, num_classes)

    # Macro accuracy
    macro_accuracy = float(np.mean(per_class))

    # Weighted accuracy
    class_counts = np.bincount(targets, minlength=num_classes)
    weighted_accuracy = float(np.sum(np.array(per_class) * class_counts) / len(targets))

    metrics = {
        "accuracy": float(accuracy),
        "per_class": [float(a) for a in per_class],
        "macro_accuracy": macro_accuracy,
        "weighted_accuracy": weighted_accuracy,
    }

    # Status check
    status = "passed"
    if any(_is_bad(v) for v in metrics.values()):
        status = "failed"

    return {
        "module": "accuracy",
        "status": status,
        "metrics": metrics
    }
