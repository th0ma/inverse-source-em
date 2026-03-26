"""
Noise Robustness Evaluation for Source-Count Classification Model

This module evaluates:
- accuracy under different noise levels
- per-class accuracy degradation
- robustness slope (accuracy drop per unit noise)

Returned structure:
{
    "module": "noise_robustness",
    "status": "passed" | "failed",
    "metrics": {
        "noise_levels": [...],
        "accuracy": [...],
        "per_class_accuracy": [...],
        "robustness_slope": ...
    }
}
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _add_noise(X, noise_level, global_std, rng):
    """Add Gaussian noise scaled by global dataset std."""
    noise = rng.normal(loc=0.0, scale=noise_level * global_std, size=X.shape)
    return X + noise


def _evaluate_accuracy(model, loader, device):
    """Compute accuracy and collect predictions."""
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


def evaluate(model, X_test, y_test, X_train, noise_levels=None, device="cpu"):
    """
    Evaluate classification robustness under additive Gaussian noise.
    """
    if noise_levels is None:
        noise_levels = [0.00, 0.01, 0.03, 0.05, 0.10]

    num_classes = int(np.max(y_test)) + 1
    global_std = float(np.std(X_train))
    rng = np.random.default_rng(12345)

    acc_list = []
    per_class_list = []

    for nl in noise_levels:
        X_noisy = _add_noise(X_test, nl, global_std, rng)

        ds = TensorDataset(
            torch.from_numpy(X_noisy).to(torch.float64),
            torch.from_numpy(y_test).to(torch.long)
        )
        loader = DataLoader(ds, batch_size=64, shuffle=False)

        acc, preds, targets = _evaluate_accuracy(model, loader, device)
        per_class = _per_class_accuracy(preds, targets, num_classes)

        acc_list.append(float(acc))
        per_class_list.append([float(a) for a in per_class])

    # Robustness slope: linear fit of accuracy vs noise
    slope = float(np.polyfit(noise_levels, acc_list, deg=1)[0])

    metrics = {
        "noise_levels": noise_levels,
        "accuracy": acc_list,
        "per_class_accuracy": per_class_list,
        "robustness_slope": slope,
    }

    # Status check
    status = "passed"
    for v in metrics.values():
        if _is_bad(v):
            status = "failed"

    return {
        "module": "noise_robustness",
        "status": status,
        "metrics": metrics
    }
