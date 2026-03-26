"""
Confusion Matrix Evaluation for Source-Count Classification Model

This module computes:
- raw confusion matrix
- normalized confusion matrix (per true class)
- precision, recall, F1 per class

Returned structure:
{
    "module": "confusion",
    "status": "passed" | "failed",
    "metrics": {
        "confusion_raw": [...],
        "confusion_norm": [...],
        "precision": [...],
        "recall": [...],
        "f1": [...]
    }
}
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix


def _collect_predictions(model, loader, device):
    """Run model on dataset and collect predictions + targets."""
    model.eval()
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            preds = logits.argmax(dim=1)

            preds_all.append(preds.cpu().numpy())
            targets_all.append(y.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    return preds_all, targets_all


def _compute_precision_recall_f1(cm):
    """Compute precision, recall, F1 per class from confusion matrix."""
    num_classes = cm.shape[0]

    precision = []
    recall = []
    f1 = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1_c = 2 * prec * rec / (prec + rec + 1e-12)

        precision.append(float(prec))
        recall.append(float(rec))
        f1.append(float(f1_c))

    return precision, recall, f1


def _is_bad(x):
    """Check for NaN/Inf in scalars, lists, or arrays."""
    x = np.asarray(x)
    return np.isnan(x).any() or np.isinf(x).any()


def evaluate(model, X_test, y_test, device="cpu"):
    """
    Evaluate confusion matrix and class-wise metrics.
    """
    num_classes = int(np.max(y_test)) + 1

    # Build dataloader
    test_ds = TensorDataset(
        torch.from_numpy(X_test).to(torch.float64),
        torch.from_numpy(y_test).to(torch.long)
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Collect predictions
    preds, targets = _collect_predictions(model, test_loader, device)

    # Raw confusion matrix
    cm_raw = confusion_matrix(targets, preds, labels=np.arange(num_classes))

    # Normalized confusion matrix (per true class)
    cm_norm = cm_raw.astype(np.float64)
    cm_norm = cm_norm / (cm_norm.sum(axis=1, keepdims=True) + 1e-12)

    # Precision / Recall / F1
    precision, recall, f1 = _compute_precision_recall_f1(cm_raw)

    metrics = {
        "confusion_raw": cm_raw.tolist(),
        "confusion_norm": cm_norm.tolist(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    # Status check
    status = "passed"
    for v in metrics.values():
        if _is_bad(v):
            status = "failed"

    return {
        "module": "confusion",
        "status": status,
        "metrics": metrics
    }
