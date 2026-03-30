"""
Training script for the source-count classification model (1–5 sources).

This module provides:

1. A PyTorch Dataset wrapper for classification data:
       X : (N, 4, num_angles)
       y : (N,) integer class labels

2. Training utilities:
       - train_one_epoch()
       - evaluate_accuracy()

3. A full training routine:
       - Loads dataset_classification.npz
       - Builds DataLoaders
       - Instantiates SourceCountResNet1D
       - Trains with Adam + cross-entropy
       - Saves best model checkpoint
       - Evaluates on test set
       - Prints confusion matrix and classification report

This script is intended to be run as a standalone training entrypoint.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from inverse_source_em.training.classification_model import SourceCountResNet1D

torch.set_default_dtype(torch.float64)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1. Dataset wrapper
# ============================================================

class ClassificationDataset(Dataset):
    """
    PyTorch dataset for source-count classification.

    Parameters
    ----------
    X : ndarray of shape (N, 4, num_angles)
        Input boundary fields.
    y : ndarray of shape (N,)
        Integer class labels in {0,1,2,3,4}.

    Notes
    -----
    - X is converted to float64 tensors.
    - y is converted to torch.long for cross-entropy.
    """

    def __init__(self, X, y):
        self.X = torch.from_numpy(X).to(torch.float64)
        self.y = torch.from_numpy(y).to(torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# 2. Training / evaluation utilities
# ============================================================

def train_one_epoch(model, loader, optimizer, device):
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        Classification model.
    loader : DataLoader
        Training data loader.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    device : str or torch.device
        Device to train on.

    Returns
    -------
    float
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


def evaluate_accuracy(model, loader, device):
    """
    Compute classification accuracy on a dataset.

    Parameters
    ----------
    model : nn.Module
        Trained model.
    loader : DataLoader
        Evaluation data loader.
    device : str or torch.device
        Device to evaluate on.

    Returns
    -------
    float
        Accuracy in [0,1].
    """
    model.eval()
    correct = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()

    return correct / len(loader.dataset)


# ============================================================
# 3. Main training routine
# ============================================================

def main():
    """
    Train the source-count classifier and evaluate on the test set.

    Steps
    -----
    1. Load dataset_classification.npz
    2. Build train/val/test DataLoaders
    3. Instantiate SourceCountResNet1D
    4. Train for EPOCHS epochs
    5. Save best model (highest val accuracy)
    6. Evaluate on test set
    7. Print confusion matrix + classification report
    """

    # Paths
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    data_path = os.path.join(root_dir, "data", "classification", "dataset_classification.npz")
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "classifier_1_to_5_resnet1d.pt")

    print("Root dir:", root_dir)
    print("Dataset path:", data_path)
    print("Models dir:", models_dir)
    print("Device:", DEVICE)

    # Load dataset
    data = np.load(data_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val   = data["X_val"]
    y_val   = data["y_val"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]

    print("\nDataset loaded:")
    print("X_train:", X_train.shape)
    print("X_val:  ", X_val.shape)
    print("X_test: ", X_test.shape)

    num_angles = X_train.shape[2]

    # Datasets / loaders
    BATCH_SIZE = 64

    train_ds = ClassificationDataset(X_train, y_train)
    val_ds   = ClassificationDataset(X_val,   y_val)
    test_ds  = ClassificationDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=(DEVICE == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=(DEVICE == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=(DEVICE == "cuda"))

    print("\nDataLoaders ready.")
    print("Train batches:", len(train_loader))
    print("Val batches:  ", len(val_loader))
    print("Test batches: ", len(test_loader))

    # Model
    model = SourceCountResNet1D(
        in_channels=4,
        num_angles=num_angles,
        num_classes=5,
        base_channels=64,
        num_blocks=4,
        dropout=0.1,
    ).to(DEVICE)

    print("\nModel:")
    print(model)

    # Optimizer / training config
    EPOCHS = 100
    LR = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    print("\n=== Starting training ===")
    print("Epochs:", EPOCHS)
    print("Learning rate:", LR)
    print("Saving best model to:", best_model_path, "\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_acc = evaluate_accuracy(model, val_loader, DEVICE)

        print(f"Epoch {epoch:03d} | "
              f"train loss = {train_loss:.4f} | "
              f"val acc = {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  → New best model saved (val acc = {best_val_acc:.4f})")

    print("\nTraining complete.")
    print("Best validation accuracy:", best_val_acc)

    # Test evaluation
    print("\n=== Test evaluation ===")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    test_acc = evaluate_accuracy(model, test_loader, DEVICE)
    print(f"Test accuracy: {test_acc:.6f}")

    # Confusion matrix & report
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(DEVICE)
            logits = model(X)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    cm = confusion_matrix(all_targets, all_preds)
    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(classification_report(all_targets, all_preds, digits=4))


if __name__ == "__main__":
    main()
