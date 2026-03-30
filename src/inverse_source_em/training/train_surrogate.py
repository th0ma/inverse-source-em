"""
Training utilities for surrogate models.

This module provides a complete training pipeline for surrogate forward
models (Esurf, Hsurf). It includes:

1. Dataset loading from NPZ files
   - X : input features
   - Y : target fields
   - automatic train/validation split
   - PyTorch DataLoaders

2. Training loop with:
   - Adam optimizer
   - MSE loss
   - ReduceLROnPlateau scheduler
   - Early stopping
   - Best-model checkpoint saving

Typical usage
-------------
>>> from inverse_source_em.surrogate import SurrogateMLP
>>> from inverse_source_em.training import load_surrogate_dataset, train_surrogate
>>>
>>> train_loader, val_loader = load_surrogate_dataset("Esurf.npz")
>>> model = SurrogateMLP()
>>> train_surrogate(model, train_loader, val_loader, "surrogate_Esurf.pth")

This module is intentionally simple and stable, since surrogate training
is a foundational step for all downstream inverse problems.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split


# ============================================================
# 1. Dataset loader
# ============================================================

def load_surrogate_dataset(
    filename: str,
    data_dir: str = "data/surrogate",
    val_ratio: float = 0.1,
    batch_size: int = 512,
    device: torch.device | None = None
):
    """
    Load a surrogate dataset from NPZ and return train/val DataLoaders.

    Parameters
    ----------
    filename : str
        Name of the NPZ file (e.g., "Esurf.npz").
    data_dir : str, optional
        Directory containing the dataset.
    val_ratio : float, optional
        Fraction of samples used for validation.
    batch_size : int, optional
        Batch size for DataLoaders.
    device : torch.device or None, optional
        If provided, tensors are moved to this device.

    Returns
    -------
    train_loader : DataLoader
        Training DataLoader.
    val_loader : DataLoader
        Validation DataLoader.

    Notes
    -----
    - X and Y are loaded as float64 tensors for numerical stability.
    - The dataset is randomly split into train/validation subsets.
    - Shuffling is enabled only for the training loader.
    """

    path = os.path.join(data_dir, filename)
    data = np.load(path)

    X = torch.from_numpy(data["X"]).double()
    Y = torch.from_numpy(data["Y"]).double()

    if device is not None:
        X = X.to(device)
        Y = Y.to(device)

    dataset = TensorDataset(X, Y)

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)

    print(f"Loaded {filename}")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Train: {train_size}, Val: {val_size}")
    print(f"  Batch size: {batch_size}")

    return train_loader, val_loader


# ============================================================
# 2. Training loop
# ============================================================

def train_surrogate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_path: str,
    lr: float = 1e-3,
    max_epochs: int = 500,
    patience: int = 50,
    device: torch.device | None = None
):
    """
    Train a surrogate model with early stopping and LR scheduler.

    Parameters
    ----------
    model : nn.Module
        The surrogate model to train.
    train_loader : DataLoader
        Training data.
    val_loader : DataLoader
        Validation data.
    save_path : str
        Path to save the best model checkpoint.
    lr : float, optional
        Learning rate for Adam optimizer.
    max_epochs : int, optional
        Maximum number of epochs.
    patience : int, optional
        Early stopping patience.
    device : torch.device or None, optional
        Device to train on.

    Returns
    -------
    best_val_loss : float
        Best validation loss achieved during training.

    Notes
    -----
    - The model is trained in float64 for consistency with surrogate data.
    - ReduceLROnPlateau halves the LR when validation loss plateaus.
    - Early stopping prevents overfitting and unnecessary computation.
    """

    if device is not None:
        model = model.to(device=device, dtype=torch.float64)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    print(f"Training surrogate → saving best model to: {save_path}")

    for epoch in range(1, max_epochs + 1):
        # -------------------------
        # Training
        # -------------------------
        model.train()
        train_loss = 0.0
        for Xb, Yb in train_loader:
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, Yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(Xb)
        train_loss /= len(train_loader.dataset)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, Yb in val_loader:
                pred = model(Xb)
                loss = criterion(pred, Yb)
                val_loss += loss.item() * len(Xb)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d} | Train: {train_loss:.6e} | Val: {val_loss:.6e}")

        # -------------------------
        # Early stopping
        # -------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"Training complete. Best val loss: {best_val_loss:.6e}")
    return best_val_loss
