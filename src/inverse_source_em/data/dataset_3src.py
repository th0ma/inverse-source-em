"""
PyTorch Dataset class for the 3‑source regression pipeline.

This module provides:
- Loading of X/y arrays for a given geometry stage
- Loading of MinMax scalers
- Optional inverse-transform utilities
- A clean PyTorch Dataset class with the same API as 1src and 2src
"""

import os
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset


# ============================================================
# 1. Utility functions for loading arrays and scalers
# ============================================================

def load_stage_arrays(data_dir, stage):
    """
    Load X_train, X_test, y_train, y_test for a given stage.
    """

    prefix = os.path.join(data_dir, f"stage_{stage}")

    X_train = np.load(prefix + "_X_train.npy")
    X_test  = np.load(prefix + "_X_test.npy")
    y_train = np.load(prefix + "_y_train.npy")
    y_test  = np.load(prefix + "_y_test.npy")

    return X_train, X_test, y_train, y_test


def load_stage_scalers(data_dir, stage):
    """
    Load MinMaxScaler objects for X and y.
    """

    prefix = os.path.join(data_dir, f"stage_{stage}")

    with open(prefix + "_scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)

    with open(prefix + "_scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)

    return scaler_X, scaler_y


# ============================================================
# 2. PyTorch Dataset class
# ============================================================

class ThreeSourceDataset(Dataset):
    """
    PyTorch Dataset for 3-source regression.

    Parameters:
        X : numpy array of shape (N, F)
        y : numpy array of shape (N, 6)
        device : torch device (optional)
        dtype : torch dtype (optional)
    """

    def __init__(self, X, y, device=None, dtype=torch.float32):
        self.X = torch.tensor(X, dtype=dtype)
        self.y = torch.tensor(y, dtype=dtype)

        if device is not None:
            self.X = self.X.to(device)
            self.y = self.y.to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# 3. Convenience loader for training/testing datasets
# ============================================================

def load_3src_datasets(
    data_dir,
    stage,
    device=None,
    dtype=torch.float32
):
    """
    Load train/test datasets + scalers for a given stage.

    Returns:
        train_ds, test_ds, scaler_X, scaler_y
    """

    X_train, X_test, y_train, y_test = load_stage_arrays(data_dir, stage)
    scaler_X, scaler_y = load_stage_scalers(data_dir, stage)

    train_ds = ThreeSourceDataset(X_train, y_train, device=device, dtype=dtype)
    test_ds  = ThreeSourceDataset(X_test,  y_test,  device=device, dtype=dtype)

    return train_ds, test_ds, scaler_X, scaler_y


# ============================================================
# 4. Optional inverse-transform utilities
# ============================================================

def inverse_transform_targets(y_scaled, scaler_y):
    """
    Inverse-transform a batch of scaled targets (numpy or torch).
    """

    if isinstance(y_scaled, torch.Tensor):
        y_np = y_scaled.detach().cpu().numpy()
    else:
        y_np = y_scaled

    y_inv = scaler_y.inverse_transform(y_np)
    return y_inv
