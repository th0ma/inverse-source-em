"""
PyTorch Dataset class and utilities for the 3‑source regression pipeline.

This module provides:

1. Loading of stage‑based arrays:
       X_train, X_test, y_train, y_test
   for any geometry stage (stage_1, stage_2, ...).

2. Loading of MinMaxScaler objects for X and y.

3. A clean PyTorch Dataset class (ThreeSourceDataset) with the same API
   as the 1‑source and 2‑source datasets.

4. Convenience loader that returns:
       train_ds, test_ds, scaler_X, scaler_y

5. Optional inverse‑transform utilities for converting scaled predictions
   back to physical (normalized Cartesian) coordinates.

The dataset format follows the unified regression convention:
    - X : (N, F) feature matrix
    - y : (N, 6) normalized Cartesian labels
          [x1/R, y1/R, x2/R, y2/R, x3/R, y3/R]
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

    Parameters
    ----------
    data_dir : str
        Directory containing the stage files.
    stage : int or str
        Stage index (e.g., 1, 2, 3).

    Returns
    -------
    X_train, X_test, y_train, y_test : ndarray
        Arrays loaded from:
            stage_{stage}_X_train.npy
            stage_{stage}_X_test.npy
            stage_{stage}_y_train.npy
            stage_{stage}_y_test.npy
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

    Parameters
    ----------
    data_dir : str
        Directory containing the scaler files.
    stage : int or str
        Stage index.

    Returns
    -------
    scaler_X, scaler_y : MinMaxScaler
        Fitted scalers loaded from:
            stage_{stage}_scaler_X.pkl
            stage_{stage}_scaler_y.pkl
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
    PyTorch Dataset for 3‑source regression.

    Parameters
    ----------
    X : ndarray of shape (N, F)
        Feature matrix.
    y : ndarray of shape (N, 6)
        Normalized Cartesian labels:
            [x1/R, y1/R, x2/R, y2/R, x3/R, y3/R]
    device : torch.device or str, optional
        Device to place tensors on ("cpu" or "cuda").
    dtype : torch.dtype, optional
        Tensor dtype (default: torch.float32).

    Notes
    -----
    - X and y are converted to tensors once for efficiency.
    - Device placement is optional and handled internally.
    """

    def __init__(self, X, y, device=None, dtype=torch.float32):
        self.X = torch.tensor(X, dtype=dtype)
        self.y = torch.tensor(y, dtype=dtype)

        if device is not None:
            self.X = self.X.to(device)
            self.y = self.y.to(device)

    def __len__(self):
        """Return the number of samples."""
        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        Retrieve a single sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        (X_i, y_i) : tuple of torch.Tensor
        """
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
    Load train/test datasets and scalers for a given stage.

    Parameters
    ----------
    data_dir : str
        Directory containing stage files.
    stage : int or str
        Stage index.
    device : torch.device or str, optional
        Device for returned tensors.
    dtype : torch.dtype, optional
        Tensor dtype.

    Returns
    -------
    train_ds : ThreeSourceDataset
    test_ds : ThreeSourceDataset
    scaler_X : MinMaxScaler
    scaler_y : MinMaxScaler
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
    Inverse-transform a batch of scaled targets.

    Parameters
    ----------
    y_scaled : ndarray or torch.Tensor
        Scaled target values.
    scaler_y : MinMaxScaler
        Fitted scaler for target space.

    Returns
    -------
    y_inv : ndarray
        Inverse-transformed targets in normalized Cartesian coordinates.

    Notes
    -----
    - If y_scaled is a torch.Tensor, it is converted to NumPy first.
    - This utility is useful for converting model predictions back to
      interpretable physical coordinates.
    """
    if isinstance(y_scaled, torch.Tensor):
        y_np = y_scaled.detach().cpu().numpy()
    else:
        y_np = y_scaled

    y_inv = scaler_y.inverse_transform(y_np)
    return y_inv
