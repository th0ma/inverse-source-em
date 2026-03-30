"""
Dataset utilities for surrogate model training.

This module defines a simple PyTorch Dataset used for training the surrogate
MLP models that approximate the analytical PhysicsTM solver. The dataset
expects NPZ files containing two arrays:

- ``X`` : input features of shape (N, 5)
- ``Y`` : target outputs of shape (N, 2)

The input features follow the normalized surrogate format:

    X = [rho_norm, cos(phi), sin(phi), cos(theta), sin(theta)]

The target outputs contain the real and imaginary parts of the boundary field:

    Y = [Re(field), Im(field)]

This dataset is used by the surrogate training scripts and integrates
seamlessly with PyTorch DataLoader.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class SurrogateDataset(Dataset):
    """
    PyTorch Dataset for surrogate training.

    This dataset loads a compressed NPZ file containing the input features
    and target outputs for training the surrogate MLP. The NPZ file must
    contain two arrays:

    - ``X`` : ndarray of shape (N, 5)
        Normalized surrogate input features.
    - ``Y`` : ndarray of shape (N, 2)
        Real and imaginary parts of the target field.

    Parameters
    ----------
    npz_path : str
        Path to the NPZ file containing ``X`` and ``Y``.

    Notes
    -----
    - All tensors are loaded as float64 for scientific consistency.
    - The dataset is fully compatible with PyTorch DataLoader.
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.X = torch.tensor(data["X"], dtype=torch.float64)
        self.Y = torch.tensor(data["Y"], dtype=torch.float64)

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        Retrieve a single training sample.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        tuple of torch.Tensor
            (X[idx], Y[idx])
        """
        return self.X[idx], self.Y[idx]
