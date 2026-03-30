"""
PyTorch Dataset for Regression Problem I (single-source localization).

This dataset loads the file ``dataset_1src.npz`` and provides:
- X : normalized boundary fields (flattened)
      shape (N, 4*M), where channels are:
          [E_real, E_imag, H_real, H_imag]
- Y : true source parameters (x, y, I)
      shape (N, 3)

The dataset also stores:
- theta_obs : observation angles
- ymax_E, ymax_H : normalization constants for E and H channels
- R : physical radius of the domain

This dataset is fully compatible with the training pipeline and supports:
- automatic conversion to torch tensors
- optional device placement ("cpu" or "cuda")
- slicing via PyTorch DataLoader
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class Regression1SrcDataset(Dataset):
    """
    PyTorch Dataset for single-source localization.

    This dataset loads precomputed regression data for the 1‑source
    localization problem. Each sample contains normalized boundary
    fields and the corresponding ground‑truth source parameters.

    Parameters
    ----------
    npz_path : str
        Path to ``dataset_1src.npz``.
    device : str or torch.device, optional
        Device on which tensors will be stored ("cpu" or "cuda").
        Default is "cpu".
    dtype : torch.dtype, optional
        Tensor dtype for X and Y. Default is torch.float32.

    Attributes
    ----------
    X : torch.Tensor, shape (N, 4*M)
        Flattened normalized boundary fields.
    Y : torch.Tensor, shape (N, 3)
        True source parameters (x, y, I).
    theta_obs : ndarray
        Observation angles used during dataset generation.
    ymax_E : float
        Normalization constant for electric field channels.
    ymax_H : float
        Normalization constant for magnetic field channels.
    R : float
        Physical radius of the domain.
    device : torch.device
        Device where tensors are stored.
    dtype : torch.dtype
        Tensor dtype.
    """

    def __init__(self, npz_path, device="cpu", dtype=torch.float32):
        super().__init__()

        D = np.load(npz_path)

        self.X = D["X"]          # shape (N, 4*M)
        self.Y = D["Y"]          # shape (N, 3)
        self.theta_obs = D["theta_obs"]
        self.ymax_E = float(D["ymax_E"])
        self.ymax_H = float(D["ymax_H"])
        self.R = float(D["R"])

        self.device = device
        self.dtype = dtype

        # Convert to tensors once (efficient)
        self.X = torch.tensor(self.X, dtype=self.dtype, device=self.device)
        self.Y = torch.tensor(self.Y, dtype=self.dtype, device=self.device)

    def __len__(self):
        """
        Number of samples in the dataset.

        Returns
        -------
        int
            Total number of samples.
        """
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
        tuple (X_i, Y_i)
            X_i : torch.Tensor of shape (4*M,)
            Y_i : torch.Tensor of shape (3,)
        """
        return self.X[idx], self.Y[idx]
