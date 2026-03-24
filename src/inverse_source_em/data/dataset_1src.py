"""
PyTorch Dataset for Regression Problem I (single-source localization).

Loads dataset_1src.npz and returns:
- X : normalized fields (E_real, E_imag, H_real, H_imag)
- Y : true parameters (x, y, I)

The dataset is compatible with the training pipeline and supports:
- automatic conversion to torch tensors
- optional slicing (train/val/test)
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class Regression1SrcDataset(Dataset):
    """
    PyTorch Dataset for single-source localization.

    Parameters
    ----------
    npz_path : str
        Path to dataset_1src.npz.
    device : str or torch.device, optional
        Device for returned tensors ("cpu" or "cuda").
    dtype : torch.dtype, optional
        Tensor dtype (default: torch.float32).
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
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
