import numpy as np
import torch
from torch.utils.data import Dataset


class SurrogateDataset(Dataset):
    """
    PyTorch Dataset for surrogate training.
    Loads NPZ files with X, Y arrays.
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.X = torch.tensor(data["X"], dtype=torch.float64)
        self.Y = torch.tensor(data["Y"], dtype=torch.float64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
