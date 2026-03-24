import torch
import torch.nn as nn
import torch.nn.functional as F


class MultitaskNet(nn.Module):
    """
    Shared-encoder multitask network for Regression I (single-source localization).

    Predicts:
        x, y, I
    Learns:
        s_xy, s_I, s_f (log-variances for uncertainty weighting)
    """

    def __init__(self, input_dim=120, hidden_dim=256):
        super().__init__()

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Localization head (x, y)
        self.head_xy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

        # Strength head (I)
        self.head_I = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Learnable log-variances
        self.s_xy = nn.Parameter(torch.tensor(0.0))
        self.s_I  = nn.Parameter(torch.tensor(0.0))
        self.s_f  = nn.Parameter(torch.tensor(0.0))

    def forward(self, X):
        h = self.encoder(X)

        xy = self.head_xy(h)
        I  = self.head_I(h)

        return {
            "xy": xy,
            "I": I,
            "s_xy": self.s_xy,
            "s_I": self.s_I,
            "s_f": self.s_f,
            "h": h,
        }
