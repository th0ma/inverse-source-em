"""
Multitask neural network for Regression Problem I (single-source localization).

This model predicts the three physical parameters of a single source:

    - x  : Cartesian x-coordinate
    - y  : Cartesian y-coordinate
    - I  : source strength (intensity)

Architecture
------------
The network uses a shared encoder followed by two task-specific heads:

    Shared encoder:
        X → MLP → h

    Localization head:
        h → MLP → (x, y)

    Strength head:
        h → MLP → I

Additionally, the model learns three log-variance parameters:

    s_xy : uncertainty weight for (x, y)
    s_I  : uncertainty weight for I
    s_f  : optional feature-space uncertainty (unused but included for completeness)

These parameters enable **heteroscedastic uncertainty weighting** during training,
following the Kendall & Gal (2018) formulation:

    L = exp(-s_xy) * ||xy_pred - xy_true||^2 + s_xy
      + exp(-s_I)  * ||I_pred  - I_true ||^2 + s_I

This allows the network to automatically balance the regression tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultitaskNet(nn.Module):
    """
    Shared-encoder multitask network for Regression I (single-source localization).

    Parameters
    ----------
    input_dim : int, optional
        Dimensionality of the input feature vector (default: 120).
        For 30 angles and 4 channels (Re(E), Im(E), Re(H), Im(H)):
            input_dim = 4 * 30 = 120
    hidden_dim : int, optional
        Width of the shared encoder and task heads.

    Outputs
    -------
    dict with:
        "xy"  : tensor of shape (B, 2)
        "I"   : tensor of shape (B, 1)
        "s_xy": scalar learnable log-variance
        "s_I" : scalar learnable log-variance
        "s_f" : scalar learnable log-variance (optional)
        "h"   : shared latent representation (B, hidden_dim)

    Notes
    -----
    - All computations use float64 for consistency with surrogate data.
    - The model does not apply any activation to the outputs (raw regression).
    """

    def __init__(self, input_dim=120, hidden_dim=256):
        super().__init__()

        # ------------------------------------------------------------
        # Shared encoder
        # ------------------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # ------------------------------------------------------------
        # Localization head (x, y)
        # ------------------------------------------------------------
        self.head_xy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

        # ------------------------------------------------------------
        # Strength head (I)
        # ------------------------------------------------------------
        self.head_I = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # ------------------------------------------------------------
        # Learnable log-variances (heteroscedastic uncertainty)
        # ------------------------------------------------------------
        self.s_xy = nn.Parameter(torch.tensor(0.0))
        self.s_I  = nn.Parameter(torch.tensor(0.0))
        self.s_f  = nn.Parameter(torch.tensor(0.0))

    # ------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------
    def forward(self, X):
        """
        Forward pass through the multitask network.

        Parameters
        ----------
        X : torch.Tensor of shape (B, input_dim)
            Input feature vectors.

        Returns
        -------
        dict
            {
                "xy":  (B, 2) tensor,
                "I":   (B, 1) tensor,
                "s_xy": scalar tensor,
                "s_I":  scalar tensor,
                "s_f":  scalar tensor,
                "h":    (B, hidden_dim) latent representation
            }
        """
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
