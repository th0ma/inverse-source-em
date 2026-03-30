"""
Model definition for the 3‑source regression pipeline.

This module provides the neural architecture used in the three‑source
inverse EM regression task. The model follows the successful design of
the original 3‑source experiment:

- A deep shared MLP backbone
- A 3‑output head for normalized radii ρ ∈ (0,1)
- A 6‑output head for angular representation using cosφ and sinφ

The angular representation avoids discontinuities at ±π and stabilizes
training for multi‑source localization.
"""

import torch
import torch.nn as nn


class ThreeSourceMultiHeadBig(nn.Module):
    """
    Multi‑head MLP for 3‑source localization.

    Architecture
    ------------
    Shared backbone:
        input_dim → 512 → 512 → 256 → 128 → 64

    Output heads:
        - rho head: 3 outputs, passed through sigmoid → (0,1)
        - phi head: 6 outputs, interpreted as:
              cos_pred = phi_raw[:, 0:3]
              sin_pred = phi_raw[:, 3:6]

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.
        For 30 angles and 4 channels (Re(E), Im(E), Re(H), Im(H)):
            input_dim = 4 * 30 = 120

    Notes
    -----
    - The model predicts (ρ₁, ρ₂, ρ₃) directly in normalized form.
    - Angles are represented as (cosφᵢ, sinφᵢ) to avoid wrap‑around issues.
    - No activation is applied to the φ head; normalization happens in loss.
    """

    def __init__(self, input_dim: int):
        super().__init__()

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Head for rho (3 outputs, in (0,1))
        self.head_rho = nn.Sequential(
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )

        # Head for cosφ/sinφ (6 outputs)
        self.head_phi = nn.Linear(64, 6)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the multi‑head network.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, input_dim)
            Input feature vectors.

        Returns
        -------
        rho_pred : torch.Tensor of shape (batch, 3)
            Normalized radii in (0,1).
        cos_pred : torch.Tensor of shape (batch, 3)
            Predicted cosines of the three angles.
        sin_pred : torch.Tensor of shape (batch, 3)
            Predicted sines of the three angles.

        Notes
        -----
        - The φ head outputs 6 raw values which are split into cos and sin.
        - No normalization is applied here; the loss function handles it.
        """
        h = self.backbone(x)

        rho_pred = self.head_rho(h)      # (B, 3)
        phi_raw = self.head_phi(h)       # (B, 6)

        cos_pred = phi_raw[:, 0:3]
        sin_pred = phi_raw[:, 3:6]

        return rho_pred, cos_pred, sin_pred
