"""
Model definition for the 3‑source regression pipeline.

This module provides:
- ThreeSourceMultiHeadBig: multi‑head MLP for 3‑source localization
"""

import torch
import torch.nn as nn


class ThreeSourceMultiHeadBig(nn.Module):
    """
    Multi-head MLP for 3-source localization.

    Architecture (from the original 3‑source experiment):
        Backbone: 512 → 512 → 256 → 128 → 64
        Head ρ:   3 outputs (sigmoid)
        Head φ:   6 outputs (cos/sin)
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
        Forward pass.

        Args:
            x: tensor of shape (batch, input_dim)

        Returns:
            rho_pred: (batch, 3)
            cos_pred: (batch, 3)
            sin_pred: (batch, 3)
        """
        h = self.backbone(x)

        rho_pred = self.head_rho(h)      # (B, 3)
        phi_raw = self.head_phi(h)       # (B, 6)

        cos_pred = phi_raw[:, 0:3]
        sin_pred = phi_raw[:, 3:6]

        return rho_pred, cos_pred, sin_pred
