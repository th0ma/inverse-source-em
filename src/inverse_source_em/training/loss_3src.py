"""
Loss function for the 3‑source regression pipeline.

This module provides:
- multihead_loss: ρ + φ + unit‑norm penalty, with distance‑aware weighting
"""

import torch
import numpy as np


def multihead_loss(
    rho_pred: torch.Tensor,
    cos_pred: torch.Tensor,
    sin_pred: torch.Tensor,
    y_unscaled: torch.Tensor,
):
    """
    Multi-head loss for 3-source localization.

    Inputs:
        rho_pred   : (batch, 3)
        cos_pred   : (batch, 3)
        sin_pred   : (batch, 3)
        y_unscaled : (batch, 6)  -- true Cartesian (x1,y1,x2,y2,x3,y3)

    Returns:
        loss    : scalar tensor
        metrics : dict with components (rho, phi, norm)
    """

    # 1. Extract true Cartesian coordinates
    xA, yA = y_unscaled[:, 0], y_unscaled[:, 1]
    xB, yB = y_unscaled[:, 2], y_unscaled[:, 3]
    xC, yC = y_unscaled[:, 4], y_unscaled[:, 5]

    # 2. Convert to true polar coordinates
    rho_true = torch.stack(
        [
            torch.sqrt(xA**2 + yA**2),
            torch.sqrt(xB**2 + yB**2),
            torch.sqrt(xC**2 + yC**2),
        ],
        dim=1,
    )

    phi_true = torch.stack(
        [
            torch.atan2(yA, xA),
            torch.atan2(yB, xB),
            torch.atan2(yC, xC),
        ],
        dim=1,
    )

    cos_true = torch.cos(phi_true)
    sin_true = torch.sin(phi_true)

    # 3. Distance-aware weighting
    w = 1.0 / (rho_true + 1e-6)

    # 4. Loss components
    loss_rho = (w * (rho_pred - rho_true) ** 2).mean()

    loss_phi = (w * ((cos_pred - cos_true) ** 2 + (sin_pred - sin_true) ** 2)).mean()

    norm_penalty = ((cos_pred**2 + sin_pred**2 - 1.0) ** 2).mean()

    # 5. Weighted total loss
    lambda_phi = 2.0
    loss = loss_rho + lambda_phi * loss_phi + 0.1 * norm_penalty

    metrics = {
        "rho": float(loss_rho.item()),
        "phi": float(loss_phi.item()),
        "norm": float(norm_penalty.item()),
    }

    return loss, metrics
