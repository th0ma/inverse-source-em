"""
Loss function for the 3‑source regression pipeline.

This module provides the multi-head loss used in the three-source
localization task. The loss combines:

1. ρ-loss:
       Weighted squared error between predicted and true radii.
       Weighting is distance-aware: w = 1 / (ρ_true + ε)

2. φ-loss:
       Angular error expressed via cosine and sine components:
           (cos_pred - cos_true)^2 + (sin_pred - sin_true)^2

3. Unit-norm penalty:
       Encourages (cos_pred, sin_pred) to lie on the unit circle.

The final loss is:
       L = L_rho + λ_phi * L_phi + 0.1 * L_norm

This formulation avoids angular discontinuities and stabilizes training
for multi-source localization.
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

    Parameters
    ----------
    rho_pred : torch.Tensor of shape (batch, 3)
        Predicted normalized radii in (0,1).
    cos_pred : torch.Tensor of shape (batch, 3)
        Predicted cosines of the three angles.
    sin_pred : torch.Tensor of shape (batch, 3)
        Predicted sines of the three angles.
    y_unscaled : torch.Tensor of shape (batch, 6)
        True Cartesian coordinates:
            [x1, y1, x2, y2, x3, y3]

    Returns
    -------
    loss : torch.Tensor (scalar)
        Total multi-head loss.
    metrics : dict
        Dictionary with individual loss components:
            {
                "rho":  float,
                "phi":  float,
                "norm": float
            }

    Notes
    -----
    - True angles are computed via atan2(y, x).
    - Angular loss is computed in cosine/sine space to avoid wrap-around.
    - A small unit-norm penalty ensures (cos_pred, sin_pred) remain valid.
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
