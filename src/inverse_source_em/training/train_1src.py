#!/usr/bin/env python
# coding: utf-8

"""
Training script for Regression Problem I (single-source localization).

This script trains the MultitaskNet model to jointly predict:
    - x, y : Cartesian coordinates of the source
    - I    : source strength (intensity)

The training objective combines three components:

1. Localization loss (x, y):
       L_xy = ||(x_pred, y_pred) - (x_true, y_true)||^2

2. Strength loss (I):
       L_I = ||I_pred - I_true||^2

3. Field-consistency loss:
       - Convert (x_pred, y_pred) → (rho_pred, phi_pred)
       - Use surrogate forward model to compute predicted fields
       - Compare to true normalized fields in X_fields

All three losses are combined using heteroscedastic uncertainty weighting
(Kendall & Gal, 2018), with learnable log-variance parameters:
    s_xy, s_I, s_f

The script performs:
    - dataset loading
    - train/validation split
    - surrogate loading
    - multitask loss computation
    - Adam optimization
    - ReduceLROnPlateau scheduling
    - early stopping
    - checkpoint saving
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from inverse_source_em.data.dataset_1src import Regression1SrcDataset
from inverse_source_em.surrogate.surrogate import SurrogateEM
from inverse_source_em.surrogate.surrogate_wrapper import SurrogateWrapper
from inverse_source_em.physics.physics_tm import PhysicsTM
from inverse_source_em.training.model_1src import MultitaskNet


# ------------------------------------------------------------
# Device & dtype
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

print("Using device:", device)


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------
DATA_PATH = "data/regression_1src/dataset_1src.npz"
print("Loading dataset:", DATA_PATH)

dataset = Regression1SrcDataset(DATA_PATH, device=device)
N = len(dataset)

# Train/val split
N_train = int(0.9 * N)
N_val = N - N_train

train_dataset = torch.utils.data.Subset(dataset, range(0, N_train))
val_dataset   = torch.utils.data.Subset(dataset, range(N_train, N))

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=256, shuffle=False)

print(f"Train samples: {N_train}")
print(f"Val samples:   {N_val}")


# ------------------------------------------------------------
# Load surrogate (canonical)
# ------------------------------------------------------------
PATH_E = "models/surrogate_Esurf.pth"
PATH_H = "models/surrogate_Hsurf.pth"

sur = SurrogateWrapper(
    SurrogateEM(path_E=PATH_E, path_H=PATH_H)
)

phys = PhysicsTM()
R = phys.R

print("Surrogate loaded.")
print("R =", R)


# ------------------------------------------------------------
# Multitask loss
# ------------------------------------------------------------
def polar_from_cartesian(x, y):
    """
    Convert Cartesian coordinates (x, y) to polar coordinates (rho, phi).

    Parameters
    ----------
    x : torch.Tensor of shape (B,)
        Predicted x-coordinates.
    y : torch.Tensor of shape (B,)
        Predicted y-coordinates.

    Returns
    -------
    rho : torch.Tensor of shape (B,)
        Radial distance sqrt(x^2 + y^2).
    phi : torch.Tensor of shape (B,)
        Angle atan2(y, x) in radians.

    Notes
    -----
    - This conversion is used to feed predictions into the surrogate
      forward model for field-consistency loss.
    """
    rho = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(y, x)
    return rho, phi


def multitask_loss(model_output, X_fields, Y_true, theta_obs, ymax_E, ymax_H):
    """
    Compute the multitask loss for Regression I.

    The loss consists of three components:

    1. Localization loss (x, y):
           L_xy = ||(x_pred, y_pred) - (x_true, y_true)||^2

    2. Strength loss (I):
           L_I = ||I_pred - I_true||^2

    3. Field-consistency loss:
           - Convert (x_pred, y_pred) → (rho_pred, phi_pred)
           - Use surrogate forward model to compute predicted fields
           - Compare to true normalized fields in X_fields

    All three losses are combined using heteroscedastic uncertainty
    weighting with learnable log-variances (s_xy, s_I, s_f):

        L_total =
            0.5 * exp(-s_xy) * L_xy + 0.5 * s_xy +
            0.5 * exp(-s_I)  * L_I  + 0.5 * s_I  +
            0.5 * exp(-s_f)  * L_fields + 0.5 * s_f

    Parameters
    ----------
    model_output : dict
        Output of MultitaskNet:
            {
                "xy": (B, 2),
                "I":  (B, 1),
                "s_xy": scalar,
                "s_I":  scalar,
                "s_f":  scalar,
                "h":    (B, hidden_dim)
            }
    X_fields : torch.Tensor of shape (B, 4*M)
        True boundary fields:
            [E_r, E_i, H_r, H_i] concatenated.
    Y_true : torch.Tensor of shape (B, 3)
        True regression targets: [x, y, I].
    theta_obs : torch.Tensor of shape (M,)
        Observation angles used by the surrogate.
    ymax_E, ymax_H : float
        Normalization constants for E and H fields.

    Returns
    -------
    L_total : torch.Tensor (scalar)
        Total multitask loss.
    metrics : dict
        Dictionary of scalar floats for logging:
            "L_total", "L_xy", "L_I", "L_fields",
            "s_xy", "s_I", "s_f"

    Notes
    -----
    - The surrogate forward model is accessed via the global `sur` object.
    - All computations use float64 for numerical stability.
    """
    xy_pred = model_output["xy"]
    I_pred  = model_output["I"].view(-1)

    s_xy = model_output["s_xy"]
    s_I  = model_output["s_I"]
    s_f  = model_output["s_f"]

    # True targets
    x_true = Y_true[:, 0]
    y_true = Y_true[:, 1]
    I_true = Y_true[:, 2]

    # Localization loss
    x_pred = xy_pred[:, 0]
    y_pred = xy_pred[:, 1]
    L_xy = ((x_pred - x_true)**2 + (y_pred - y_true)**2).mean()

    # Strength loss
    L_I = ((I_pred - I_true)**2).mean()

    # Field-consistency loss
    M = theta_obs.shape[0]

    E_r_true = X_fields[:, 0:M]
    E_i_true = X_fields[:, M:2*M]
    H_r_true = X_fields[:, 2*M:3*M]
    H_i_true = X_fields[:, 3*M:4*M]

    rho_pred, phi_pred = polar_from_cartesian(x_pred, y_pred)

    E_r_pred, E_i_pred, H_r_pred, H_i_pred = sur.batch_forward(
        rho_pred, phi_pred, I_pred, theta_obs
    )

    E_r_pred /= ymax_E
    E_i_pred /= ymax_E
    H_r_pred /= ymax_H
    H_i_pred /= ymax_H

    L_fields = (
        (E_r_pred - E_r_true)**2 +
        (E_i_pred - E_i_true)**2 +
        (H_r_pred - H_r_true)**2 +
        (H_i_pred - H_i_true)**2
    ).mean()

    # Uncertainty-weighted total loss
    L_total = (
        0.5 * torch.exp(-s_xy) * L_xy + 0.5 * s_xy +
        0.5 * torch.exp(-s_I)  * L_I  + 0.5 * s_I  +
        0.5 * torch.exp(-s_f)  * L_fields + 0.5 * s_f
    )

    return L_total, {
        "L_total":  L_total.item(),
        "L_xy":     L_xy.item(),
        "L_I":      L_I.item(),
        "L_fields": L_fields.item(),
        "s_xy":     s_xy.item(),
        "s_I":      s_I.item(),
        "s_f":      s_f.item(),
    }


# ------------------------------------------------------------
# Model, optimizer, scheduler
# ------------------------------------------------------------
model = MultitaskNet(input_dim=120, hidden_dim=256).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, threshold=1e-4
)


# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------
CHECKPOINT_DIR = "models/regression_1src"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

best_val_loss = float("inf")
patience = 20
epochs_no_improve = 0

theta_obs = dataset.theta_obs
ymax_E = dataset.ymax_E
ymax_H = dataset.ymax_H

print("Starting training...")


def main():
    """
    Main training routine for Regression I.

    Steps
    -----
    1. Load dataset_1src.npz via Regression1SrcDataset.
    2. Split into training and validation subsets.
    3. Load surrogate forward model (Esurf, Hsurf).
    4. Instantiate MultitaskNet.
    5. Train for up to 200 epochs with:
           - Adam optimizer
           - ReduceLROnPlateau scheduler
           - Early stopping (patience = 20)
    6. Save best-performing checkpoints.
    7. Print training progress and validation loss.

    Notes
    -----
    - The surrogate is used inside the loss function for field consistency.
    - All tensors are cast to float64 for consistency with surrogate models.
    - Checkpoints are saved in models/regression_1src/.
    """
    global best_val_loss, epochs_no_improve

    for epoch in range(1, 201):

        # TRAIN
        model.train()
        train_losses = []

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device, dtype=torch.float64)
            Y_batch = Y_batch.to(device, dtype=torch.float64)

            optimizer.zero_grad()

            out = model(X_batch)
            loss, _ = multitask_loss(out, X_batch, Y_batch, theta_obs, ymax_E, ymax_H)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        # VAL
        model.eval()
        val_losses = []

        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device, dtype=torch.float64)
                Y_batch = Y_batch.to(device, dtype=torch.float64)

                out = model(X_batch)
                loss, _ = multitask_loss(out, X_batch, Y_batch, theta_obs, ymax_E, ymax_H)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        # Checkpoint
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            epochs_no_improve = 0

            torch.save(
                model.state_dict(),
                os.path.join(CHECKPOINT_DIR, f"best_epoch_{epoch:04d}.pth")
            )

            print(f"[epoch {epoch:04d}] train={train_loss:.6f} val={val_loss:.6f}")

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping.")
            break

    print("Training completed.")


if __name__ == "__main__":
    main()
