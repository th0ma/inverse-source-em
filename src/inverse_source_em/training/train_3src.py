"""
Training pipeline for the 3‑source regression model.

This module implements the full curriculum‑based training strategy used
for the three‑source inverse EM regression task. The pipeline includes:

1. Curriculum training over geometry stages 1..8
   Each stage corresponds to a different geometric configuration of the
   three sources. The model is trained sequentially across stages, with
   later stages initialized from the best checkpoint of the previous one.

2. Noise‑consistency regularization
   During training, 15% of the angular measurements are perturbed with
   small Gaussian noise. The model is encouraged to produce consistent
   predictions between clean and noisy inputs, improving robustness.

3. Cosine annealing learning‑rate scheduler
   Uses CosineAnnealingWarmRestarts to gradually reduce the learning rate
   while allowing periodic warm restarts.

4. Early stopping
   Training for each stage stops when validation loss does not improve
   for a specified patience window.

5. Checkpointing
   Best model for each stage is saved under:
       models/regression_3src/best_model_stage_{k}.pt

This module does NOT define the dataset or the model architecture; it
only orchestrates the training process.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from inverse_source_em.data.dataset_3src import load_3src_datasets
from inverse_source_em.training.model_3src import ThreeSourceMultiHeadBig
from inverse_source_em.training.loss_3src import multihead_loss


# ============================================================
# 1. Device
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. Load model for a given stage
# ============================================================

def load_model_for_stage(input_dim: int, stage_index: int, ckpt_dir: str):
    """
    Load or initialize the model for a given curriculum stage.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.
    stage_index : int
        Current stage index (1..8).
    ckpt_dir : str
        Directory containing stage checkpoints.

    Returns
    -------
    model : nn.Module
        Initialized or stage‑pretrained model.

    Notes
    -----
    - Stage 1 always starts from a fresh model.
    - Stage k>1 loads the best checkpoint from stage k‑1 if available.
    """
    model = ThreeSourceMultiHeadBig(input_dim).to(DEVICE)

    if stage_index > 1:
        prev_ckpt = os.path.join(ckpt_dir, f"best_model_stage_{stage_index-1}.pt")
        if os.path.exists(prev_ckpt):
            model.load_state_dict(torch.load(prev_ckpt, map_location=DEVICE))
            print(f"Loaded previous stage weights from {prev_ckpt}")
        else:
            print(f"WARNING: previous checkpoint {prev_ckpt} not found. Starting fresh.")
    else:
        print("Stage 1: initializing fresh model.")

    return model


# ============================================================
# 3. Training function for a single stage
# ============================================================

def train_stage(
    stage_index: int,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    scaler_y,
    ckpt_dir: str,
    epochs: int = 3000,
    lr: float = 5e-4,
    patience: int = 300,
    block: int = 100,
):
    """
    Train the model for a single curriculum stage.

    Parameters
    ----------
    stage_index : int
        Current stage index (1..8).
    model : nn.Module
        Three‑source regression model.
    train_loader : DataLoader
        Training dataset loader.
    test_loader : DataLoader
        Validation dataset loader.
    scaler_y : sklearn-like scaler
        Used to inverse-transform target coordinates.
    ckpt_dir : str
        Directory for saving checkpoints.
    epochs : int, optional
        Maximum number of epochs for this stage.
    lr : float, optional
        Learning rate for Adam optimizer.
    patience : int, optional
        Early stopping patience.
    block : int, optional
        Logging interval.

    Returns
    -------
    model : nn.Module
        Model loaded with the best checkpoint for this stage.

    Notes
    -----
    - Includes noise‑consistency regularization: 15% of angles are
      perturbed with small Gaussian noise.
    - Uses CosineAnnealingWarmRestarts for LR scheduling.
    - Best model is saved as best_model_stage_{stage}.pt.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=200,
        T_mult=2,
        eta_min=1e-6,
    )

    best_val = float("inf")
    patience_counter = 0

    ckpt_path = os.path.join(ckpt_dir, f"best_model_stage_{stage_index}.pt")

    for epoch in range(1, epochs + 1):

        # =====================================================
        # TRAINING LOOP
        # =====================================================
        model.train()
        train_losses = []
        train_rho = []
        train_phi = []

        for Xb, yb in train_loader:
            Xb = Xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            # --------------------------------------------
            # Clean forward pass
            # --------------------------------------------
            rho_clean, cos_clean, sin_clean = model(Xb)

            # Unscale targets
            yb_unscaled = scaler_y.inverse_transform(yb.cpu().numpy())
            yb_unscaled = torch.tensor(yb_unscaled, dtype=torch.float32).to(DEVICE)

            # Clean supervised loss
            loss_clean, metrics = multihead_loss(
                rho_clean, cos_clean, sin_clean, yb_unscaled
            )

            # --------------------------------------------
            # Noise-consistency pass (15% noisy angles)
            # --------------------------------------------
            Xb_noisy = Xb.clone()
            batch_size, input_dim = Xb_noisy.shape

            num_angles = 30
            feats_per_angle = input_dim // num_angles

            Xb_noisy = Xb_noisy.view(batch_size, num_angles, feats_per_angle)

            k = int(0.15 * num_angles)

            for i in range(batch_size):
                idx = torch.randperm(num_angles)[:k]
                noise = 0.01 * torch.randn(
                    len(idx), feats_per_angle,
                    dtype=Xb.dtype, device=Xb.device
                )
                Xb_noisy[i, idx] += noise

            Xb_noisy = Xb_noisy.view(batch_size, input_dim)

            rho_noisy, cos_noisy, sin_noisy = model(Xb_noisy)

            consistency_loss = (
                torch.nn.functional.mse_loss(rho_clean, rho_noisy) +
                torch.nn.functional.mse_loss(cos_clean, cos_noisy) +
                torch.nn.functional.mse_loss(sin_clean, sin_noisy)
            )

            # --------------------------------------------
            # Total loss
            # --------------------------------------------
            loss = loss_clean + 0.1 * consistency_loss

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_rho.append(metrics["rho"])
            train_phi.append(metrics["phi"])

        # =====================================================
        # VALIDATION LOOP
        # =====================================================
        model.eval()
        val_losses = []
        val_rho = []
        val_phi = []

        with torch.no_grad():
            for Xv, yv in test_loader:
                Xv = Xv.to(DEVICE)
                yv = yv.to(DEVICE)

                rho_p, cos_p, sin_p = model(Xv)

                yv_unscaled = scaler_y.inverse_transform(yv.cpu().numpy())
                yv_unscaled = torch.tensor(yv_unscaled, dtype=torch.float32).to(DEVICE)

                vloss, vmetrics = multihead_loss(
                    rho_p, cos_p, sin_p, yv_unscaled
                )

                val_losses.append(vloss.item())
                val_rho.append(vmetrics["rho"])
                val_phi.append(vmetrics["phi"])

        val_loss = np.mean(val_losses)
        scheduler.step()

        # =====================================================
        # LOGGING
        # =====================================================
        if epoch % block == 0:
            print(
                f"[Stage {stage_index} | Epoch {epoch}] "
                f"TRAIN loss={np.mean(train_losses):.4f}, "
                f"ρ={np.mean(train_rho):.4f}, φ={np.mean(train_phi):.4f}, "
                f"VAL loss={val_loss:.4f}, "
                f"ρ={np.mean(val_rho):.4f}, φ={np.mean(val_phi):.4f}, "
                f"LR={optimizer.param_groups[0]['lr']:.2e}"
            )

        # =====================================================
        # EARLY STOPPING
        # =====================================================
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    print(f"Loaded best model for Stage {stage_index} from {ckpt_path}")

    return model


# ============================================================
# 4. Curriculum training (Stage 1 → 8)
# ============================================================

def train_full_curriculum(
    data_dir: str,
    ckpt_dir: str,
    stages=(1, 2, 3, 4, 5, 6, 7, 8),
):
    """
    Train the 3‑source model across all geometry stages.

    Parameters
    ----------
    data_dir : str
        Directory containing the stage‑structured datasets.
    ckpt_dir : str
        Directory where checkpoints will be saved.
    stages : iterable of int, optional
        Sequence of stages to train (default: 1..8).

    Notes
    -----
    - Each stage loads its own dataset via load_3src_datasets().
    - The model for stage k>1 is initialized from the best checkpoint of
      stage k‑1, enabling curriculum learning.
    - After all stages complete, the directory contains:
          best_model_stage_1.pt
          best_model_stage_2.pt
          ...
          best_model_stage_8.pt
    """

    os.makedirs(ckpt_dir, exist_ok=True)

    for stage in stages:

        print("\n====================================================")
        print(f"=== TRAINING STAGE {stage} ===")
        print("====================================================")

        # Load dataset for this stage
        train_ds, test_ds, scaler_X, scaler_y = load_3src_datasets(
            data_dir=data_dir,
            stage=stage,
            device=DEVICE,
            dtype=torch.float32,
        )

        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

        input_dim = train_ds.X.shape[1]

        # Load model (fresh for stage 1, pretrained for others)
        model = load_model_for_stage(input_dim, stage, ckpt_dir)

        # Train this stage
        model = train_stage(
            stage_index=stage,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            scaler_y=scaler_y,
            ckpt_dir=ckpt_dir,
        )

    print("\nAll stages completed.")
