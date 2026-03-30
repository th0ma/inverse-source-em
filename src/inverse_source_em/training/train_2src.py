"""
train_2src.py

Training loop for the two-source inverse EM regression task.

This module implements a production-grade training routine for the
two-source localization model. It does NOT load data or define the model;
instead, it provides a clean training function that accepts:

    - model          : nn.Module
    - train_loader   : DataLoader
    - test_loader    : DataLoader
    - config         : dict of hyperparameters and loss weights
    - device         : torch.device

The training loop integrates:
    - structured_loss (distance, angle, area constraints)
    - dynamic tail-weight scheduling (p99-based)
    - gradient clipping
    - ReduceLROnPlateau scheduler
    - full per-epoch logging (returned as a pandas DataFrame)

This design allows flexible experimentation while keeping the training
logic centralized and clean.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .loss_2src import TailWeightScheduler, structured_loss


def build_tail_schedulers(config: dict):
    """
    Construct all tail-weight schedulers from the configuration dictionary.

    Parameters
    ----------
    config : dict
        Must contain sub-dictionaries:
            - TAIL_SCHEDULER_DISTA
            - TAIL_SCHEDULER_DISTB
            - TAIL_SCHEDULER_ANGLEA
            - TAIL_SCHEDULER_ANGLEB

    Returns
    -------
    tuple
        (sched_distA, sched_distB, sched_angleA, sched_angleB)

    Notes
    -----
    Each scheduler dynamically adjusts the weight of a loss component
    based on its 99th percentile error (p99). This helps stabilize
    training by emphasizing difficult tail cases when needed.
    """
    sched_distA = TailWeightScheduler(config["TAIL_SCHEDULER_DISTA"])
    sched_distB = TailWeightScheduler(config["TAIL_SCHEDULER_DISTB"])
    sched_angleA = TailWeightScheduler(config["TAIL_SCHEDULER_ANGLEA"])
    sched_angleB = TailWeightScheduler(config["TAIL_SCHEDULER_ANGLEB"])
    return sched_distA, sched_distB, sched_angleA, sched_angleB


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: dict,
    device: torch.device,
):
    """
    Full training loop for the two-source regression model.

    Parameters
    ----------
    model : nn.Module
        The regression model (e.g., TwoSourcePredictor).
    train_loader : DataLoader
        Training dataset loader.
    test_loader : DataLoader
        Validation/test dataset loader.
    config : dict
        Must contain:
            NUM_EPOCHS : int
            PATIENCE : int
            LEARNING_RATE : float
            CLIP_NORM : float
            LAMBDA_AREA : float
            LAMBDA_DISTA : float
            LAMBDA_DISTB : float
            LAMBDA_ANGLEA : float
            LAMBDA_ANGLEB : float
            TAIL_SCHEDULER_DISTA : dict
            TAIL_SCHEDULER_DISTB : dict
            TAIL_SCHEDULER_ANGLEA : dict
            TAIL_SCHEDULER_ANGLEB : dict
    device : torch.device
        CPU or CUDA device.

    Returns
    -------
    best_model : nn.Module
        The model loaded with the best (lowest validation loss) weights.
    logs_df : pandas.DataFrame
        Per-epoch training log including:
            - train_loss, val_loss
            - learning rate
            - gradient norm
            - weighted loss components
            - p99 diagnostics
            - dynamic weights

    Notes
    -----
    - The structured loss returns a dictionary of weighted components.
      The training loop sums the components:
          area + distA + distB + angleA + angleB
    - Gradient clipping improves stability for deep MLPs.
    - ReduceLROnPlateau lowers LR when validation loss plateaus.
    - The best model is tracked via validation loss.
    """

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["LEARNING_RATE"],
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config["PATIENCE"],
    )

    sched_distA, sched_distB, sched_angleA, sched_angleB = build_tail_schedulers(config)

    best_loss = float("inf")
    best_state = None
    logs = []

    for epoch in range(config["NUM_EPOCHS"]):

        model.train()
        total_loss = 0.0
        grad_norms = []

        # Accumulators for logging
        acc = {
            "area": 0.0,
            "distA": 0.0,
            "distB": 0.0,
            "angleA": 0.0,
            "angleB": 0.0,
            "w_distA": 0.0,
            "w_distB": 0.0,
            "w_angleA": 0.0,
            "w_angleB": 0.0,
            "p99_distA": 0.0,
            "p99_distB": 0.0,
            "p99_angleA": 0.0,
            "p99_angleB": 0.0,
        }

        # -----------------------------
        # TRAIN LOOP
        # -----------------------------
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_X)

            loss_dict = structured_loss(
                preds,
                batch_y,
                config=config,
                sched_distA=sched_distA,
                sched_distB=sched_distB,
                sched_angleA=sched_angleA,
                sched_angleB=sched_angleB,
            )

            loss = sum(
                loss_dict[k] for k in ["area", "distA", "distB", "angleA", "angleB"]
            )
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["CLIP_NORM"]
            )
            grad_norms.append(grad_norm.item())

            optimizer.step()
            total_loss += loss.item()

            for k in acc:
                acc[k] += loss_dict[k]

        # Normalize accumulators
        for k in acc:
            v = acc[k]
            if hasattr(v, "item"):
                v = v.item()
            acc[k] = v / len(train_loader)

        train_loss = total_loss / len(train_loader)

        # -----------------------------
        # VALIDATION LOOP
        # -----------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                preds = model(batch_X)

                loss_dict = structured_loss(
                    preds,
                    batch_y,
                    config=config,
                    sched_distA=sched_distA,
                    sched_distB=sched_distB,
                    sched_angleA=sched_angleA,
                    sched_angleB=sched_angleB,
                )

                val_loss += sum(
                    loss_dict[k] for k in ["area", "distA", "distB", "angleA", "angleB"]
                ).item()

        val_loss /= len(test_loader)
        scheduler.step(val_loss)

        # -----------------------------
        # LOGGING
        # -----------------------------
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "grad_norm": float(np.mean(grad_norms)),
            "area": acc["area"],
            "distA": acc["distA"],
            "distB": acc["distB"],
            "angleA": acc["angleA"],
            "angleB": acc["angleB"],
            "p99_distA": acc["p99_distA"],
            "p99_distB": acc["p99_distB"],
            "p99_angleA": acc["p99_angleA"],
            "p99_angleB": acc["p99_angleB"],
            "w_distA": acc["w_distA"],
            "w_distB": acc["w_distB"],
            "w_angleA": acc["w_angleA"],
            "w_angleB": acc["w_angleB"],
        }

        # Convert tensors to floats
        epoch_row = {
            k: (v.item() if hasattr(v, "item") else float(v))
            for k, v in epoch_row.items()
        }

        logs.append(epoch_row)

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch}: "
                f"train={train_loss:.4f}, val={val_loss:.4f}, "
                f"lr={epoch_row['lr']:.2e}"
            )

        # Track best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    logs_df = pd.DataFrame(logs)
    return model, logs_df
