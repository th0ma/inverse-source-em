"""
train_2src.py

Training loop for the two-source inverse EM regression task.

Uses:
    - model_2src.TwoSourcePredictor (or any compatible model)
    - loss_2src.structured_loss + TailWeightScheduler

This module does NOT load data or define the model; it only
implements the training procedure given:
    - model
    - train_loader
    - test_loader
    - config
    - device
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .loss_2src import TailWeightScheduler, structured_loss


def build_tail_schedulers(config: dict):
    """
    Build tail-weight schedulers from CONFIG.
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
    Production-grade training loop for the two-source regression model.

    Parameters
    ----------
    model : nn.Module
    train_loader : DataLoader
    test_loader : DataLoader
    config : dict
        Must contain:
            NUM_EPOCHS, PATIENCE, LEARNING_RATE, CLIP_NORM,
            LAMBDA_AREA, LAMBDA_DISTA, LAMBDA_DISTB,
            LAMBDA_ANGLEA, LAMBDA_ANGLEB,
            TAIL_SCHEDULER_* configs
    device : torch.device

    Returns
    -------
    best_model : nn.Module
        Model loaded with best (lowest val_loss) weights.
    logs_df : pd.DataFrame
        Training log per epoch.
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

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    logs_df = pd.DataFrame(logs)
    return model, logs_df
