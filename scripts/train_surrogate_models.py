"""
Train surrogate models (Esurf and Hsurf) using the inverse_source_em package.

This script:
    - Loads Esurf.npz and Hsurf.npz from data/surrogate/
    - Builds two SurrogateMLP models (one for Esurf, one for Hsurf)
    - Trains each model with fixed hyperparameters for reproducibility
    - Saves the trained weights into:
         models/surrogate_Esurf.pth
         models/surrogate_Hsurf.pth
    - Automatically selects CPU or GPU depending on availability

Usage:
    python train_surrogate_models.py

Notes:
    - Must be executed from the project root directory.
    - Requires the surrogate datasets to already exist in data/surrogate/.
    - Training uses early stopping (patience=20) and max_epochs=200.
"""


import os
import torch

torch.set_default_dtype(torch.float64)

from inverse_source_em.surrogate import SurrogateMLP
from inverse_source_em.training import load_surrogate_dataset, train_surrogate


def main():

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # ------------------------------------------------------------
    # Train Esurf surrogate
    # ------------------------------------------------------------
    print("\n=== Training Esurf surrogate ===")

    train_loader_E, val_loader_E = load_surrogate_dataset(
        filename="Esurf.npz",
        data_dir="data/surrogate",
        batch_size=512,
        device=device
    )

    model_E = SurrogateMLP(
        input_dim=5,
        output_dim=2,
        hidden_dim=128,
        num_layers=4,
        activation="relu"
    )

    save_path_E = "models/surrogate_Esurf.pth"

    train_surrogate(
        model=model_E,
        train_loader=train_loader_E,
        val_loader=val_loader_E,
        save_path=save_path_E,
        lr=1e-3,
        max_epochs=200,
        patience=20,
        device=device
    )

    print(f"Esurf surrogate saved to: {save_path_E}")

    # ------------------------------------------------------------
    # Train Hsurf surrogate
    # ------------------------------------------------------------
    print("\n=== Training Hsurf surrogate ===")

    train_loader_H, val_loader_H = load_surrogate_dataset(
        filename="Hsurf.npz",
        data_dir="data/surrogate",
        batch_size=512,
        device=device
    )

    model_H = SurrogateMLP(
        input_dim=5,
        output_dim=2,
        hidden_dim=128,
        num_layers=4,
        activation="relu"
    )

    save_path_H = "models/surrogate_Hsurf.pth"

    train_surrogate(
        model=model_H,
        train_loader=train_loader_H,
        val_loader=val_loader_H,
        save_path=save_path_H,
        lr=1e-3,
        max_epochs=200,
        patience=20,
        device=device
    )

    print(f"Hsurf surrogate saved to: {save_path_H}")

    print("\nAll surrogate models trained successfully.")


if __name__ == "__main__":
    main()
