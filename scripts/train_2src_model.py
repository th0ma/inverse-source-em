#!/usr/bin/env python
# coding: utf-8

"""
train_2src_model.py

Clean, modular, testable training script for the two-source regression model.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import joblib
import pandas as pd

from src.inverse_source_em.training.model_2src import build_model
from src.inverse_source_em.training.train_2src import train_model


# ------------------------------------------------------------
# Global CONFIG
# ------------------------------------------------------------
CONFIG = {
    "HIDDEN_DIMS": [512, 512, 256, 128],
    "OUTPUT_SIZE": 4,

    "NUM_EPOCHS": 2500,
    "PATIENCE": 150,
    "LEARNING_RATE": 1e-3,
    "CLIP_NORM": 1.0,
    "BATCH_SIZE_TRAIN": 128,
    "BATCH_SIZE_TEST": 256,

    "LAMBDA_AREA":   1.0,
    "LAMBDA_DISTA":  1.0,
    "LAMBDA_DISTB":  1.0,
    "LAMBDA_ANGLEA": 1.0,
    "LAMBDA_ANGLEB": 1.0,

    "TAIL_SCHEDULER_DISTA":  {"target_p99": 0.01, "min_w": 1.0, "max_w": 3.0, "delta": 0.05},
    "TAIL_SCHEDULER_DISTB":  {"target_p99": 0.01, "min_w": 1.0, "max_w": 3.0, "delta": 0.05},
    "TAIL_SCHEDULER_ANGLEA": {"target_p99": 0.01, "min_w": 1.0, "max_w": 5.0, "delta": 0.05},
    "TAIL_SCHEDULER_ANGLEB": {"target_p99": 0.01, "min_w": 1.0, "max_w": 5.0, "delta": 0.05},

    "DATA_DIR": "./data/regression_2src",
    "TRAINING_DIR": "./models",

    "SEED": 1234,
}


# ------------------------------------------------------------
# Device selection
# ------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")


# ------------------------------------------------------------
# Load dataset + scalers
# ------------------------------------------------------------
def load_data(config):
    dataset_file = os.path.join(config["DATA_DIR"], "dataset_2src_fullfield_30obs_20k.npz")
    scaler_X_file = dataset_file.replace(".npz", "_scaler_X.pkl")
    scaler_Y_file = dataset_file.replace(".npz", "_scaler_Y.pkl")

    print("\nLoading dataset...")
    data = np.load(dataset_file)

    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_test  = data["X_test"]
    Y_test  = data["Y_test"]

    scaler_X = joblib.load(scaler_X_file)
    scaler_Y = joblib.load(scaler_Y_file)

    print("Dataset loaded.")
    print("  X_train:", X_train.shape)
    print("  Y_train:", Y_train.shape)
    print("  X_test :", X_test.shape)
    print("  Y_test :", Y_test.shape)

    return X_train, Y_train, X_test, Y_test, scaler_X, scaler_Y


# ------------------------------------------------------------
# Build DataLoaders
# ------------------------------------------------------------
def build_loaders(X_train, Y_train, X_test, Y_test, device, config):
    X_train_t = torch.tensor(X_train, dtype=torch.float64, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float64, device=device)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float64, device=device)
    Y_test_t  = torch.tensor(Y_test,  dtype=torch.float64, device=device)

    train_loader = DataLoader(
        list(zip(X_train_t, Y_train_t)),
        batch_size=config["BATCH_SIZE_TRAIN"],
        shuffle=True
    )

    test_loader = DataLoader(
        list(zip(X_test_t, Y_test_t)),
        batch_size=config["BATCH_SIZE_TEST"],
        shuffle=False
    )

    return train_loader, test_loader


# ------------------------------------------------------------
# Train + save model
# ------------------------------------------------------------
def train_and_save(train_loader, test_loader, X_train, config, device):
    input_size = X_train.shape[1]

    model = build_model(
        input_size=input_size,
        hidden_dims=config["HIDDEN_DIMS"],
        output_size=config["OUTPUT_SIZE"]
    ).to(torch.float64).to(device)

    print("\nModel instantiated:")
    print(model)

    trained_model, logs_df = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )

    os.makedirs(config["TRAINING_DIR"], exist_ok=True)

    model_path = os.path.join(config["TRAINING_DIR"], "best_model_2src.pth")
    logs_path  = os.path.join(config["TRAINING_DIR"], "training_logs_2src.csv")

    torch.save({
        "model_state_dict": trained_model.state_dict(),
        "input_size": input_size,
        "hidden_dims": config["HIDDEN_DIMS"],
        "output_size": config["OUTPUT_SIZE"],
        "CONFIG": config
    }, model_path)

    logs_df.to_csv(logs_path, index=False)

    print("\nTraining complete.")
    print("Model saved to:", model_path)
    print("Logs saved to :", logs_path)


# ------------------------------------------------------------
# Main entrypoint
# ------------------------------------------------------------
def main():
    torch.set_default_dtype(torch.float64)
    device = get_device()

    X_train, Y_train, X_test, Y_test, scaler_X, scaler_Y = load_data(CONFIG)
    train_loader, test_loader = build_loaders(X_train, Y_train, X_test, Y_test, device, CONFIG)
    train_and_save(train_loader, test_loader, X_train, CONFIG, device)


if __name__ == "__main__":
    main()
