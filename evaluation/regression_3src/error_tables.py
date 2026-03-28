# ============================================================
# error_tables.py — 3-source regression error statistics
# ============================================================

import os
import numpy as np

from .eval_utils_3src import (
    load_model,
    load_scalers,
    load_surrogate,
    generate_eval_dataset,
    forward_pass,
    to_cartesian,
    cartesian_error,
)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT, "data", "regression_3src")
MODEL_DIR = os.path.join(ROOT, "models", "regression_3src")
SURR_DIR = os.path.join(ROOT, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "best_model_stage_8.pt")
SCALER_X_PATH = os.path.join(DATA_DIR, "stage_8_scaler_X.pkl")
SCALER_Y_PATH = os.path.join(DATA_DIR, "stage_8_scaler_y.pkl")

PATH_E = os.path.join(SURR_DIR, "surrogate_Esurf.pth")
PATH_H = os.path.join(SURR_DIR, "surrogate_Hsurf.pth")

NUM_ANGLES = 30
INPUT_DIM = 4 * NUM_ANGLES
N_EVAL = 5000
EVAL_STAGE = 8


# ============================================================
# Main evaluation
# ============================================================

def evaluate_error_tables():
    print("\n=== 3-Source Regression: Error Tables ===\n")

    # Load model + scalers
    model = load_model(MODEL_PATH, INPUT_DIM)
    scaler_X, scaler_y = load_scalers(SCALER_X_PATH, SCALER_Y_PATH)

    # Load surrogate forward model
    sur_wrap = load_surrogate(PATH_E, PATH_H)

    # Generate clean evaluation dataset
    print(f"Generating evaluation dataset (N={N_EVAL}, stage={EVAL_STAGE})...")
    X_raw, y_true = generate_eval_dataset(
        N_EVAL, EVAL_STAGE, sur_wrap, num_angles=NUM_ANGLES
    )

    # Scale inputs
    X_scaled = scaler_X.transform(X_raw)

    # Forward pass
    rho_pred, cos_pred, sin_pred = forward_pass(model, X_scaled)

    # Convert to Cartesian
    y_pred = to_cartesian(rho_pred, cos_pred, sin_pred)

    # --------------------------------------------------------
    # Per-source errors
    # --------------------------------------------------------
    true_A = y_true[:, 0:2]
    true_B = y_true[:, 2:4]
    true_C = y_true[:, 4:6]

    pred_A = y_pred[:, 0:2]
    pred_B = y_pred[:, 2:4]
    pred_C = y_pred[:, 4:6]

    err_A = cartesian_error(true_A, pred_A)
    err_B = cartesian_error(true_B, pred_B)
    err_C = cartesian_error(true_C, pred_C)

    # --------------------------------------------------------
    # Max-triplet error
    # --------------------------------------------------------
    err_max_triplet = np.maximum.reduce([err_A, err_B, err_C])

    # --------------------------------------------------------
    # Print tables
    # --------------------------------------------------------
    print("\n================ ERROR TABLES (CLEAN) ================\n")

    print("Per-source MAE:")
    print(f"  Source A: {np.mean(err_A):.4f}")
    print(f"  Source B: {np.mean(err_B):.4f}")
    print(f"  Source C: {np.mean(err_C):.4f}\n")

    print("Per-source RMSE:")
    print(f"  Source A: {np.sqrt(np.mean(err_A**2)):.4f}")
    print(f"  Source B: {np.sqrt(np.mean(err_B**2)):.4f}")
    print(f"  Source C: {np.sqrt(np.mean(err_C**2)):.4f}\n")

    print("Per-source Median Error:")
    print(f"  Source A: {np.median(err_A):.4f}")
    print(f"  Source B: {np.median(err_B):.4f}")
    print(f"  Source C: {np.median(err_C):.4f}\n")

    print("Max-triplet error statistics:")
    print(f"  Mean: {np.mean(err_max_triplet):.4f}")
    print(f"  Median: {np.median(err_max_triplet):.4f}")
    print(f"  90th percentile: {np.percentile(err_max_triplet, 90):.4f}")
    print(f"  95th percentile: {np.percentile(err_max_triplet, 95):.4f}")
    print(f"  99th percentile: {np.percentile(err_max_triplet, 99):.4f}")
    print(f"  Max: {np.max(err_max_triplet):.4f}")

    print("\n======================================================\n")

    return {
        "err_A": err_A,
        "err_B": err_B,
        "err_C": err_C,
        "err_max_triplet": err_max_triplet,
    }


# ============================================================
# CLI entrypoint
# ============================================================

if __name__ == "__main__":
    evaluate_error_tables()
