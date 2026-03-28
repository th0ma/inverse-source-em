# ============================================================
# noise_robustness.py — 3-source regression noise robustness
# ============================================================

import os
import numpy as np
from sklearn.metrics import r2_score

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

NOISE_LEVELS = [0.0, 0.01, 0.03, 0.05, 0.10]


# ============================================================
# Main evaluation
# ============================================================

def evaluate_noise_robustness():
    print("\n=== 3-Source Regression: Noise Robustness ===\n")

    # Load model + scalers
    model = load_model(MODEL_PATH, INPUT_DIM)
    scaler_X, scaler_y = load_scalers(SCALER_X_PATH, SCALER_Y_PATH)

    # Load surrogate forward model
    sur_wrap = load_surrogate(PATH_E, PATH_H)

    # Generate clean evaluation dataset
    print(f"Generating base evaluation dataset (N={N_EVAL}, stage={EVAL_STAGE})...")
    X_raw, y_true = generate_eval_dataset(
        N_EVAL, EVAL_STAGE, sur_wrap, num_angles=NUM_ANGLES
    )

    results = []

    # --------------------------------------------------------
    # Noise sweep
    # --------------------------------------------------------
    for noise_frac in NOISE_LEVELS:

        print(f"\n--- Noise level: {int(noise_frac*100)}% ---")

        X_noisy = X_raw.copy()

        # Add noise to random angles
        if noise_frac > 0:
            feats_per_angle = X_noisy.shape[1] // NUM_ANGLES
            X_noisy = X_noisy.reshape(N_EVAL, NUM_ANGLES, feats_per_angle)

            k = int(noise_frac * NUM_ANGLES)

            for i in range(N_EVAL):
                idx = np.random.permutation(NUM_ANGLES)[:k]
                noise = 0.01 * np.random.randn(len(idx), feats_per_angle)
                X_noisy[i, idx] += noise

            X_noisy = X_noisy.reshape(N_EVAL, NUM_ANGLES * feats_per_angle)

        # Scale
        X_scaled = scaler_X.transform(X_noisy)

        # Forward pass
        rho_pred, cos_pred, sin_pred = forward_pass(model, X_scaled)

        # Cartesian
        y_pred = to_cartesian(rho_pred, cos_pred, sin_pred)

        # Per-source R²
        R2_A = r2_score(y_true[:, 0:2], y_pred[:, 0:2])
        R2_B = r2_score(y_true[:, 2:4], y_pred[:, 2:4])
        R2_C = r2_score(y_true[:, 4:6], y_pred[:, 4:6])

        # MAE
        MAE = np.mean(np.abs(y_pred - y_true))

        # Error distribution
        err = cartesian_error(y_true, y_pred)

        p50 = np.percentile(err, 50)
        p75 = np.percentile(err, 75)
        p90 = np.percentile(err, 90)
        p95 = np.percentile(err, 95)
        p99 = np.percentile(err, 99)
        max_err = np.max(err)

        print(f"R² A={R2_A:.4f}, B={R2_B:.4f}, C={R2_C:.4f}")
        print(f"MAE={MAE:.4f}")
        print(f"Percentiles: 50={p50:.4f}, 75={p75:.4f}, 90={p90:.4f}, 95={p95:.4f}, 99={p99:.4f}")
        print(f"Max error: {max_err:.4f}")

        results.append({
            "noise": noise_frac,
            "R2_A": R2_A,
            "R2_B": R2_B,
            "R2_C": R2_C,
            "MAE": MAE,
            "p50": p50,
            "p75": p75,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "max": max_err,
        })

    print("\n================ NOISE SWEEP COMPLETE ================\n")
    return results


# ============================================================
# CLI entrypoint
# ============================================================

if __name__ == "__main__":
    evaluate_noise_robustness()
