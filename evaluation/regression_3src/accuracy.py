"""
accuracy.py — 3‑source regression accuracy evaluation.

This module evaluates the clean (noise‑free) accuracy of the 3‑source
regression model on a freshly generated evaluation dataset. Each sample
contains three point sources with polar coordinates
$\left(\rho_1,\\,\phi_1\right)$,
$\left(\rho_2,\\,\phi_2\right)$,
$\left(\rho_3,\\,\phi_3\right)$
and strengths $I_1, I_2, I_3$, together with the boundary fields
$E_r,\\,E_i,\\,H_r,\\,H_i$ computed by the canonical surrogate models.

The model predicts $\rho$, $\cos\phi$, $\sin\phi$ for each source. These
are converted to Cartesian coordinates via:
$$
x = \rho \cos\phi, \qquad y = \rho \sin\phi
$$

The module reports:
- $R^2$ scores for each source in Cartesian space
- mean absolute error (MAE) over all coordinates

Run as a standalone script:
    python accuracy.py
"""



# ============================================================
# accuracy.py — 3-source regression accuracy evaluation
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
    compute_mae,
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

def evaluate_accuracy():
    print("\n=== 3-Source Regression: Accuracy Evaluation ===\n")

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

    # Split true values
    xA_true, yA_true = y_true[:, 0], y_true[:, 1]
    xB_true, yB_true = y_true[:, 2], y_true[:, 3]
    xC_true, yC_true = y_true[:, 4], y_true[:, 5]

    # Split predictions
    xA_pred, yA_pred = y_pred[:, 0], y_pred[:, 1]
    xB_pred, yB_pred = y_pred[:, 2], y_pred[:, 3]
    xC_pred, yC_pred = y_pred[:, 4], y_pred[:, 5]

    # Compute R²
    R2_A = r2_score(np.column_stack([xA_true, yA_true]),
                    np.column_stack([xA_pred, yA_pred]))

    R2_B = r2_score(np.column_stack([xB_true, yB_true]),
                    np.column_stack([xB_pred, yB_pred]))

    R2_C = r2_score(np.column_stack([xC_true, yC_true]),
                    np.column_stack([xC_pred, yC_pred]))

    # Compute MAE
    MAE = compute_mae(y_true, y_pred)

    # Print results
    print("\n================ CLEAN ACCURACY RESULTS ================")
    print(f"R² Source A: {R2_A:.4f}")
    print(f"R² Source B: {R2_B:.4f}")
    print(f"R² Source C: {R2_C:.4f}")
    print(f"Mean Absolute Error (Cartesian): {MAE:.4f}")
    print("========================================================\n")

    return {
        "R2_A": R2_A,
        "R2_B": R2_B,
        "R2_C": R2_C,
        "MAE": MAE,
    }


# ============================================================
# CLI entrypoint
# ============================================================

if __name__ == "__main__":
    evaluate_accuracy()
