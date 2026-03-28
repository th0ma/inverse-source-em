# ============================================================
# timing.py — 3-source regression timing benchmarks
# ============================================================

import os
import time
import numpy as np

from .eval_utils_3src import (
    load_model,
    load_scalers,
    load_surrogate,
    generate_eval_dataset,
    forward_pass,
    to_cartesian,
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
# Main timing benchmark
# ============================================================

def evaluate_timing():
    print("\n=== 3-Source Regression: Timing Benchmark ===\n")

    # --------------------------------------------------------
    # 1. Load model + scalers
    # --------------------------------------------------------
    t0 = time.time()
    model = load_model(MODEL_PATH, INPUT_DIM)
    scaler_X, scaler_y = load_scalers(SCALER_X_PATH, SCALER_Y_PATH)
    t1 = time.time()

    load_time = t1 - t0
    print(f"Model + scaler load time: {load_time:.4f} s")

    # --------------------------------------------------------
    # 2. Load surrogate
    # --------------------------------------------------------
    t0 = time.time()
    sur_wrap = load_surrogate(PATH_E, PATH_H)
    t1 = time.time()

    surrogate_time = t1 - t0
    print(f"Surrogate load time: {surrogate_time:.4f} s")

    # --------------------------------------------------------
    # 3. Generate evaluation dataset
    # --------------------------------------------------------
    print(f"Generating evaluation dataset (N={N_EVAL}, stage={EVAL_STAGE})...")
    t0 = time.time()
    X_raw, y_true = generate_eval_dataset(
        N_EVAL, EVAL_STAGE, sur_wrap, num_angles=NUM_ANGLES
    )
    t1 = time.time()

    dataset_time = t1 - t0
    print(f"Dataset generation time: {dataset_time:.4f} s")

    # --------------------------------------------------------
    # 4. Scaling time
    # --------------------------------------------------------
    t0 = time.time()
    X_scaled = scaler_X.transform(X_raw)
    t1 = time.time()

    scaling_time = t1 - t0
    print(f"Scaling time: {scaling_time:.4f} s")

    # --------------------------------------------------------
    # 5. Forward pass time
    # --------------------------------------------------------
    t0 = time.time()
    rho_pred, cos_pred, sin_pred = forward_pass(model, X_scaled)
    t1 = time.time()

    forward_time = t1 - t0
    print(f"Forward pass time: {forward_time:.4f} s")

    # --------------------------------------------------------
    # 6. Cartesian conversion time
    # --------------------------------------------------------
    t0 = time.time()
    y_pred = to_cartesian(rho_pred, cos_pred, sin_pred)
    t1 = time.time()

    cart_time = t1 - t0
    print(f"Cartesian conversion time: {cart_time:.4f} s")

    # --------------------------------------------------------
    # 7. Throughput
    # --------------------------------------------------------
    throughput = N_EVAL / forward_time
    print(f"\nThroughput: {throughput:.2f} samples/sec")

    print("\n================ TIMING COMPLETE ================\n")

    return {
        "load_time": load_time,
        "surrogate_time": surrogate_time,
        "dataset_time": dataset_time,
        "scaling_time": scaling_time,
        "forward_time": forward_time,
        "cart_time": cart_time,
        "throughput": throughput,
    }


# ============================================================
# CLI entrypoint
# ============================================================

if __name__ == "__main__":
    evaluate_timing()
