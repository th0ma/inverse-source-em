"""
eval_utils_3src.py — Shared utilities for 3‑source regression evaluation.

This module provides the common functionality required by all evaluation
components of the 3‑source regression pipeline. Each sample contains three
point sources with polar coordinates
$\left(\rho_1,\\,\phi_1\right)$,
$\left(\rho_2,\\,\phi_2\right)$,
$\left(\rho_3,\\,\phi_3\right)$
and strengths $I_1, I_2, I_3$, together with the boundary fields
$E_r,\\,E_i,\\,H_r,\\,H_i$ computed by the canonical surrogate models.

The utilities include:

1. **Model + scaler loading**  
   Loads the trained 3‑source regression model and the corresponding
   input/output scalers.

2. **Surrogate forward model**  
   Wraps the canonical electromagnetic surrogate to compute boundary fields
   for arbitrary 3‑source configurations.

3. **Canonical ordering**  
   Ensures a consistent ordering of the three sources by sorting
   $\left(\rho,\\,\phi\right)$ pairs.

4. **Angular difference helper**  
   Computes wrapped angular differences:
   $$
   \Delta\phi = \left\lvert \left( \phi_{\mathrm{pred}} - \phi_{\mathrm{true}}
   + \pi \right) \bmod 2\pi - \pi \right\rvert
   $$

5. **Forward model (E + H)**  
   Computes the total boundary fields from the three sources.

6. **Evaluation dataset generator**  
   Generates clean evaluation datasets for geometry levels 1–8, enforcing
   minimum radial and angular separations.

7. **Forward‑pass utilities**  
   Converts scaled inputs to model predictions and transforms
   $\left(\rho,\\,\cos\phi,\\,\sin\phi\right)$ into Cartesian coordinates:
   $$
   x = \rho \cos\phi, \qquad y = \rho \sin\phi
   $$

8. **Error metrics**  
   Provides $R^2$, MAE, and Cartesian distance errors.

This module is imported by all 3‑source evaluation scripts:
`accuracy.py`, `error_tables.py`, `noise_robustness.py`, `timing.py`,
and `run_all.py`.
"""


# ============================================================
# eval_utils_3src.py — Shared utilities for 3-source regression evaluation
# ============================================================

import os
import numpy as np
import torch
from torch import nn
from sklearn.metrics import r2_score
import joblib
from tqdm import tqdm

# ------------------------------------------------------------
# Torch settings
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

# ------------------------------------------------------------
# Import project modules
# ------------------------------------------------------------
from inverse_source_em.training.model_3src import ThreeSourceMultiHeadBig
from inverse_source_em.surrogate.surrogate import SurrogateEM
from inverse_source_em.surrogate.surrogate_wrapper import SurrogateWrapper
from inverse_source_em.physics.physics_tm import PhysicsTM


# ============================================================
# 1. Model + scaler loading
# ============================================================

def load_model(model_path, input_dim):
    model = ThreeSourceMultiHeadBig(input_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def load_scalers(scaler_X_path, scaler_y_path):
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    return scaler_X, scaler_y


# ============================================================
# 2. Surrogate forward model
# ============================================================

def load_surrogate(path_E, path_H):
    phys = PhysicsTM()
    R = phys.R
    sur = SurrogateEM(path_E=path_E, path_H=path_H, R=R)
    return SurrogateWrapper(sur)


# ============================================================
# 3. Canonical ordering for 3 sources
# ============================================================

def canonical_order_three(rho1, phi1, rho2, phi2, rho3, phi3):
    pts = [(rho1, phi1), (rho2, phi2), (rho3, phi3)]
    pts_sorted = sorted(pts, key=lambda t: (t[0], t[1]))
    (rho1, phi1), (rho2, phi2), (rho3, phi3) = pts_sorted
    return rho1, phi1, rho2, phi2, rho3, phi3


# ============================================================
# 4. Angular difference helper
# ============================================================

def ang_diff(a, b):
    return np.abs(((a - b + np.pi) % (2*np.pi)) - np.pi)


# ============================================================
# 5. Forward model (E + H)
# ============================================================

def get_features_three_sources(sur_wrap, theta, rho1, phi1, rho2, phi2, rho3, phi3):
    E1 = sur_wrap.Esurf(rho1, phi1, theta)
    E2 = sur_wrap.Esurf(rho2, phi2, theta)
    E3 = sur_wrap.Esurf(rho3, phi3, theta)

    H1 = sur_wrap.Hsurf(rho1, phi1, theta)
    H2 = sur_wrap.Hsurf(rho2, phi2, theta)
    H3 = sur_wrap.Hsurf(rho3, phi3, theta)

    E_total = E1 + E2 + E3
    H_total = H1 + H2 + H3

    Ere = np.real(E_total)
    Eim = np.imag(E_total)
    Hre = np.real(H_total)
    Him = np.imag(H_total)

    return np.concatenate([Ere, Eim, Hre, Him], axis=0).astype(np.float64)


# ============================================================
# 6. Evaluation dataset generator
# ============================================================

GEOMETRY_LEVELS = {
    1: {"dr_min": 0.10, "dphi_min_deg": 40},
    2: {"dr_min": 0.08, "dphi_min_deg": 25},
    3: {"dr_min": 0.08, "dphi_min_deg": 20},
    4: {"dr_min": 0.07, "dphi_min_deg": 15},
    5: {"dr_min": 0.06, "dphi_min_deg": 10},
    6: {"dr_min": 0.05, "dphi_min_deg": 8},
    7: {"dr_min": 0.05, "dphi_min_deg": 6},
    8: {"dr_min": 0.05, "dphi_min_deg": 5},
}


def generate_eval_dataset(num_samples, geom_level, sur_wrap, num_angles=30):
    params = GEOMETRY_LEVELS[geom_level]
    dr_min = params["dr_min"]
    dphi_min = np.deg2rad(params["dphi_min_deg"])

    theta = np.linspace(0, 2*np.pi, num_angles, endpoint=False)

    X_list = []
    Y_list = []

    pbar = tqdm(total=num_samples, desc=f"Eval Level {geom_level}")

    while len(X_list) < num_samples:

        rho = np.random.uniform(0.1, 0.9, size=3)
        phi = np.random.uniform(-np.pi, np.pi, size=3)

        ok = True
        for i in range(3):
            for j in range(i+1, 3):
                if abs(rho[i] - rho[j]) < dr_min:
                    ok = False
                if ang_diff(phi[i], phi[j]) < dphi_min:
                    ok = False
        if not ok:
            continue

        rho1, phi1, rho2, phi2, rho3, phi3 = canonical_order_three(
            rho[0], phi[0], rho[1], phi[1], rho[2], phi[2]
        )

        feats = get_features_three_sources(
            sur_wrap, theta, rho1, phi1, rho2, phi2, rho3, phi3
        )

        y_vec = [
            rho1*np.cos(phi1), rho1*np.sin(phi1),
            rho2*np.cos(phi2), rho2*np.sin(phi2),
            rho3*np.cos(phi3), rho3*np.sin(phi3)
        ]

        X_list.append(feats)
        Y_list.append(y_vec)
        pbar.update(1)

    pbar.close()

    return np.array(X_list, dtype=np.float64), np.array(Y_list, dtype=np.float64)


# ============================================================
# 7. Forward pass utilities
# ============================================================

def forward_pass(model, X_scaled):
    X_t = torch.tensor(X_scaled, dtype=torch.float64).to(DEVICE)
    with torch.no_grad():
        rho_pred, cos_pred, sin_pred = model(X_t)
    return (
        rho_pred.cpu().numpy(),
        cos_pred.cpu().numpy(),
        sin_pred.cpu().numpy()
    )


def to_cartesian(rho_pred, cos_pred, sin_pred):
    phi_pred = np.arctan2(sin_pred, cos_pred)
    x_pred = rho_pred * np.cos(phi_pred)
    y_pred = rho_pred * np.sin(phi_pred)
    return np.column_stack([
        x_pred[:,0], y_pred[:,0],
        x_pred[:,1], y_pred[:,1],
        x_pred[:,2], y_pred[:,2]
    ])


# ============================================================
# 8. Error metrics
# ============================================================

def compute_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def compute_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def cartesian_error(y_true, y_pred):
    return np.linalg.norm(y_pred - y_true, axis=1)
