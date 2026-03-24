"""
Dataset generator for Regression Problem I (single-source localization).

This module:
- loads the canonical surrogate forward model
- computes E_real, E_imag, H_real, H_imag for each source
- performs 2-pass normalization (global maxima)
- assembles the dataset (X, Y)
- saves dataset_1src.npz

The output is compatible with dataset_1src.py and the training pipeline.
"""

import os
import numpy as np
from tqdm import tqdm

from inverse_source_em.data.sampling_1src import sample_sources_1src, sample_angles
from inverse_source_em.surrogate.surrogate import SurrogateEM
from inverse_source_em.physics.physics_tm import PhysicsTM


# ------------------------------------------------------------
# Forward computation using the surrogate
# ------------------------------------------------------------

def compute_fields_surrogate(sur, rho_s, phi_s, theta_obs):
    """
    Compute Esurf, Hsurf for a single source using the surrogate model.

    Returns
    -------
    E_real, E_imag, H_real, H_imag : (M,) arrays
    """
    E = sur.Esurf(rho_s, phi_s, theta_obs)
    H = sur.Hsurf(rho_s, phi_s, theta_obs)

    return (
        np.real(E),
        np.imag(E),
        np.real(H),
        np.imag(H),
    )


# ------------------------------------------------------------
# Build dataset (2-pass generation)
# ------------------------------------------------------------

def build_dataset_1src(
    n_sources: int,
    num_angles: int,
    path_E: str,
    path_H: str,
    rho_min: float = 0.05,
    rho_max: float = 0.95,
):
    """
    Build inverse dataset for Regression Problem I.

    Parameters
    ----------
    n_sources : int
        Number of samples.
    num_angles : int
        Number of observation angles.
    path_E, path_H : str
        Paths to surrogate .pth files.
    rho_min, rho_max : float
        Radial sampling limits.

    Returns
    -------
    X : (N, 4*M) ndarray
    Y : (N, 3) ndarray
    theta_obs : (M,) ndarray
    ymax_E, ymax_H : float
    R : float
    """

    # Instantiate physics + surrogate
    phys = PhysicsTM()
    sur = SurrogateEM(path_E=path_E, path_H=path_H)

    R = phys.R
    theta_obs = sample_angles(num_angles)

    # ------------------------------------------------------------
    # 1. Sample all sources
    # ------------------------------------------------------------
    rho_all, phi_all, x_all, y_all = sample_sources_1src(
        n_sources, R=R, rho_min=rho_min, rho_max=rho_max
    )

    # ------------------------------------------------------------
    # 2. PASS 1 — compute global maxima
    # ------------------------------------------------------------
    ymax_E = 0.0
    ymax_H = 0.0

    for i in tqdm(range(n_sources), desc="Pass 1: global maxima"):
        E_r, E_i, H_r, H_i = compute_fields_surrogate(
            sur, rho_all[i], phi_all[i], theta_obs
        )

        ymax_E = max(ymax_E, np.max(np.abs(E_r)), np.max(np.abs(E_i)))
        ymax_H = max(ymax_H, np.max(np.abs(H_r)), np.max(np.abs(H_i)))

    # ------------------------------------------------------------
    # 3. PASS 2 — build dataset
    # ------------------------------------------------------------
    M = num_angles
    X = np.zeros((n_sources, 4 * M), dtype=np.float64)
    Y = np.zeros((n_sources, 3), dtype=np.float64)

    for i in tqdm(range(n_sources), desc="Pass 2: dataset assembly"):
        rho_s = rho_all[i]
        phi_s = phi_all[i]

        # True parameters
        x_s = x_all[i]
        y_s = y_all[i]
        I_s = 1.0  # intensity is fixed but predicted

        # Forward fields
        E_r, E_i, H_r, H_i = compute_fields_surrogate(
            sur, rho_s, phi_s, theta_obs
        )

        # Normalize
        E_r_n = E_r / ymax_E
        E_i_n = E_i / ymax_E
        H_r_n = H_r / ymax_H
        H_i_n = H_i / ymax_H

        # Fill X
        X[i, :] = np.concatenate([E_r_n, E_i_n, H_r_n, H_i_n])

        # Fill Y
        Y[i, :] = np.array([x_s, y_s, I_s])

    return X, Y, theta_obs, ymax_E, ymax_H, R


# ------------------------------------------------------------
# Save dataset
# ------------------------------------------------------------

def save_dataset_1src(
    save_path: str,
    X,
    Y,
    theta_obs,
    ymax_E,
    ymax_H,
    R,
):
    """
    Save dataset_1src.npz in compressed format.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.savez_compressed(
        save_path,
        X=X,
        Y=Y,
        theta_obs=theta_obs,
        ymax_E=ymax_E,
        ymax_H=ymax_H,
        R=R,
    )
