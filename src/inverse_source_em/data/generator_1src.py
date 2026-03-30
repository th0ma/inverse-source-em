"""
Dataset generator for Regression Problem I (single-source localization).

This module constructs the full inverse dataset for the 1‑source localization
problem using the surrogate forward model. The pipeline performs:

1. Sampling of random single-source configurations.
2. Surrogate-based forward evaluation of Esurf and Hsurf.
3. Two-pass normalization:
       - Pass 1: compute global maxima (ymax_E, ymax_H)
       - Pass 2: normalize all samples using these maxima
4. Assembly of the dataset:
       X : (N, 4*M)  → normalized [E_r, E_i, H_r, H_i]
       Y : (N, 3)    → true parameters [x, y, I]
5. Saving the dataset in compressed NPZ format.

The output is fully compatible with:
- Regression1SrcDataset (dataset_1src.py)
- the training pipeline for Problem I
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
    Compute Esurf and Hsurf for a single source using the surrogate model.

    Parameters
    ----------
    sur : SurrogateEM
        Unified surrogate forward model.
    rho_s : float
        Radial coordinate of the source.
    phi_s : float
        Angular coordinate of the source.
    theta_obs : ndarray of shape (M,)
        Observation angles.

    Returns
    -------
    E_real, E_imag, H_real, H_imag : ndarray of shape (M,)
        Real and imaginary parts of the boundary fields.
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
    Build the inverse dataset for Regression Problem I (single-source localization).

    This function performs the full 2‑pass dataset construction:
    - Pass 1: compute global maxima for normalization
    - Pass 2: compute normalized fields and assemble X, Y

    Parameters
    ----------
    n_sources : int
        Number of samples to generate.
    num_angles : int
        Number of observation angles.
    path_E, path_H : str
        Paths to the trained surrogate .pth files for Esurf and Hsurf.
    rho_min, rho_max : float
        Radial sampling limits for the source.

    Returns
    -------
    X : ndarray of shape (N, 4*M)
        Normalized boundary fields.
    Y : ndarray of shape (N, 3)
        True source parameters [x, y, I].
    theta_obs : ndarray of shape (M,)
        Observation angles.
    ymax_E : float
        Global maximum used to normalize E_r and E_i.
    ymax_H : float
        Global maximum used to normalize H_r and H_i.
    R : float
        Physical radius of the domain.

    Notes
    -----
    - Intensities are fixed to I = 1.0 but included in Y for completeness.
    - All computations use float64 for numerical stability.
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
    Save the 1‑source regression dataset in compressed NPZ format.

    Parameters
    ----------
    save_path : str
        Output file path (e.g., "data/regression_1src/dataset_1src.npz").
    X : ndarray
        Normalized boundary fields, shape (N, 4*M).
    Y : ndarray
        True parameters, shape (N, 3).
    theta_obs : ndarray
        Observation angles.
    ymax_E : float
        Normalization constant for E channels.
    ymax_H : float
        Normalization constant for H channels.
    R : float
        Physical radius.

    Notes
    -----
    - The directory is created automatically if needed.
    - Uses NumPy's compressed NPZ format.
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
