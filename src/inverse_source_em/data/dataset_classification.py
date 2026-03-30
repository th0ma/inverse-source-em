"""
Dataset generation utilities for the source-count classification problem.

This module generates datasets for classifying the number of active sources
(S = 1, 2, 3, 4, 5) inside a circular domain. It uses the unified surrogate
forward model (SurrogateWrapper) to compute boundary fields for randomly
sampled source configurations.

The generated dataset contains:

    X : (N, 4, num_angles)
        Boundary field channels:
            [Re(E), Im(E), Re(H), Im(H)]

    y : (N,)
        Class index in {0, 1, 2, 3, 4}, corresponding to S = 1..5 sources.

    theta : ndarray
        Observation angles.

    R : float
        Physical radius of the domain.

The dataset is split into train/val/test subsets and standardized per channel.
Scalers are saved as joblib files for later use during inference.

Public API
----------
generate_classification_dataset(...)
"""

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

from inverse_source_em.data.sampling_classification import sample_sources
from inverse_source_em.data.generator_classification import forward_fields


# ---------------------------------------------------------------------
# Utility: generate samples for a single class S
# ---------------------------------------------------------------------

def _generate_class_samples(S, N, R, theta, sur_wrap, rng):
    """
    Generate N samples for class S (S sources).

    Parameters
    ----------
    S : int
        Number of sources (1–5).
    N : int
        Number of samples to generate.
    R : float
        Physical radius of the domain.
    theta : ndarray of shape (num_angles,)
        Observation angles in radians.
    sur_wrap : SurrogateWrapper
        Unified surrogate forward model providing Esurf/Hsurf.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    X : ndarray of shape (N, 4, num_angles)
        Boundary field channels:
            [Re(E), Im(E), Re(H), Im(H)]
    y : ndarray of shape (N,)
        Class labels (S - 1).
    """
    X_list = []
    y_list = []

    for _ in range(N):
        sources = sample_sources(S, R, rng)
        feat = forward_fields(sources, theta, sur_wrap)
        X_list.append(feat)
        y_list.append(S - 1)

    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.int64)


# ---------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------

def generate_classification_dataset(
    out_dir,
    sur_wrap,
    R,
    theta,
    samples_per_class,
    seed=2025
):
    """
    Generate a full classification dataset for S = 1..5 sources.

    This function:
    - samples source configurations for each class
    - computes boundary fields using the surrogate forward model
    - concatenates and shuffles all samples
    - splits into train/val/test subsets
    - standardizes each channel independently
    - saves the dataset and scalers to disk

    Parameters
    ----------
    out_dir : str
        Directory where the dataset and scalers will be saved.
    sur_wrap : SurrogateWrapper
        Unified surrogate forward model.
    R : float
        Physical radius of the domain.
    theta : ndarray of shape (num_angles,)
        Observation angles in radians.
    samples_per_class : list or tuple of length 5
        Number of samples for classes S = 1..5.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    out_file : str
        Path to the saved .npz dataset.

    Notes
    -----
    - The dataset is saved in compressed NPZ format.
    - Four StandardScaler objects (one per channel) are saved as joblib files.
    - The output dataset uses float32 for compact storage.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # Generate each class separately
    # ------------------------------------------------------------
    X_all = []
    y_all = []

    for S, N in enumerate(samples_per_class, start=1):
        Xc, yc = _generate_class_samples(S, N, R, theta, sur_wrap, rng)
        X_all.append(Xc)
        y_all.append(yc)

    # Concatenate all classes
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    # ------------------------------------------------------------
    # Shuffle
    # ------------------------------------------------------------
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]

    # ------------------------------------------------------------
    # Train/Val/Test split
    # ------------------------------------------------------------
    N_total = len(X)
    N_train = int(0.7 * N_total)
    N_val   = int(0.15 * N_total)
    N_test  = N_total - N_train - N_val

    X_train = X[:N_train]
    y_train = y[:N_train]

    X_val = X[N_train:N_train+N_val]
    y_val = y[N_train:N_train+N_val]

    X_test = X[N_train+N_val:]
    y_test = y[N_train+N_val:]

    # ------------------------------------------------------------
    # Standardization per channel
    # ------------------------------------------------------------
    scalers = [StandardScaler() for _ in range(4)]

    X_train_std = X_train.copy()
    for ci in range(4):
        scalers[ci].fit(X_train[:, ci, :])
        X_train_std[:, ci, :] = scalers[ci].transform(X_train[:, ci, :])

    X_val_std = X_val.copy()
    X_test_std = X_test.copy()

    for ci in range(4):
        X_val_std[:, ci, :]  = scalers[ci].transform(X_val[:, ci, :])
        X_test_std[:, ci, :] = scalers[ci].transform(X_test[:, ci, :])

    # Save scalers
    for ci in range(4):
        joblib.dump(scalers[ci], os.path.join(out_dir, f"scaler_channel_{ci}.pkl"))

    # ------------------------------------------------------------
    # Save dataset
    # ------------------------------------------------------------
    out_file = os.path.join(out_dir, "dataset_classification.npz")

    np.savez_compressed(
        out_file,
        X_train=X_train_std.astype(np.float32),
        y_train=y_train,
        X_val=X_val_std.astype(np.float32),
        y_val=y_val,
        X_test=X_test_std.astype(np.float32),
        y_test=y_test,
        theta=theta,
        R=R
    )

    return out_file
