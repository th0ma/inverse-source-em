"""
Unified dataset builder for the two-source inverse EM problem.

This module generates a full dataset for the 2‑source localization problem
using the surrogate forward model. Each sample consists of:

    X : (120,) feature vector
        Flattened boundary fields:
            [Re(E), Im(E), Re(H), Im(H)] over all observation angles

    Y : (4,) normalized Cartesian labels
        True source parameters:
            [x1, y1, x2, y2]

The pipeline performs:
- surrogate-based forward evaluation
- train/test split
- MinMax normalization for X and Y
- saving the dataset in compressed NPZ format
- saving the fitted scalers (joblib)

The output is fully compatible with the training pipeline for the 2‑source
regression model.
"""

import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

from inverse_source_em.data.generator_2src import (
    load_surrogate,
    generate_sample
)


# ------------------------------------------------------------
# 1. Dataset builder
# ------------------------------------------------------------
def build_dataset_2src(
    N_samples,
    theta,
    path_E,
    path_H,
    test_size=0.30,
    out_dir="./data",
    dataset_name="dataset_2src.npz"
):
    """
    Build and save a full two-source inverse EM dataset.

    This function:
    - loads the surrogate forward model
    - generates N_samples synthetic examples
    - splits into train/test subsets
    - applies MinMax normalization to X and Y
    - saves the dataset and scalers to disk

    Parameters
    ----------
    N_samples : int
        Number of samples to generate.
    theta : ndarray of shape (num_angles,)
        Observation angles in radians.
    path_E, path_H : str
        Paths to the trained surrogate .pth files for Esurf and Hsurf.
    test_size : float, optional
        Fraction of samples used for testing. Default is 0.30.
    out_dir : str, optional
        Output directory for dataset and scalers.
    dataset_name : str, optional
        Name of the saved NPZ dataset file.

    Returns
    -------
    dataset_path : str
        Full path to the saved dataset file.

    Notes
    -----
    - X has shape (N, 120) assuming 30 angles × 4 channels.
    - Y has shape (N, 4) containing normalized Cartesian coordinates.
    - MinMaxScaler is used for both X and Y.
    - The surrogate wrapper is loaded via load_surrogate().
    """

    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------------
    # Load surrogate
    # --------------------------------------------------------
    sur_wrap, R = load_surrogate(path_E, path_H)

    # --------------------------------------------------------
    # Generate samples
    # --------------------------------------------------------
    X_list = []
    Y_list = []

    print(f"Generating {N_samples} two-source samples...\n")

    for _ in tqdm(range(N_samples), ncols=80):
        X, Y = generate_sample(theta, sur_wrap, R)
        X_list.append(X)
        Y_list.append(Y)

    X = np.array(X_list, dtype=np.float64)
    Y = np.array(Y_list, dtype=np.float64)

    print("\nDataset generation complete.")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=42
    )

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled  = scaler_X.transform(X_test)

    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    Y_test_scaled  = scaler_Y.transform(Y_test)

    # --------------------------------------------------------
    # Save dataset
    # --------------------------------------------------------
    dataset_path = os.path.join(out_dir, dataset_name)

    np.savez_compressed(
        dataset_path,
        X_train=X_train_scaled,
        Y_train=Y_train_scaled,
        X_test=X_test_scaled,
        Y_test=Y_test_scaled,
        theta=theta,
        R=R,
        N_samples=N_samples,
        test_size=test_size
    )

    print("\nSaved dataset:")
    print(" →", dataset_path)

    # --------------------------------------------------------
    # Save scalers
    # --------------------------------------------------------
    scaler_X_path = dataset_path.replace(".npz", "_scaler_X.pkl")
    scaler_Y_path = dataset_path.replace(".npz", "_scaler_Y.pkl")

    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_Y, scaler_Y_path)

    print("Saved scalers:")
    print(" →", scaler_X_path)
    print(" →", scaler_Y_path)

    return dataset_path
