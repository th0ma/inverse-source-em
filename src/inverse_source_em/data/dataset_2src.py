"""
dataset_2src.py

Unified dataset builder for the two-source inverse EM problem.
Generates (X, Y) pairs using the surrogate forward model and
the sampling utilities.

Produces:
    - X: (N, 120) feature matrix
    - Y: (N, 4)   normalized Cartesian labels
    - train/test split
    - MinMax normalization
    - compressed .npz dataset
    - saved scalers (joblib)
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
    Build a full two-source dataset and save it to disk.

    Parameters
    ----------
    N_samples : int
        Number of samples to generate.
    theta : ndarray
        Observation angles (num_angles,)
    path_E, path_H : str
        Paths to surrogate .pth files.
    test_size : float
        Fraction of samples used for testing.
    out_dir : str
        Output directory for dataset and scalers.
    dataset_name : str
        Name of the saved .npz file.

    Returns
    -------
    dataset_path : str
        Path to the saved dataset.
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
