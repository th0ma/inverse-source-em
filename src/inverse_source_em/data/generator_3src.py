"""
Dataset generator for the 3‑source regression pipeline.

This module implements the full dataset generation workflow for the
three‑source inverse EM problem. It provides:

1. A unified surrogate‑based forward model for computing Esurf/Hsurf
   for three sources.

2. Feature extraction:
       X = [Re(E), Im(E), Re(H), Im(H)]
   with total length 4 * num_angles.

3. Stage‑based dataset generation:
   Each geometry stage imposes different constraints on the relative
   positions of the three sources (see sampling_3src.py).

4. Scaling and saving utilities:
   - Train/test split
   - MinMax normalization for X and y
   - Saving arrays and scalers per stage

The generator follows the same structure as the 1‑source and 2‑source
pipelines, but extended to handle the combinatorial geometry of 3 sources.
"""

import os
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from inverse_source_em.surrogate.surrogate import SurrogateEM
from inverse_source_em.surrogate.surrogate_wrapper import SurrogateWrapper
from inverse_source_em.physics.physics_tm import PhysicsTM

from .sampling_3src import (
    GEOMETRY_LEVELS,
    ang_diff,
    canonical_order_three,
    sample_three_sources,
)


# ============================================================
# 1. Unified forward model (surrogate wrapper)
# ============================================================

class ThreeSourceForwardModel:
    """
    Unified surrogate‑based forward model for 3 sources.

    This wrapper provides:
        - Surrogate evaluation for Esurf/Hsurf
        - Automatic superposition of fields from 3 sources
        - Feature extraction into a 4 * num_angles vector

    Parameters
    ----------
    path_E : str
        Path to trained surrogate model for Esurf.
    path_H : str
        Path to trained surrogate model for Hsurf.
    num_angles : int, optional
        Number of observation angles. Default is 30.

    Attributes
    ----------
    theta : ndarray of shape (num_angles,)
        Observation angles.
    R : float
        Physical radius of the domain.
    sur_wrap : SurrogateWrapper
        Unified surrogate forward model.
    """

    def __init__(self, path_E, path_H, num_angles=30):
        self.num_angles = num_angles
        self.theta = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

        # PhysicsTM only used to retrieve radius R
        phys = PhysicsTM()
        self.R = phys.R

        # Surrogate forward model
        sur = SurrogateEM(path_E=path_E, path_H=path_H)
        self.sur_wrap = SurrogateWrapper(sur)

    # --------------------------------------------------------

    def get_features(self, rho1, phi1, rho2, phi2, rho3, phi3):
        """
        Compute the feature vector for 3 sources.

        Parameters
        ----------
        rho1, rho2, rho3 : float
            Radial coordinates of the sources.
        phi1, phi2, phi3 : float
            Angular coordinates of the sources.

        Returns
        -------
        X : ndarray of shape (4 * num_angles,)
            Concatenated feature vector:
                [Re(E), Im(E), Re(H), Im(H)]
        """

        # Compute fields for each source
        E1 = self.sur_wrap.Esurf(rho1, phi1, self.theta)
        E2 = self.sur_wrap.Esurf(rho2, phi2, self.theta)
        E3 = self.sur_wrap.Esurf(rho3, phi3, self.theta)

        H1 = self.sur_wrap.Hsurf(rho1, phi1, self.theta)
        H2 = self.sur_wrap.Hsurf(rho2, phi2, self.theta)
        H3 = self.sur_wrap.Hsurf(rho3, phi3, self.theta)

        # Superposition
        E_total = E1 + E2 + E3
        H_total = H1 + H2 + H3

        # Split into real/imag
        Ere = np.real(E_total)
        Eim = np.imag(E_total)
        Hre = np.real(H_total)
        Him = np.imag(H_total)

        return np.concatenate([Ere, Eim, Hre, Him], axis=0).astype(np.float64)


# ============================================================
# 2. Dataset generation per stage
# ============================================================

def generate_dataset_for_stage(
    stage,
    num_samples,
    forward_model
):
    """
    Generate a dataset for a specific geometry stage.

    Parameters
    ----------
    stage : int
        Geometry stage index (1–8).
    num_samples : int
        Number of samples to generate.
    forward_model : ThreeSourceForwardModel
        Unified surrogate forward model.

    Returns
    -------
    X_raw : ndarray of shape (N, 4*num_angles)
        Raw feature vectors.
    y_raw : ndarray of shape (N, 6)
        Cartesian coordinates:
            [x1, y1, x2, y2, x3, y3]
    """

    X_list = []
    y_list = []

    pbar = tqdm(total=num_samples, desc=f"Stage {stage}: generating")

    while len(X_list) < num_samples:

        # Sample 3 sources with geometry constraints
        rho1, phi1, rho2, phi2, rho3, phi3 = sample_three_sources(stage)

        # Forward model
        feats = forward_model.get_features(
            rho1, phi1, rho2, phi2, rho3, phi3
        )

        # Targets in Cartesian coordinates
        y_vec = [
            rho1 * np.cos(phi1), rho1 * np.sin(phi1),
            rho2 * np.cos(phi2), rho2 * np.sin(phi2),
            rho3 * np.cos(phi3), rho3 * np.sin(phi3),
        ]

        X_list.append(feats)
        y_list.append(y_vec)

        pbar.update(1)

    pbar.close()

    return (
        np.array(X_list, dtype=np.float64),
        np.array(y_list, dtype=np.float64)
    )


# ============================================================
# 3. Scaling + saving utilities
# ============================================================

def save_stage_data(out_dir, stage, X_train, X_test, y_train, y_test, scaler_X, scaler_y):
    """
    Save dataset and scalers for a specific geometry stage.

    Parameters
    ----------
    out_dir : str
        Output directory.
    stage : int
        Geometry stage index.
    X_train, X_test : ndarray
        Scaled feature matrices.
    y_train, y_test : ndarray
        Scaled target matrices.
    scaler_X, scaler_y : MinMaxScaler
        Fitted scalers for X and y.
    """

    prefix = os.path.join(out_dir, f"stage_{stage}")

    np.save(prefix + "_X_train.npy", X_train)
    np.save(prefix + "_X_test.npy",  X_test)
    np.save(prefix + "_y_train.npy", y_train)
    np.save(prefix + "_y_test.npy",  y_test)

    with open(prefix + "_scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)

    with open(prefix + "_scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)

    print(f"[Stage {stage}] Saved dataset to {out_dir}/")


# ============================================================
# 4. Main entrypoint for dataset creation
# ============================================================

def create_3src_datasets(
    out_dir,
    path_E,
    path_H,
    data_E=None,
    data_H=None,
    stages=(1,2,3,4,5,6,7,8),
    num_samples_per_stage=70000,
    num_angles=30
):
    """
    Create datasets for all geometry stages of the 3‑source pipeline.

    Parameters
    ----------
    out_dir : str
        Output directory.
    path_E, path_H : str
        Paths to trained surrogate models.
    stages : iterable of int
        Geometry stages to generate.
    num_samples_per_stage : int
        Number of samples per stage.
    num_angles : int
        Number of observation angles.

    Notes
    -----
    - Each stage is saved independently.
    - MinMaxScaler is fitted on train only.
    - The forward model is reused across all stages.
    """

    os.makedirs(out_dir, exist_ok=True)

    # Unified forward model
    forward_model = ThreeSourceForwardModel(
        path_E=path_E,
        path_H=path_H,
        num_angles=num_angles
    )

    for stage in stages:

        print("\n====================================================")
        print(f"=== Generating dataset for Stage {stage} ===")
        print("====================================================")

        # Raw dataset
        X_raw, y_raw = generate_dataset_for_stage(
            stage=stage,
            num_samples=num_samples_per_stage,
            forward_model=forward_model
        )

        # Train/test split
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X_raw, y_raw, test_size=0.30, random_state=42
        )

        # Fit scalers on train only
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_train = scaler_X.fit_transform(X_train_raw)
        X_test  = scaler_X.transform(X_test_raw)

        y_train = scaler_y.fit_transform(y_train_raw)
        y_test  = scaler_y.transform(y_test_raw)

        # Save
        save_stage_data(
            out_dir,
            stage,
            X_train, X_test,
            y_train, y_test,
            scaler_X, scaler_y
        )

    print("\nAll stages completed.")
