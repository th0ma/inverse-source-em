"""
Dataset generator for the 3-source regression pipeline.

This module provides:
- Unified surrogate-based forward model for 3 sources
- Feature extraction (Re/Im of E and H)
- Dataset generation per geometry stage
- Scaling and saving utilities

The generator follows the same structure as the 1src and 2src pipelines.
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
    Wrapper around SurrogateWrapper for computing
    E/H surface fields for 3 sources.
    """
    def __init__(self, path_E, path_H, num_angles=30):
        self.num_angles = num_angles
        self.theta = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

        # PhysicsTM only used to retrieve radius R
        phys = PhysicsTM()
        self.R = phys.R

        # Surrogate forward model
        sur = SurrogateEM(
            path_E=path_E,
            path_H=path_H
        )

        self.sur_wrap = SurrogateWrapper(sur)

    # --------------------------------------------------------

    def get_features(self, rho1, phi1, rho2, phi2, rho3, phi3):
        """
        Compute feature vector:
            [Re(E), Im(E), Re(H), Im(H)]
        length = 4 * num_angles
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

    Returns:
        X_raw: (N, 4*num_angles)
        y_raw: (N, 6)
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
    Create datasets for all geometry stages of the 3-source pipeline.
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
