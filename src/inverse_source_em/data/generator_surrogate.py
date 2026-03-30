"""
Surrogate dataset generator using the analytical PhysicsTM solver.

This module generates training datasets for the surrogate MLP models that
approximate the TM-mode boundary fields (Esurf, Hsurf). The generator uses
the analytical PhysicsTM solver to compute ground-truth fields for a large
number of randomly sampled source locations and observation angles.

The generated dataset is stored in NPZ format with two arrays:

    X : (N*M, 5)
        Normalized surrogate input features:
            [rho_norm, cos(phi_s), sin(phi_s), cos(theta), sin(theta)]

    Y : (N*M, 2)
        Real and imaginary parts of the boundary field:
            [Re(field), Im(field)]

This dataset is used to train the surrogate MLPs that replace the analytical
solver in large-scale experiments and inverse-model training.
"""

import os
import numpy as np
from tqdm import tqdm

from inverse_source_em.physics.physics_tm import PhysicsTM
from inverse_source_em.data.sampling_surrogate import (
    sample_sources,
    sample_angles,
)


class SurrogateDataGenerator:
    """
    Generator for surrogate training datasets (Esurf, Hsurf).

    This class samples random source locations, evaluates the analytical
    PhysicsTM solver at a fixed set of observation angles, and constructs
    the normalized surrogate input features (X) and target outputs (Y).

    Parameters
    ----------
    physics : PhysicsTM
        Analytical forward model used to compute ground-truth fields.
    output_dir : str, optional
        Directory where NPZ files will be saved. Default is "data/surrogate".
    num_angles : int, optional
        Number of observation angles per source. Default is 72.
    rho_min : float, optional
        Minimum radial coordinate for sampling sources (relative to R).
    rho_max : float, optional
        Maximum radial coordinate for sampling sources (relative to R).

    Notes
    -----
    - All computations are performed in float64 for numerical stability.
    - The sampling functions are defined in `sampling_surrogate.py`.
    - The generated dataset is fully compatible with SurrogateDataset.
    """

    def __init__(self,
                 physics: PhysicsTM,
                 output_dir: str = "data/surrogate",
                 num_angles: int = 72,
                 rho_min: float = 0.05,
                 rho_max: float = 0.95):

        self.phys = physics
        self.R = physics.R

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.num_angles = num_angles
        self.rho_min = rho_min
        self.rho_max = rho_max

        # Precompute observation angles
        self.thetas = sample_angles(num_angles)

    # -------------------------------------------------------------
    def generate(self, kind: str, n_sources: int):
        """
        Generate surrogate training data for Esurf or Hsurf.

        Parameters
        ----------
        kind : {"Esurf", "Hsurf"}
            Field type to generate.
        n_sources : int
            Number of randomly sampled sources.

        Returns
        -------
        X : ndarray of shape (n_sources * M, 5)
            Normalized surrogate input features.
        Y : ndarray of shape (n_sources * M, 2)
            Real and imaginary parts of the boundary field.

        Notes
        -----
        - M = num_angles
        - Sampling is uniform in (rho, phi) within the specified bounds.
        - The PhysicsTM solver is used to compute ground-truth fields.
        """
        assert kind in ("Esurf", "Hsurf")
        M = self.num_angles

        # Sample all sources
        rho_s_all, phi_s_all = sample_sources(
            n_sources,
            R=self.R,
            rho_min=self.rho_min,
            rho_max=self.rho_max
        )

        print(f"[Generator] {kind}: {n_sources} sources × {M} angles")

        # Allocate arrays (5 features)
        X = np.zeros((n_sources * M, 5), dtype=np.float64)
        Y = np.zeros((n_sources * M, 2), dtype=np.float64)

        # Fill X
        for i in range(n_sources):
            start = i * M
            end = start + M

            rho_norm = rho_s_all[i] / self.R

            X[start:end, 0] = rho_norm
            X[start:end, 1] = np.cos(phi_s_all[i])
            X[start:end, 2] = np.sin(phi_s_all[i])
            X[start:end, 3] = np.cos(self.thetas)
            X[start:end, 4] = np.sin(self.thetas)

        # Compute Y
        for i in tqdm(range(n_sources)):
            rho_s = rho_s_all[i]
            phi_s = phi_s_all[i]

            if kind == "Esurf":
                F = self.phys.Esurf(rho_s, phi_s, M)
            else:
                F = self.phys.Hsurf(rho_s, phi_s, M)

            F = F.astype(np.complex128)

            start = i * M
            end = start + M

            Y[start:end, 0] = np.real(F)
            Y[start:end, 1] = np.imag(F)

        return X, Y

    # -------------------------------------------------------------
    def save_npz(self, X, Y, filename: str):
        """
        Save surrogate dataset to a compressed NPZ file.

        Parameters
        ----------
        X : ndarray
            Input feature matrix of shape (N, 5).
        Y : ndarray
            Target output matrix of shape (N, 2).
        filename : str
            Name of the output NPZ file.

        Notes
        -----
        - The file is saved under `output_dir`.
        - Uses NumPy's compressed NPZ format.
        """
        path = os.path.join(self.output_dir, filename)
        np.savez_compressed(path, X=X, Y=Y)
        print(f"[Saved] {path}")
