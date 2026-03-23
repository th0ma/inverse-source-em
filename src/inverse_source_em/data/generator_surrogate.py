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
    Generates surrogate datasets (Esurf, Hsurf) using the PhysicsTM solver.
    Saves datasets in NPZ format:

        X: (N*M, 5)
            [rho_norm, cos(phi_s), sin(phi_s), cos(theta), sin(theta)]

        Y: (N*M, 2)
            [Re(field), Im(field)]
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
        kind: 'Esurf' or 'Hsurf'
        Returns X, Y arrays.
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
        path = os.path.join(self.output_dir, filename)
        np.savez_compressed(path, X=X, Y=Y)
        print(f"[Saved] {path}")
