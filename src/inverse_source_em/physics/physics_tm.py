"""
PhysicsTM: Analytical TM forward model for a dielectric cylinder
with one internal line source. Fully vectorized (NumPy + SciPy).

Unified forward API:
- Esurf(rho_s, phi_s, theta_or_num_angles)
- Hsurf(rho_s, phi_s, theta_or_num_angles)

The third argument can be:
- int: number of angles → uniform θ-grid in [0, 2π)
- scalar: single observation angle θ → returns scalar
- array: explicit θ-array → returns array
"""

import numpy as np
import math
from scipy.special import jv, jvp, hankel1, h1vp


class PhysicsTM:
    """
    Analytical TM forward solver for cylindrical scattering.
    """

    def __init__(self,
                 R=1.0,
                 OMEGA=4.0,
                 EPS0=1.0,
                 MU0=1.0,
                 EPS1=1.3225,
                 MU1=1.0,
                 N=20,
                 I0=1.0):

        self.R     = float(R)
        self.OMEGA = float(OMEGA)
        self.EPS0  = float(EPS0)
        self.MU0   = float(MU0)
        self.EPS1  = float(EPS1)
        self.MU1   = float(MU1)
        self.N     = int(N)
        self.I0    = float(I0)

        self.K0   = self.OMEGA * math.sqrt(self.EPS0 * self.MU0)
        self.ETA  = math.sqrt(self.EPS1 * self.MU1)
        self.K1   = self.K0 * self.ETA
        self.A    = -self.OMEGA * self.MU0 * self.MU1 / 4.0

        self.nvals = np.arange(-self.N, self.N + 1, dtype=np.int64)

    def make_obs(self, num_angles):
        return np.linspace(0.0, 2*np.pi, int(num_angles), endpoint=False)

    def alpha_n_vec(self, rho_s):
        rho_s = float(rho_s)
        n = self.nvals
        R = self.R

        num = -2j * jv(n, self.K1 * rho_s)

        Jn_prime_K1R = jvp(n, self.K1 * R, 1)
        Hn_K0R       = hankel1(n, self.K0 * R)
        Hn_prime_K0R = h1vp(n, self.K0 * R, 1)
        Jn_K1R       = jv(n, self.K1 * R)

        den_inner = self.ETA * Jn_prime_K1R * Hn_K0R - self.MU1 * Hn_prime_K0R * Jn_K1R
        den = self.K0 * R * np.pi * den_inner

        return num / den

    def Esurf_theta(self, rho_s, phi_s, theta_array):
        rho_s = float(rho_s)
        phi_s = float(phi_s)

        obs = np.asarray(theta_array, dtype=float)
        n = self.nvals

        coeffs = self.alpha_n_vec(rho_s)
        H = hankel1(n, self.K0 * self.R).astype(np.complex128)

        phase = np.exp(1j * n[:, None] * (obs[None, :] - phi_s))
        field = np.sum(coeffs[:, None] * H[:, None] * phase, axis=0)

        return np.complex128(self.A * self.I0) * field

    def Hsurf_theta(self, rho_s, phi_s, theta_array):
        rho_s = float(rho_s)
        phi_s = float(phi_s)

        obs = np.asarray(theta_array, dtype=float)
        n = self.nvals

        coeffs = self.alpha_n_vec(rho_s)
        Hprime = h1vp(n, self.K0 * self.R, 1).astype(np.complex128)

        phase = np.exp(1j * n[:, None] * (obs[None, :] - phi_s))
        field = np.sum(coeffs[:, None] * Hprime[:, None] * phase, axis=0)

        return np.complex128(self.A * self.I0 / self.MU0) * field

    def Esurf(self, rho_s, phi_s, theta_or_num_angles=720):
        if isinstance(theta_or_num_angles, (int, np.integer)):
            obs = self.make_obs(int(theta_or_num_angles))
            return self.Esurf_theta(rho_s, phi_s, obs)

        if np.isscalar(theta_or_num_angles):
            obs = np.array([theta_or_num_angles], dtype=float)
            out = self.Esurf_theta(rho_s, phi_s, obs)
            return out[0]

        obs = np.asarray(theta_or_num_angles, dtype=float)
        return self.Esurf_theta(rho_s, phi_s, obs)

    def Hsurf(self, rho_s, phi_s, theta_or_num_angles=720):
        if isinstance(theta_or_num_angles, (int, np.integer)):
            obs = self.make_obs(int(theta_or_num_angles))
            return self.Hsurf_theta(rho_s, phi_s, obs)

        if np.isscalar(theta_or_num_angles):
            obs = np.array([theta_or_num_angles], dtype=float)
            out = self.Hsurf_theta(rho_s, phi_s, obs)
            return out[0]

        obs = np.asarray(theta_or_num_angles, dtype=float)
        return self.Hsurf_theta(rho_s, phi_s, obs)
