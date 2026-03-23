import numpy as np
import torch


class SurrogateWrapper:
    """
    PhysicsTM-compatible wrapper around SurrogateEM.

    Provides:
    - Esurf(rho, phi_s, theta_or_num_angles)
    - Hsurf(rho, phi_s, theta_or_num_angles)

    Supports:
    - scalar θ
    - array θ
    - integer num_angles
    - full NumPy broadcasting

    Also provides:
    - batch_forward(...) for training inverse models
    """

    def __init__(self, surrogate_model):
        self.sur = surrogate_model

    # ------------------------------------------------------------
    # θ preparation
    # ------------------------------------------------------------
    def _prepare_theta(self, theta_or_num_angles):
        if isinstance(theta_or_num_angles, (int, np.integer)):
            return np.linspace(0, 2*np.pi, int(theta_or_num_angles), endpoint=False)

        if np.isscalar(theta_or_num_angles):
            return np.array([theta_or_num_angles], dtype=float)

        return np.asarray(theta_or_num_angles, dtype=float)

    # ------------------------------------------------------------
    # Esurf
    # ------------------------------------------------------------
    def Esurf(self, rho, phi_s, theta_or_num_angles):
        theta = self._prepare_theta(theta_or_num_angles)
        rho_arr = np.broadcast_to(rho, theta.shape)
        phi_arr = np.broadcast_to(phi_s, theta.shape)

        E = self.sur.batch_Esurf(rho_arr, phi_arr, theta)

        if np.isscalar(theta_or_num_angles) and not isinstance(theta_or_num_angles, (int, np.integer)):
            return E[0]

        return E

    # ------------------------------------------------------------
    # Hsurf
    # ------------------------------------------------------------
    def Hsurf(self, rho, phi_s, theta_or_num_angles):
        theta = self._prepare_theta(theta_or_num_angles)
        rho_arr = np.broadcast_to(rho, theta.shape)
        phi_arr = np.broadcast_to(phi_s, theta.shape)

        H = self.sur.batch_Hsurf(rho_arr, phi_arr, theta)

        if np.isscalar(theta_or_num_angles) and not isinstance(theta_or_num_angles, (int, np.integer)):
            return H[0]

        return H

    # ------------------------------------------------------------
    # Batch forward for inverse training
    # ------------------------------------------------------------
    def batch_forward(self, rho_t, phi_t, I_t, theta_obs):
        rho = rho_t.detach().cpu().numpy()
        phi = phi_t.detach().cpu().numpy()
        I   = I_t.detach().cpu().numpy()

        batch = rho.shape[0]
        M = theta_obs.shape[0]

        rho_arr   = np.repeat(rho[:, None],   M, axis=1)
        phi_arr   = np.repeat(phi[:, None],   M, axis=1)
        I_arr     = np.repeat(I[:, None],     M, axis=1)
        theta_arr = np.repeat(theta_obs[None, :], batch, axis=0)

        rho_f   = rho_arr.reshape(-1)
        phi_f   = phi_arr.reshape(-1)
        I_f     = I_arr.reshape(-1)
        theta_f = theta_arr.reshape(-1)

        E = self.sur.batch_Esurf(rho_f, phi_f, theta_f) * I_f
        H = self.sur.batch_Hsurf(rho_f, phi_f, theta_f) * I_f

        E = E.reshape(batch, M)
        H = H.reshape(batch, M)

        device = rho_t.device
        E_r = torch.from_numpy(E.real).to(device)
        E_i = torch.from_numpy(E.imag).to(device)
        H_r = torch.from_numpy(H.real).to(device)
        H_i = torch.from_numpy(H.imag).to(device)

        return E_r, E_i, H_r, H_i
