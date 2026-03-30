"""
PhysicsTM-compatible wrapper around SurrogateEM.

This module provides a compatibility layer that makes a surrogate-based forward
model behave exactly like the analytical PhysicsTM solver. It exposes the same
public API:

    Esurf(rho, phi_s, theta_or_num_angles)
    Hsurf(rho, phi_s, theta_or_num_angles)

and supports:
- scalar θ
- array θ
- integer number of angles
- full NumPy broadcasting

Additionally, it provides a batch_forward(...) method used by inverse models
during training to generate full boundary fields (E_r, E_i, H_r, H_i) for a
batch of sources and observation angles.

This wrapper is typically used when training regression models that require
fast surrogate-based forward evaluations.
"""

import numpy as np
import torch


class SurrogateWrapper:
    """
    Wrapper that adapts a SurrogateEM instance to a PhysicsTM-like interface.

    This class ensures that the surrogate forward model exposes the same API
    and behavior as the analytical PhysicsTM solver. It handles broadcasting,
    angle preparation, and batch evaluation for inverse-model training.

    Parameters
    ----------
    surrogate_model : SurrogateEM
        A loaded surrogate forward model providing:
            batch_Esurf(rho, phi, theta)
            batch_Hsurf(rho, phi, theta)

    Notes
    -----
    - All returned fields are complex-valued.
    - Broadcasting rules match NumPy semantics.
    - The unified API supports int, scalar, and array θ inputs.
    """

    def __init__(self, surrogate_model):
        self.sur = surrogate_model

    # ------------------------------------------------------------
    # θ preparation
    # ------------------------------------------------------------
    def _prepare_theta(self, theta_or_num_angles):
        """
        Normalize θ input into an explicit NumPy array.

        Parameters
        ----------
        theta_or_num_angles : int, float, or array-like
            - int: number of angles → uniform grid in [0, 2π)
            - float: single angle
            - array-like: explicit angle array

        Returns
        -------
        ndarray of float
            Array of observation angles in radians.
        """
        if isinstance(theta_or_num_angles, (int, np.integer)):
            return np.linspace(0, 2*np.pi, int(theta_or_num_angles), endpoint=False)

        if np.isscalar(theta_or_num_angles):
            return np.array([theta_or_num_angles], dtype=float)

        return np.asarray(theta_or_num_angles, dtype=float)

    # ------------------------------------------------------------
    # Esurf
    # ------------------------------------------------------------
    def Esurf(self, rho, phi_s, theta_or_num_angles):
        """
        Compute surrogate Esurf with PhysicsTM-compatible behavior.

        Parameters
        ----------
        rho : float or array-like
            Radial coordinate(s) of the source.
        phi_s : float or array-like
            Angular coordinate(s) of the source.
        theta_or_num_angles : int, float, or array-like
            Number of angles, single angle, or explicit angle array.

        Returns
        -------
        complex or ndarray of complex
            Electric field values at the specified angles.
        """
        theta = self._prepare_theta(theta_or_num_angles)
        rho_arr = np.broadcast_to(rho, theta.shape)
        phi_arr = np.broadcast_to(phi_s, theta.shape)

        E = self.sur.batch_Esurf(rho_arr, phi_arr, theta)

        # Return scalar if input was scalar
        if np.isscalar(theta_or_num_angles) and not isinstance(theta_or_num_angles, (int, np.integer)):
            return E[0]

        return E

    # ------------------------------------------------------------
    # Hsurf
    # ------------------------------------------------------------
    def Hsurf(self, rho, phi_s, theta_or_num_angles):
        """
        Compute surrogate Hsurf with PhysicsTM-compatible behavior.

        Parameters
        ----------
        rho : float or array-like
            Radial coordinate(s) of the source.
        phi_s : float or array-like
            Angular coordinate(s) of the source.
        theta_or_num_angles : int, float, or array-like
            Number of angles, single angle, or explicit angle array.

        Returns
        -------
        complex or ndarray of complex
            Magnetic field values at the specified angles.
        """
        theta = self._prepare_theta(theta_or_num_angles)
        rho_arr = np.broadcast_to(rho, theta.shape)
        phi_arr = np.broadcast_to(phi_s, theta.shape)

        H = self.sur.batch_Hsurf(rho_arr, phi_arr, theta)

        # Return scalar if input was scalar
        if np.isscalar(theta_or_num_angles) and not isinstance(theta_or_num_angles, (int, np.integer)):
            return H[0]

        return H

    # ------------------------------------------------------------
    # Batch forward for inverse training
    # ------------------------------------------------------------
    def batch_forward(self, rho_t, phi_t, I_t, theta_obs):
        """
        Compute full boundary fields for a batch of sources and observation angles.

        This method is used by inverse regression models during training to
        generate the complete set of boundary fields (E_r, E_i, H_r, H_i) for
        each sample in the batch.

        Parameters
        ----------
        rho_t : torch.Tensor, shape (B,)
            Radial coordinates of the sources.
        phi_t : torch.Tensor, shape (B,)
            Angular coordinates of the sources.
        I_t : torch.Tensor, shape (B,)
            Source strengths.
        theta_obs : ndarray of shape (M,)
            Observation angles in radians.

        Returns
        -------
        E_r, E_i, H_r, H_i : torch.Tensor
            Real and imaginary parts of the electric and magnetic fields,
            each of shape (B, M).

        Notes
        -----
        - All computations are performed using NumPy, then converted back to torch.
        - Broadcasting is handled automatically.
        - This method is optimized for training loops.
        """
        rho = rho_t.detach().cpu().numpy()
        phi = phi_t.detach().cpu().numpy()
        I   = I_t.detach().cpu().numpy()

        batch = rho.shape[0]
        M = theta_obs.shape[0]

        # Broadcast to (B, M)
        rho_arr   = np.repeat(rho[:, None],   M, axis=1)
        phi_arr   = np.repeat(phi[:, None],   M, axis=1)
        I_arr     = np.repeat(I[:, None],     M, axis=1)
        theta_arr = np.repeat(theta_obs[None, :], batch, axis=0)

        # Flatten for batch inference
        rho_f   = rho_arr.reshape(-1)
        phi_f   = phi_arr.reshape(-1)
        I_f     = I_arr.reshape(-1)
        theta_f = theta_arr.reshape(-1)

        # Surrogate forward
        E = self.sur.batch_Esurf(rho_f, phi_f, theta_f) * I_f
        H = self.sur.batch_Hsurf(rho_f, phi_f, theta_f) * I_f

        # Reshape back to (B, M)
        E = E.reshape(batch, M)
        H = H.reshape(batch, M)

        device = rho_t.device
        E_r = torch.from_numpy(E.real).to(device)
        E_i = torch.from_numpy(E.imag).to(device)
        H_r = torch.from_numpy(H.real).to(device)
        H_i = torch.from_numpy(H.imag).to(device)

        return E_r, E_i, H_r, H_i
