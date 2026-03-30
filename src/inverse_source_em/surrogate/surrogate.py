"""
Surrogate EM forward model using trained MLP approximators for Esurf and Hsurf.

This module provides a unified surrogate-based forward solver that mimics the
analytical PhysicsTM model. It loads two trained MLP surrogate networks:
one for the electric field (Esurf) and one for the magnetic field (Hsurf).

The surrogate operates on normalized geometric features:

    X = [rho_norm, cos(phi), sin(phi), cos(theta), sin(theta)]

and outputs the real and imaginary parts of the boundary field:

    Y = [Re(field), Im(field)]

The unified API mirrors the analytical solver:

    Esurf(rho, phi_s, theta_or_num_angles)
    Hsurf(rho, phi_s, theta_or_num_angles)

where the third argument may be:
- int: number of observation angles → uniform grid in [0, 2π)
- float: single angle
- array-like: explicit array of angles

This surrogate is used for:
- fast forward evaluations
- dataset generation for regression pipelines
- replacing the analytical solver in large-scale experiments
"""

import numpy as np
import torch
import torch.nn as nn

# Use float64 everywhere for scientific consistency
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
#  MLP definition
# ------------------------------------------------------------
class SurrogateMLP(nn.Module):
    """
    Fully-connected MLP for surrogate modeling.

    This network approximates the TM-mode boundary fields produced by the
    analytical PhysicsTM solver. It maps normalized geometric features and
    observation angles to the real and imaginary parts of the surface field.

    Input features:
        [rho_norm, cos(phi), sin(phi), cos(theta), sin(theta)]

    Output features:
        [Re(field), Im(field)]

    Parameters
    ----------
    input_dim : int, optional
        Dimensionality of the input feature vector. Default is 5.
    output_dim : int, optional
        Dimensionality of the output vector. Default is 2.
    hidden_dim : int, optional
        Width of each hidden layer. Default is 128.
    num_layers : int, optional
        Number of hidden layers. Default is 4.

    Notes
    -----
    - The model is fully differentiable and suitable for MSE training.
    - Designed for use inside the surrogate forward pipeline.
    - Uses ReLU activations in all hidden layers.
    """

    def __init__(self, input_dim=5, output_dim=2, hidden_dim=128, num_layers=4):
        super().__init__()
        layers = []
        dim_in = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim_in, hidden_dim))
            layers.append(nn.ReLU())
            dim_in = hidden_dim
        layers.append(nn.Linear(dim_in, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the surrogate MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., output_dim), containing:
                [Re(field), Im(field)]
        """
        return self.net(x)


# ------------------------------------------------------------
#  Surrogate EM (Unified API)
# ------------------------------------------------------------
class SurrogateEM:
    """
    Unified surrogate forward model for Esurf and Hsurf.

    This class loads two trained surrogate MLPs (one for Esurf, one for Hsurf)
    and exposes a unified API identical to the analytical PhysicsTM solver.

    Supported calls:
        Esurf(rho, phi_s, theta_or_num_angles)
        Hsurf(rho, phi_s, theta_or_num_angles)

    The argument `theta_or_num_angles` may be:
    - int: number of angles → uniform grid in [0, 2π)
    - float: single angle
    - array-like: explicit array of angles

    No normalization is applied to the output fields (raw complex values).

    Parameters
    ----------
    path_E : str
        Path to the trained surrogate model for Esurf.
    path_H : str
        Path to the trained surrogate model for Hsurf.
    R : float, optional
        Radius of the observation circle. Used for rho normalization.
    """

    def __init__(self,
                 path_E="models/surrogate_Esurf.pth",
                 path_H="models/surrogate_Hsurf.pth",
                 R=1.0):

        self.R = R

        # Load models
        self.model_E = SurrogateMLP().to(device)
        self.model_H = SurrogateMLP().to(device)

        self.model_E.load_state_dict(torch.load(path_E, map_location=device))
        self.model_H.load_state_dict(torch.load(path_H, map_location=device))

        self.model_E.eval()
        self.model_H.eval()

    # --------------------------------------------------------
    #  Input feature builder
    # --------------------------------------------------------
    def _make_input(self, rho, phi, theta):
        """
        Build normalized input features for the surrogate MLP.

        Parameters
        ----------
        rho : array-like
            Radial coordinate(s) of the source.
        phi : array-like
            Angular coordinate(s) of the source.
        theta : array-like
            Observation angle(s).

        Returns
        -------
        torch.Tensor
            Tensor of shape (N, 5) containing:
                [rho_norm, cos(phi), sin(phi), cos(theta), sin(theta)]
        """
        rho = np.asarray(rho)
        phi = np.asarray(phi)
        theta = np.asarray(theta)

        rho_norm = rho / self.R

        X = np.column_stack([
            rho_norm,
            np.cos(phi), np.sin(phi),
            np.cos(theta), np.sin(theta)
        ])

        return torch.from_numpy(X).to(device)

    # --------------------------------------------------------
    #  Low-level batch inference
    # --------------------------------------------------------
    def batch_Esurf(self, rho, phi, theta):
        """
        Compute Esurf for a batch of (rho, phi, theta) inputs.

        Returns
        -------
        ndarray of complex128
            Complex electric field values.
        """
        X = self._make_input(rho, phi, theta)
        with torch.no_grad():
            Y = self.model_E(X).cpu().numpy()
        return Y[:, 0] + 1j * Y[:, 1]

    def batch_Hsurf(self, rho, phi, theta):
        """
        Compute Hsurf for a batch of (rho, phi, theta) inputs.

        Returns
        -------
        ndarray of complex128
            Complex magnetic field values.
        """
        X = self._make_input(rho, phi, theta)
        with torch.no_grad():
            Y = self.model_H(X).cpu().numpy()
        return Y[:, 0] + 1j * Y[:, 1]

    # --------------------------------------------------------
    #  Explicit θ API
    # --------------------------------------------------------
    def Esurf_theta(self, rho, phi_s, theta_array):
        """
        Compute Esurf for an explicit array of observation angles.

        Parameters
        ----------
        rho : float
            Radial coordinate of the source.
        phi_s : float
            Angular coordinate of the source.
        theta_array : array-like
            Observation angles in radians.

        Returns
        -------
        ndarray of complex128
            Electric field values at the specified angles.
        """
        theta = np.asarray(theta_array, dtype=float)
        rho_arr = np.broadcast_to(rho, theta.shape)
        phi_arr = np.broadcast_to(phi_s, theta.shape)
        return self.batch_Esurf(rho_arr, phi_arr, theta)

    def Hsurf_theta(self, rho, phi_s, theta_array):
        """
        Compute Hsurf for an explicit array of observation angles.

        Parameters
        ----------
        rho : float
            Radial coordinate of the source.
        phi_s : float
            Angular coordinate of the source.
        theta_array : array-like
            Observation angles in radians.

        Returns
        -------
        ndarray of complex128
            Magnetic field values at the specified angles.
        """
        theta = np.asarray(theta_array, dtype=float)
        rho_arr = np.broadcast_to(rho, theta.shape)
        phi_arr = np.broadcast_to(phi_s, theta.shape)
        return self.batch_Hsurf(rho_arr, phi_arr, theta)

    # --------------------------------------------------------
    #  Unified API
    # --------------------------------------------------------
    def Esurf(self, rho, phi_s, theta_or_num_angles):
        """
        Unified API for Esurf.

        Parameters
        ----------
        rho : float
            Radial coordinate of the source.
        phi_s : float
            Angular coordinate of the source.
        theta_or_num_angles : int, float, or array-like
            Number of angles, single angle, or explicit angle array.

        Returns
        -------
        complex or ndarray of complex
            Electric field values.
        """
        if isinstance(theta_or_num_angles, (int, np.integer)):
            theta = np.linspace(0, 2*np.pi, int(theta_or_num_angles), endpoint=False)
            return self.Esurf_theta(rho, phi_s, theta)

        if np.isscalar(theta_or_num_angles):
            theta = np.array([theta_or_num_angles], dtype=float)
            return self.Esurf_theta(rho, phi_s, theta)[0]

        theta = np.asarray(theta_or_num_angles, dtype=float)
        return self.Esurf_theta(rho, phi_s, theta)

    def Hsurf(self, rho, phi_s, theta_or_num_angles):
        """
        Unified API for Hsurf.

        Parameters
        ----------
        rho : float
            Radial coordinate of the source.
        phi_s : float
            Angular coordinate of the source.
        theta_or_num_angles : int, float, or array-like
            Number of angles, single angle, or explicit angle array.

        Returns
        -------
        complex or ndarray of complex
            Magnetic field values.
        """
        if isinstance(theta_or_num_angles, (int, np.integer)):
            theta = np.linspace(0, 2*np.pi, int(theta_or_num_angles), endpoint=False)
            return self.Hsurf_theta(rho, phi_s, theta)

        if np.isscalar(theta_or_num_angles):
            theta = np.array([theta_or_num_angles], dtype=float)
            return self.Hsurf_theta(rho, phi_s, theta)[0]

        theta = np.asarray(theta_or_num_angles, dtype=float)
        return self.Hsurf_theta(rho, phi_s, theta)
