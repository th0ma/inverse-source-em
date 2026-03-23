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
    Input:  5 features  [rho_norm, cos(phi), sin(phi), cos(theta), sin(theta)]
    Output: 2 features  [Re(field), Im(field)]
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
        return self.net(x)


# ------------------------------------------------------------
#  Surrogate EM (Unified API)
# ------------------------------------------------------------
class SurrogateEM:
    """
    Unified surrogate forward model for Esurf and Hsurf.

    Supports:
    - Esurf(rho, phi_s, theta_or_num_angles)
    - Hsurf(rho, phi_s, theta_or_num_angles)

    θ can be:
    - int: number of angles → uniform grid
    - float: single angle
    - array: explicit θ-array

    No normalization is applied (raw fields).
    """

    def __init__(self,
                 path_E="models/surrogate_Esurf.pth",
                 path_H="models/surrogate_Hsurf.pth",
                 R=1.0):
        """
        Parameters
        ----------
        path_E, path_H : str
            Paths to trained surrogate models.
        R : float
            Radius of the observation circle (for rho normalization).
        """
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

        rho, phi, theta must be broadcastable to same shape.
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
        X = self._make_input(rho, phi, theta)
        with torch.no_grad():
            Y = self.model_E(X).cpu().numpy()
        return Y[:, 0] + 1j * Y[:, 1]

    def batch_Hsurf(self, rho, phi, theta):
        X = self._make_input(rho, phi, theta)
        with torch.no_grad():
            Y = self.model_H(X).cpu().numpy()
        return Y[:, 0] + 1j * Y[:, 1]

    # --------------------------------------------------------
    #  Explicit θ API
    # --------------------------------------------------------
    def Esurf_theta(self, rho, phi_s, theta_array):
        theta = np.asarray(theta_array, dtype=float)
        rho_arr = np.broadcast_to(rho, theta.shape)
        phi_arr = np.broadcast_to(phi_s, theta.shape)
        return self.batch_Esurf(rho_arr, phi_arr, theta)

    def Hsurf_theta(self, rho, phi_s, theta_array):
        theta = np.asarray(theta_array, dtype=float)
        rho_arr = np.broadcast_to(rho, theta.shape)
        phi_arr = np.broadcast_to(phi_s, theta.shape)
        return self.batch_Hsurf(rho_arr, phi_arr, theta)

    # --------------------------------------------------------
    #  Unified API
    # --------------------------------------------------------
    def Esurf(self, rho, phi_s, theta_or_num_angles):
        if isinstance(theta_or_num_angles, (int, np.integer)):
            theta = np.linspace(0, 2*np.pi, int(theta_or_num_angles), endpoint=False)
            return self.Esurf_theta(rho, phi_s, theta)

        if np.isscalar(theta_or_num_angles):
            theta = np.array([theta_or_num_angles], dtype=float)
            return self.Esurf_theta(rho, phi_s, theta)[0]

        theta = np.asarray(theta_or_num_angles, dtype=float)
        return self.Esurf_theta(rho, phi_s, theta)

    def Hsurf(self, rho, phi_s, theta_or_num_angles):
        if isinstance(theta_or_num_angles, (int, np.integer)):
            theta = np.linspace(0, 2*np.pi, int(theta_or_num_angles), endpoint=False)
            return self.Hsurf_theta(rho, phi_s, theta)

        if np.isscalar(theta_or_num_angles):
            theta = np.array([theta_or_num_angles], dtype=float)
            return self.Hsurf_theta(rho, phi_s, theta)[0]

        theta = np.asarray(theta_or_num_angles, dtype=float)
        return self.Hsurf_theta(rho, phi_s, theta)
