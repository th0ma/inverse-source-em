"""
Surrogate MLP model for approximating TM-mode surface fields (Esurf, Hsurf).

This module defines a flexible, fully-connected neural network used as a
surrogate for the analytical PhysicsTM solver. The surrogate approximates the
complex-valued boundary fields on the cylinder surface using normalized inputs:

    X = [rho_norm, cos(phi_s), sin(phi_s), cos(theta), sin(theta)]

The model outputs the real and imaginary parts of the field:

    Y = [Re(field), Im(field)]

The surrogate is used for:
- fast forward evaluations
- dataset generation for regression pipelines
- enforcing field-consistency losses during training
- replacing the analytical solver in large-scale experiments

Typical usage
-------------
>>> from inverse_source_em.surrogate import SurrogateMLP
>>> model = SurrogateMLP(input_dim=5, output_dim=2, hidden_dim=128, num_layers=4)
>>> y = model(x)

Notes
-----
- The architecture is intentionally simple and robust.
- Inputs must be normalized according to the surrogate training pipeline.
- Fourier features or positional encodings can be added externally if needed.
"""

import torch
import torch.nn as nn
from typing import Literal


class SurrogateMLP(nn.Module):
    """
    A configurable multilayer perceptron (MLP) surrogate model.

    This network approximates the TM-mode boundary fields produced by the
    analytical PhysicsTM solver. It maps normalized geometric features and
    observation angles to the real and imaginary parts of the surface field.

    Parameters
    ----------
    input_dim : int, optional
        Dimensionality of the input feature vector. Default is 5.
        Expected input format:
            [rho_norm, cos(phi_s), sin(phi_s), cos(theta), sin(theta)]
    output_dim : int, optional
        Dimensionality of the output vector. Default is 2.
        Output format:
            [Re(field), Im(field)]
    hidden_dim : int, optional
        Width of each hidden layer. Default is 128.
    num_layers : int, optional
        Number of hidden layers. Default is 4.
    activation : {"relu", "gelu", "tanh"}, optional
        Activation function used in all hidden layers. Default is "relu".

    Notes
    -----
    - The model is fully differentiable and suitable for training with MSE loss.
    - Designed for use inside the surrogate forward pipeline.
    - The architecture is intentionally minimal to ensure fast inference.
    """

    def __init__(
        self,
        input_dim: int = 5,
        output_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 4,
        activation: Literal["relu", "gelu", "tanh"] = "relu"
    ):
        super().__init__()

        # Select activation
        if activation == "relu":
            act = nn.ReLU
        elif activation == "gelu":
            act = nn.GELU
        elif activation == "tanh":
            act = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        dim_in = input_dim

        # Hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(dim_in, hidden_dim))
            layers.append(act())
            dim_in = hidden_dim

        # Output layer
        layers.append(nn.Linear(dim_in, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the surrogate MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., input_dim). Inputs must be normalized
            according to the surrogate preprocessing pipeline.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., output_dim), containing:
                [Re(field), Im(field)]
        """
        return self.net(x)
