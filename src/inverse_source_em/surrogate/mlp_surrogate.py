"""
Surrogate MLP model for approximating the TM-mode surface fields (Esurf, Hsurf).

This module defines a flexible, fully-connected neural network used as a
surrogate for the PhysicsTM solver. It supports configurable depth, width,
and activation functions, and is designed to work with normalized inputs:

    X = [rho_norm, cos(phi_s), sin(phi_s), cos(theta), sin(theta)]

The model outputs:
    Y = [Re(field), Im(field)]

Typical usage:
    from inverse_source_em.surrogate import SurrogateMLP

    model = SurrogateMLP(
        input_dim=5,
        output_dim=2,
        hidden_dim=128,
        num_layers=4,
        activation="relu"
    )
"""

import torch
import torch.nn as nn
from typing import Literal


class SurrogateMLP(nn.Module):
    """
    A configurable multilayer perceptron (MLP) surrogate model.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features (default: 5).
    output_dim : int
        Dimensionality of the output (default: 2).
    hidden_dim : int
        Width of each hidden layer (default: 128).
    num_layers : int
        Number of hidden layers (default: 4).
    activation : {"relu", "gelu", "tanh"}
        Activation function to use (default: "relu").

    Notes
    -----
    - The model is designed for normalized surrogate inputs.
    - The architecture is intentionally simple and robust.
    - Fourier features can be added later if needed.
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
        """Forward pass."""
        return self.net(x)
