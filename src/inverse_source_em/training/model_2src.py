"""
model_2src.py

Neural network model for the two-source inverse EM regression task.
Defines a clean MLP architecture that maps 120-dimensional full-field
features to 4 normalized Cartesian coordinates:
    [x1/R, y1/R, x2/R, y2/R]
"""

import torch
from torch import nn


class TwoSourcePredictor(nn.Module):
    """
    Simple, deep MLP for two-source localization.
    """

    def __init__(self, input_size, hidden_dims, output_size):
        super().__init__()

        layers = []
        prev = input_size

        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())

            # Light dropout on early layers (empirically stabilizes training)
            if i < 2:
                layers.append(nn.Dropout(0.05))

            prev = h

        layers.append(nn.Linear(prev, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def build_model(input_size, hidden_dims, output_size):
    """
    Factory function for clean instantiation.
    """
    return TwoSourcePredictor(
        input_size=input_size,
        hidden_dims=hidden_dims,
        output_size=output_size
    )
