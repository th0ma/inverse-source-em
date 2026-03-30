"""
model_2src.py

Neural network model for the two-source inverse EM regression task.

This module defines a clean, deep MLP architecture that maps
120-dimensional full-field features into 4 normalized Cartesian
coordinates:

    [x1/R, y1/R, x2/R, y2/R]

The model is intentionally simple and stable, following the successful
design pattern used in the 1-source and 3-source pipelines:

- Fully connected layers with ReLU activations
- Light dropout in early layers (empirically stabilizes training)
- No activation on the output layer (raw regression targets)

Typical usage
-------------
>>> from inverse_source_em.training.model_2src import build_model
>>> model = build_model(input_size=120,
...                     hidden_dims=[256, 256, 256, 256],
...                     output_size=4)
>>> preds = model(X_batch)
"""

import torch
from torch import nn


class TwoSourcePredictor(nn.Module):
    """
    Deep MLP for two-source localization regression.

    Parameters
    ----------
    input_size : int
        Dimensionality of the input feature vector.
        For 30 angles and 4 channels (Re(E), Im(E), Re(H), Im(H)):
            input_size = 4 * 30 = 120
    hidden_dims : list of int
        Widths of the hidden fully connected layers.
        Example: [256, 256, 256, 256]
    output_size : int
        Number of regression outputs.
        For two sources in normalized Cartesian coordinates:
            output_size = 4  →  [x1/R, y1/R, x2/R, y2/R]

    Notes
    -----
    - ReLU activations are used after each hidden layer.
    - Light dropout (0.05) is applied only to the first two layers.
    - The output layer is linear (no activation), suitable for regression.
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
        """
        Forward pass through the MLP.

        Parameters
        ----------
        x : torch.Tensor of shape (B, input_size)
            Input feature vectors.

        Returns
        -------
        torch.Tensor of shape (B, output_size)
            Predicted normalized Cartesian coordinates:
                [x1/R, y1/R, x2/R, y2/R]
        """
        return self.model(x)


def build_model(input_size, hidden_dims, output_size):
    """
    Factory function for clean instantiation of the TwoSourcePredictor.

    Parameters
    ----------
    input_size : int
        Dimensionality of the input feature vector.
    hidden_dims : list of int
        Hidden layer widths.
    output_size : int
        Number of regression outputs.

    Returns
    -------
    TwoSourcePredictor
        Instantiated model ready for training.
    """
    return TwoSourcePredictor(
        input_size=input_size,
        hidden_dims=hidden_dims,
        output_size=output_size
    )
