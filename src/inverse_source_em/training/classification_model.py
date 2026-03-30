"""
Neural network architectures for the source-count classification task.

This module implements a lightweight 1D ResNet-style CNN designed for
classifying the number of active sources (1–5) from boundary field
measurements. The input consists of:

    x : (B, 4, num_angles)
        Channels correspond to:
            [Re(E), Im(E), Re(H), Im(H)]

The network uses:
- An initial Conv1d stem
- A stack of residual 1D blocks
- Global average pooling
- A final linear classifier

All computations use float64 for consistency with the surrogate models.
"""

import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


# ============================================================
# 1. Residual block
# ============================================================

class ResidualBlock1D(nn.Module):
    """
    Basic 1D residual block.

    Architecture:
        x → Conv1d → ReLU → Dropout → Conv1d → +x → ReLU

    Parameters
    ----------
    channels : int
        Number of input/output channels.
    kernel_size : int, optional
        Convolution kernel size. Default is 3.
    padding : int, optional
        Padding for Conv1d. Default is 1 (keeps length constant).
    dropout : float, optional
        Dropout probability.

    Notes
    -----
    - Input and output shapes are identical.
    - Residual connection improves gradient flow and stability.
    """

    def __init__(self, channels, kernel_size=3, padding=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.act(self.conv1(x))
        out = self.dropout(out)
        out = self.conv2(out)
        out = out + residual
        out = self.act(out)
        return out


# ============================================================
# 2. ResNet-style classifier
# ============================================================

class SourceCountResNet1D(nn.Module):
    """
    ResNet-style 1D CNN for source-count classification (1–5 sources).

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels. Default is 4.
    num_angles : int, optional
        Number of observation angles. Default is 30.
    num_classes : int, optional
        Number of output classes. Default is 5.
    base_channels : int, optional
        Number of channels in the first convolution. Default is 64.
    num_blocks : int, optional
        Number of residual blocks. Default is 4.
    dropout : float, optional
        Dropout probability inside residual blocks.

    Input
    -----
    x : torch.Tensor of shape (B, 4, num_angles)

    Output
    ------
    logits : torch.Tensor of shape (B, num_classes)
        Raw class logits (no softmax applied).

    Notes
    -----
    - AdaptiveAvgPool1d(1) makes the model independent of num_angles.
    - The model is intentionally lightweight and stable.
    """

    def __init__(self, in_channels=4, num_angles=30, num_classes=5,
                 base_channels=64, num_blocks=4, dropout=0.1):
        super().__init__()

        self.conv_in = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()

        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                ResidualBlock1D(
                    base_channels,
                    kernel_size=3,
                    padding=1,
                    dropout=dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels, num_classes)

    def forward(self, x):
        x = self.act(self.conv_in(x))   # (B, C, L)
        x = self.blocks(x)              # (B, C, L)
        x = self.pool(x)                # (B, C, 1)
        x = x.squeeze(-1)               # (B, C)
        logits = self.fc(x)             # (B, num_classes)
        return logits
