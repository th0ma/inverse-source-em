import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


class ResidualBlock1D(nn.Module):
    """
    Basic 1D residual block:
        x -> Conv1d -> ReLU -> Dropout -> Conv1d -> +x -> ReLU
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


class SourceCountResNet1D(nn.Module):
    """
    ResNet-style 1D CNN for source-count classification (1–5 sources).

    Input:
        x: (B, 4, num_angles)
    Output:
        logits: (B, num_classes)
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
