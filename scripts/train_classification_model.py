"""
Train a 1–5 source-count classifier using the inverse_source_em package.

This script:
    - Loads classification_dataset.npz from data/classification/
    - Builds PyTorch DataLoaders for train/val/test
    - Instantiates the SourceCountResNet1D classifier
    - Trains with validation monitoring and early stopping
    - Saves the best checkpoint to:
          models/classifier_1_to_5_resnet1d.pth

Usage:
    python train_classification_model.py

Input features:
    X: (B, 4, num_angles)
       Channels = [Re(E), Im(E), Re(H), Im(H)]

Output:
    logits: (B, 5)   # classes S = 1..5
"""

import os
import torch

torch.set_default_dtype(torch.float64)

from inverse_source_em.training.train_classification import main


if __name__ == "__main__":
    main()
