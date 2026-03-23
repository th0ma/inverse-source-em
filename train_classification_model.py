"""
Train classification model (1–5 sources) using the inverse_source_em package.

This script:
    - Loads dataset_classification.npz from data/classification/
    - Builds DataLoaders
    - Instantiates the ResNet1D classifier
    - Trains with validation monitoring
    - Saves the best model to models/classifier_1_to_5_resnet1d.pth

Usage:
    python train_classification_model.py
"""

import os
import torch

torch.set_default_dtype(torch.float64)

from inverse_source_em.training.train_classification import main


if __name__ == "__main__":
    main()
