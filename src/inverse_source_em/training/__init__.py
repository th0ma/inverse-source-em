"""
Training subpackage for the inverse_source_em project.

This package contains all training utilities used across the full
inverse EM pipeline, including:

1. Surrogate model training
   - load_surrogate_dataset
   - train_surrogate
   Provides dataset loading, DataLoaders, early stopping, LR scheduling,
   and checkpointing for the Esurf/Hsurf surrogate forward models.

2. Classification training (1–5 sources)
   - ResNet-style 1D CNN classifier
   - Cross-entropy training loop
   - Confusion matrix and evaluation utilities

3. Regression training
   a) Single-source (1src)
      - MultitaskNet with uncertainty weighting (Kendall & Gal)
      - Surrogate-based field consistency loss
      - Adam + ReduceLROnPlateau + early stopping

   b) Two-source (2src)
      - TwoSourcePredictor MLP
      - Structured loss (distance, angle, area constraints)
      - Dynamic tail-weight scheduling (p99-based)
      - Gradient clipping and full logging

   c) Three-source (3src)
      - ThreeSourceMultiHeadBig (ρ + cosφ/sinφ heads)
      - Curriculum learning across geometry stages 1..8
      - Noise-consistency regularization
      - Cosine annealing warm restarts
      - Stage-wise checkpointing

The training subpackage is intentionally modular:
each model family (surrogate, classification, 1src, 2src, 3src)
has its own training logic, loss functions, and data loaders,
while sharing a consistent design philosophy:
    - numerical stability (float64 where needed)
    - reproducibility
    - clean separation of model / loss / training loop
    - physics-aware regularization where appropriate

Only the surrogate training API is exported at the package level.
Other training modules are imported explicitly by users as needed.
"""

# ---------------------------------------------------------------------
# Public API (the only functions intended for direct import by users)
# ---------------------------------------------------------------------
from .train_surrogate import load_surrogate_dataset, train_surrogate


# ---------------------------------------------------------------------
# Internal modules
# ---------------------------------------------------------------------
from . import (
    classification_model,
    loss_2src,
    loss_3src,
    model_1src,
    model_2src,
    model_3src,
    train_1src,
    train_2src,
    train_3src,
    train_classification,
)


# ---------------------------------------------------------------------
# __all__ controls what pdoc3 shows as part of the package.
# ---------------------------------------------------------------------
__all__ = [
    # Public API
    "load_surrogate_dataset",
    "train_surrogate",

    # Internal modules (exposed for documentation)
    "classification_model",
    "loss_2src",
    "loss_3src",
    "model_1src",
    "model_2src",
    "model_3src",
    "train_1src",
    "train_2src",
    "train_3src",
    "train_classification",
]

