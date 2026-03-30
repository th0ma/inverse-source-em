"""
Inverse Source Problem – Electromagnetic Surrogates

This package provides a complete, modular framework for solving
electromagnetic inverse source problems using a combination of:

1. Physics‑based forward models
   - TM‑polarized analytical solutions
   - Canonical geometry and observation setup
   - Utilities for field synthesis and coordinate transforms

2. Neural surrogate models
   - Fast approximations of boundary fields (Esurf, Hsurf)
   - Unified SurrogateEM interface
   - SurrogateWrapper for batch inference and normalization

3. Machine‑learning pipelines for inverse regression
   - Single‑source localization with multitask uncertainty weighting
   - Two‑source localization with structured geometric losses
   - Three‑source localization with curriculum learning and
     multi‑head angular representation

4. Classification models
   - ResNet‑style 1D CNN for predicting the number of active sources

5. Training utilities
   - Surrogate training
   - Classification training
   - Regression pipelines (1src, 2src, 3src)
   - Curriculum training, early stopping, LR scheduling, checkpointing

The package is designed for research‑grade reproducibility:
consistent dtype handling (float64 where required), clean separation
between data, models, losses, and training loops, and physics‑aware
regularization throughout.

Subpackages
-----------
- physics: analytical EM forward models
- surrogate: neural surrogate architectures + wrappers
- data: dataset loaders for all inverse problems
- training: training loops, losses, and utilities
- utils: shared helpers (normalization, geometry, plotting)

This file exposes only the high‑level package identity.
All functionality is accessed through the respective submodules.
"""
