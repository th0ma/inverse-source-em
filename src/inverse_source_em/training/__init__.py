"""
Training scripts for surrogate models and classifiers.
"""

from .train_surrogate import load_surrogate_dataset, train_surrogate

__all__ = ["load_surrogate_dataset", "train_surrogate"]
