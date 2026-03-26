"""
Classification Evaluation Package

This package provides modular evaluation components for the
source-count classification model.

Modules included:
- accuracy: clean test accuracy and per-class metrics
- confusion: confusion matrix and class-wise precision/recall/F1
- noise_robustness: accuracy degradation under additive noise
- timing: inference speed benchmarking
"""

from .accuracy import evaluate as evaluate_accuracy
from .confusion import evaluate as evaluate_confusion
from .noise_robustness import evaluate as evaluate_noise
from .timing import evaluate as evaluate_timing

__all__ = [
    "evaluate_accuracy",
    "evaluate_confusion",
    "evaluate_noise",
    "evaluate_timing",
]
