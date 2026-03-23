"""
Public API for inverse_source_em.data

Surrogate and classification pipelines are kept separate.
"""

# ------------------------------------------------------------
# Surrogate pipeline
# ------------------------------------------------------------
from .sampling_surrogate import sample_sources, sample_angles
from .generator_surrogate import SurrogateDataGenerator

# ------------------------------------------------------------
# Classification pipeline
# ------------------------------------------------------------
from .sampling_classification import (
    sample_single_source,
    sample_sources as sample_sources_classification,
)
from .generator_classification import forward_fields
from .dataset_classification import generate_classification_dataset

__all__ = [
    # Surrogate
    "sample_sources",
    "sample_angles",
    "SurrogateDataGenerator",

    # Classification
    "sample_single_source",
    "sample_sources_classification",
    "forward_fields",
    "generate_classification_dataset",
]
