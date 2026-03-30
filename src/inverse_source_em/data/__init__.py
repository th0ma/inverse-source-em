"""
Public API for `inverse_source_em.data`.

This subpackage contains all dataset-generation pipelines used in the
inverse EM project. The pipelines are grouped into five domains:

1. Surrogate dataset generation
   - Sampling utilities for surrogate training
   - SurrogateDataGenerator for Esurf/Hsurf datasets

2. Classification dataset generation
   - Sampling utilities for multi-source classification
   - Unified surrogate-based forward model
   - Full dataset builder for the 1–5 source count task

3. Regression Problem I (single-source localization)
   - Sampling utilities for 1-source geometry
   - Surrogate-based forward model
   - Dataset builder + PyTorch Dataset class

4. Regression Problem II (two-source localization)
   - Sampling utilities with geometric priors
   - Feature construction using surrogate forward model
   - Dataset builder with MinMax scaling

5. Regression Problem III (three-source localization)
   - Geometry curriculum levels
   - Sampling with stage-based constraints
   - Unified forward model
   - Stage-based dataset generation and scaling

Only the high-level public entrypoints are exported here.
Internal helpers remain inside their respective modules.
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

# ------------------------------------------------------------
# Regression 1-source
# ------------------------------------------------------------
from .dataset_1src import Regression1SrcDataset
from .generator_1src import (
    build_dataset_1src,
    save_dataset_1src,
)
from .sampling_1src import (
    sample_sources_1src,
    sample_angles as sample_angles_1src,
)

# ------------------------------------------------------------
# Regression 2-source
# ------------------------------------------------------------
from .dataset_2src import build_dataset_2src
from .generator_2src import (
    load_surrogate as load_surrogate_2src,
    generate_sample as generate_sample_2src,
)
from .sampling_2src import (
    sample_two_sources,
    polar_to_cart_normalized,
)

# ------------------------------------------------------------
# Regression 3-source
# ------------------------------------------------------------
from .dataset_3src import (
    ThreeSourceDataset,
    load_3src_datasets,
    inverse_transform_targets,
)
from .generator_3src import (
    ThreeSourceForwardModel,
    create_3src_datasets,
)
from .sampling_3src import (
    GEOMETRY_LEVELS,
    sample_three_sources,
)

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

    # Regression 1-source
    "Regression1SrcDataset",
    "build_dataset_1src",
    "save_dataset_1src",
    "sample_sources_1src",
    "sample_angles_1src",

    # Regression 2-source
    "build_dataset_2src",
    "load_surrogate_2src",
    "generate_sample_2src",
    "sample_two_sources",
    "polar_to_cart_normalized",

    # Regression 3-source
    "ThreeSourceDataset",
    "load_3src_datasets",
    "inverse_transform_targets",
    "ThreeSourceForwardModel",
    "create_3src_datasets",
    "GEOMETRY_LEVELS",
    "sample_three_sources",
]
