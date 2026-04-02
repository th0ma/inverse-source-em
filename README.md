# Inverse Source EM

**Deep Learning Methods for Electromagnetic Inverse Source Problems**

Version: **0.1.0**

This repository provides a complete scientific machine learning framework for
electromagnetic inverse source problems in cylindrical geometries. It combines:

- Physics-based analytical solvers (Helmholtz equation, TM polarization)
- Neural surrogate models for fast forward simulations
- Deep learning pipelines for classification and localization of internal sources
- Modular evaluation suites for reproducible benchmarking
- A clean, pip-installable Python package (inverse_source_em)

The project has been fully migrated from notebooks into a modern, modular,
production-ready architecture.

---

## 1. Abstract

We study an inverse source problem inside a dielectric cylinder, belonging to
the broader class of inverse scattering problems. The goal is to recover the:

- number of internal line sources,
- positions (Cartesian or polar),
- strengths,

using only electric surface field measurements at a single frequency.

The forward problem is solved analytically using cylindrical harmonics and
Bessel/Hankel special functions. The PhysicsTM class provides a stable,
vectorized implementation of the TM-polarized Helmholtz equation.

On top of this, we build three machine learning pipelines:

1. Surrogate EM models for fast forward simulations  
2. Classification models for predicting the number of sources (1–5)  
3. Regression models for localizing 1, 2, or 3 internal sources  

The surrogate models enable large-scale dataset generation with high accuracy.
The regression pipelines incorporate geometric sampling, canonical ordering,
structured losses, and permutation-invariant evaluation.

The results demonstrate high accuracy, strong noise robustness, and reliable
generalization even in challenging geometries.

---

## 2. Project Structure

    inverse-source-em/
    │
    ├── data/
    │   ├── classification/
    │   ├── regression_1src/
    │   ├── regression_2src/
    │   ├── regression_3src/
    │   └── surrogate/
    │
    ├── models/
    │   ├── surrogate_Esurf.pth
    │   ├── surrogate_Hsurf.pth
    │   ├── classifier_1_to_5_resnet1d.pt
    │   ├── regression_1src/
    │   ├── regression_2src/
    │   └── regression_3src/
    │
    ├── scripts/
    │   ├── make_1src_dataset.py
    │   ├── make_2src_dataset.py
    │   ├── make_3src_dataset.py
    │   ├── make_classification_dataset.py
    │   ├── make_surrogate_dataset.py
    │   ├── train_1src_model.py
    │   ├── train_2src_model.py
    │   ├── train_3src_model.py
    │   ├── train_classification_model.py
    │   └── train_surrogate_models.py
    │
    ├── evaluation/
    │   ├── classification/
    │   ├── regression_1src/
    │   ├── regression_2src/
    │   ├── regression_3src/
    │   └── surrogates/
    │
    ├── notebooks/
    │   ├── regression_1src_evaluation.ipynb
    │   ├── regression_2src_evaluation.ipynb
    │   ├── regression_3src_evaluation.ipynb
    │   ├── classification_evaluation.ipynb
    │   └── physics_vs_surrogate.ipynb
    │
    ├── src/inverse_source_em/
    │   ├── surrogate/
    │   ├── data/
    │   ├── physics/
    │   ├── training/
    │   ├── evaluation/
    │   └── utils/
    │
    ├── pyproject.toml
    ├── requirements.txt
    └── README.md

---

## 3. Surrogate Pipeline (Canonical & Frozen)

The surrogate pipeline is complete and frozen. It provides:

- SurrogateEM — unified EM surrogate model  
- SurrogateMLP — configurable MLP surrogate  
- SurrogateWrapper — PhysicsTM-compatible interface  
- Sampling and dataset generation utilities  

These models form the reference baseline for all inverse tasks.

---

## 4. Classification Pipeline (Completed)

Predicts the number of internal sources (1–5) from surface field intensities.

Includes:

- Sampling (sampling_classification.py)
- Dataset generation (generator_classification.py)
- PyTorch dataset wrapper (dataset_classification.py)
- ResNet1D classifier (classification_model.py)
- Training script (train_classification_model.py)
- Evaluation suite (evaluation/classification/)

Baseline performance:

- Validation accuracy: ~0.984  
- Test accuracy: ~0.985  
- Strong robustness to additive noise  

---

## 5. Regression Pipelines

### 5.1 Single-Source Localization (1src)

- Predicts (x, y, I)  
- Canonical sampling  
- High accuracy and noise robustness  
- Full evaluation suite

### 5.2 Two-Source Localization (2src)

- Predicts (x1, y1), (x2, y2)  
- Fully permutation-invariant  
- Strong performance across all geometry levels  
- Full evaluation suite

### 5.3 Three-Source Localization (3src) — New in v0.9.0-beta

- Predicts (x1, y1), (x2, y2), (x3, y3)  
- Canonical ordering  
- Geometry-level sampling (Levels 1–8)  
- Modular evaluation suite:
  - accuracy  
  - error tables  
  - noise robustness  
  - stress tests  
  - timing  
  - run_all (CLI entrypoint)

Run the full evaluation:

    python -m evaluation.regression_3src.run_all

---

## 6. Installation

### Install from PyPI (recommended)

The library is available as a pip-installable package:

    pip install inverse-source-em

This installs the full framework, including:
- surrogate EM models (inference-ready)
- physics solvers
- dataset utilities
- training and evaluation pipelines

### Install from source (development mode)

    git clone https://github.com/th0ma/inverse-source-em.git
    cd inverse-source-em
    pip install -e .

### Requirements

All dependencies are automatically installed via pip.  
For manual installation:

    pip install -r requirements.txt


---

## 7. Usage Examples

### Surrogate forward model

    from inverse_source_em.surrogate import SurrogateEM, SurrogateWrapper
    from inverse_source_em.physics import PhysicsTM

    phys = PhysicsTM()
    sur = SurrogateEM(
        path_E="models/surrogate_Esurf.pth",
        path_H="models/surrogate_Hsurf.pth",
        R=phys.R
    )
    wrap = SurrogateWrapper(sur)

    E = wrap.Esurf(rho_s, phi_s, theta)

### Classification inference

    import torch
    from inverse_source_em.training.classification_model import SourceCountResNet1D

    model = SourceCountResNet1D(...)
    model.load_state_dict(torch.load("models/classifier_1_to_5_resnet1d.pt"))
    model.eval()

    pred = model(X).argmax(dim=1)

---

## 8. Evaluation Suites

Each task has a complete evaluation suite under evaluation/:

- classification/
- regression_1src/
- regression_2src/
- regression_3src/
- surrogates/

Each suite includes:

- accuracy  
- error tables  
- noise robustness  
- stress tests  
- timing  
- run_all (CLI entrypoint)

Run an evaluation:

    python -m evaluation.regression_3src.run_all

---

## 9. Reproducibility

- Surrogate pipeline is canonical and frozen  
- Deterministic dataset generation  
- Fixed seeds in training scripts  
- Versioned datasets and models  
- Modular evaluation suites  
- Clean separation of concerns (physics / surrogate / ML / evaluation)

---

## 10. License

This project is licensed under the MIT License.

---

## 11. Citation

If you use this project in academic work, please cite:

Papadopoulos, T. D.  
Master Thesis in Applied Mathematics  
Hellenic Open University  

---

## 12. Roadmap

### Completed
- Canonical surrogate pipeline (frozen)
- Classification pipeline (1–5 sources)
- Regression pipelines: 1src, 2src, 3src
- Unified evaluation suites for all tasks
- GitHub release v0.1.0
- PyPI packaging and distribution
- Documentation unification (GitHub Wiki)
- Theory summary page (forward & inverse problem)
- API reference documentation
- Example-driven tutorials

---

## 13. Acknowledgments

I would like to express my sincere gratitude to my supervisor, Professor Nikolaos Tsitsas, whose guidance, insight, and continuous support were instrumental throughout the development of this research.

I would also like to acknowledge the team behind Microsoft Copilot for developing an advanced AI assistant that greatly facilitated the organization, refinement, and documentation of this project.
