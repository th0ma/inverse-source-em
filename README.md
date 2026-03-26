# Inverse Source EM

**Deep Learning Methods for Electromagnetic Inverse Source Problems**

This repository provides a complete scientific machine learning framework for
electromagnetic inverse source problems in cylindrical geometries. It combines:

- **Physics-based analytical solvers** (Helmholtz equation, TM polarization)
- **Neural surrogate models** for fast forward simulations
- **Deep learning pipelines** for classification and localization of internal sources
- **Modular evaluation suites** for reproducible benchmarking

The project has been fully migrated from notebooks into a clean, modular,
pip-installable Python package.

---

## 1. Abstract

We study an inverse source problem inside a dielectric cylinder, belonging to
the broader class of inverse scattering problems. The goal is to recover the:

- **number** of internal line sources,
- **positions** (Cartesian or polar),
- **strengths**,

using only electric surface field measurements at a single frequency.

The forward problem is solved analytically using cylindrical harmonics and
Bessel/Hankel special functions. The `PhysicsTM` class provides a stable,
vectorized implementation of the TM-polarized Helmholtz equation.

On top of this, we build three machine learning pipelines:

1. **Surrogate EM models** for fast forward simulations  
2. **Classification models** for predicting the number of sources (1–5)  
3. **Regression models** for localizing 1 or 2 internal sources  

The surrogate models enable large-scale dataset generation with high accuracy.
The regression pipelines incorporate geometric sampling, canonical ordering,
structured losses, and permutation-invariant evaluation.

The results demonstrate high accuracy, strong noise robustness, and reliable
generalization even in challenging geometries.

---

## 2. Project Structure

```
inverse-source-em/
│
├── data/
│   ├── classification/
│   ├── regression_1src/
│   ├── regression_2src/
│   └── surrogate/
│
├── models/
│   ├── surrogate_Esurf.pth
│   ├── surrogate_Hsurf.pth
│   ├── classifier_1_to_5_resnet1d.pt
│   ├── best_model_2src.pth
│   └── regression_1src/
│       └── best_epoch_XXXX.pth
│
├── scripts/
│   ├── make_1src_dataset.py
│   ├── make_2src_dataset.py
│   ├── make_classification_dataset.py
│   ├── make_surrogate_dataset.py
│   ├── train_1src_model.py
│   ├── train_2src_model.py
│   ├── train_classification_model.py
│   └── train_surrogate_models.py
│
├── evaluation/
│   ├── classification/
│   ├── regression_1src/
│   ├── regression_2src/
│   └── surrogates/
│
├── notebooks/
│   ├── regression_1src_evaluation.ipynb
│   ├── regression_2src_evaluation.ipynb
│   ├── classification_evaluation.ipynb
│   └── physics_vs_surrogate.ipynb
│
├── src/inverse_source_em/
│   ├── surrogate/
│   ├── data/
│   ├── physics/
│   ├── training/
│   └── utils/
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 3. Surrogate Pipeline (Canonical & Frozen)

The surrogate pipeline is **complete and frozen**. It provides:

- `SurrogateEM` — unified EM surrogate model  
- `SurrogateMLP` — configurable MLP surrogate  
- `SurrogateWrapper` — PhysicsTM-compatible interface  
- Sampling and dataset generation utilities  

These models form the **reference baseline** for all inverse tasks.

---

## 4. Classification Pipeline (Completed)

Predicts the number of internal sources (1–5) from surface field intensities.

Includes:

- Sampling (`sampling_classification.py`)
- Dataset generation (`generator_classification.py`)
- PyTorch dataset wrapper (`dataset_classification.py`)
- ResNet1D classifier (`classification_model.py`)
- Training script (`train_classification_model.py`)
- Evaluation suite (`evaluation/classification/`)

**Baseline performance:**

- Validation accuracy: ~0.984  
- Test accuracy: ~0.985  
- Strong robustness to additive noise  

---

## 5. Regression Pipelines

### 5.1 Single-Source Localization (1src)

- Dataset: `data/regression_1src/dataset_1src.npz`
- Model: `models/regression_1src/best_epoch_XXXX.pth`
- Predicts:  
  - Cartesian coordinates (x, y)  
  - Source strength I  
- Evaluation suite:  
  - accuracy, error tables, noise robustness, timing, run_all

### 5.2 Two-Source Localization (2src)

- Dataset: `data/regression_2src/`
- Model: `models/best_model_2src.pth`
- Predicts:  
  - (x₁, y₁), (x₂, y₂)  
- Fully **permutation-invariant** evaluation  
- Includes:  
  - p99 zoom distributions  
  - upper-tail analysis  
  - R² metrics  
  - scatter plots with regression lines  
  - unified metrics table  

---

## 6. Installation

### From source

```bash
git clone https://github.com/<username>/inverse-source-em.git
cd inverse-source-em
pip install -e .
```

### Requirements

```bash
pip install -r requirements.txt
```

---

## 7. Usage Examples

### Surrogate forward model

```python
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
```

### Classification inference

```python
import torch
from inverse_source_em.training.classification_model import SourceCountResNet1D

model = SourceCountResNet1D(...)
model.load_state_dict(torch.load("models/classifier_1_to_5_resnet1d.pt"))
model.eval()

pred = model(X).argmax(dim=1)
```

---

## 8. Evaluation Suites

Each task has a complete evaluation suite under `evaluation/`:

- `classification/`
- `regression_1src/`
- `regression_2src/`
- `surrogates/`

Each suite includes:

- accuracy  
- error tables  
- noise robustness  
- timing  
- run_all (CLI entrypoint)

Run an evaluation:

```bash
python -m evaluation.regression_2src.run_all
```

---

## 9. Reproducibility

- Surrogate pipeline is **canonical** and frozen  
- Deterministic dataset generation  
- Fixed seeds in training scripts  
- Versioned datasets and models  
- Modular evaluation suites  

---

## 10. License

This project is licensed under the **MIT License**.

---

## 11. Citation

If you use this project in academic work, please cite:

**Papadopoulos, T. D.**  
*Master Thesis in Applied Mathematics*  
Hellenic Open University  

---

## 12. Roadmap

- [x] Surrogate pipeline (canonical)  
- [x] Classification pipeline  
- [x] Regression 1src  
- [x] Regression 2src  
- [x] Unified evaluation suites  
- [ ] GitHub release v0.9.0  
- [ ] pip packaging  
- [ ] Regression III (variable number of sources)  
- [ ] v1.0.0 stable release  

---

## 13. Acknowledgments

I would like to express my sincere gratitude to my supervisor, Professor Nikolaos Tsitsas, whose guidance, insight, and continuous support were instrumental throughout the development of this research.

I would also like to acknowledge the team behind Microsoft Copilot for developing an advanced AI assistant that greatly facilitated the organization, refinement, and documentation of this project.
