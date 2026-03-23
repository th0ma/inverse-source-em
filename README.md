# Inverse Source EM

**Exploring Inverse Source Problems with Deep Learning Methods**

This repository contains a complete scientific machine learning framework for
electromagnetic inverse source problems in cylindrical geometries. It combines:

- **Physics-based analytical solvers** (Helmholtz equation, TM polarization)
- **Neural surrogate models** for fast forward simulations
- **Deep learning pipelines** for classification and localization of internal sources

The project is currently being migrated from Jupyter notebooks into a clean,
modular, pip-installable Python package.

---

## 1. Abstract

This work investigates an inverse source problem in a dielectric cylinder,
belonging to the broader class of inverse scattering problems. The objective is
to recover the **number**, **positions**, and **strengths** of internal line
sources using measurements of electric surface fields at a single operating
frequency.

The forward problem is solved through an analytical, numerically stable
implementation of the Helmholtz equation for TM polarization, based on
cylindrical harmonics and Bessel/Hankel special functions. This implementation
is provided by the `PhysicsTM` class, which computes surface fields in a fully
vectorized manner.

Building on this foundation, a complete deep learning pipeline is developed:

- A **classification model** identifies the number of active sources (1–5).
- **Localization models** recover the Cartesian coordinates of 1, 2, or 3
  internal sources.
- Training data are generated using high-accuracy **surrogate models**, enabling
  large-scale dataset creation.

For the multi-source localization problems, specialized techniques are used:
geometric sampling constraints, canonical ordering, structured loss, curriculum
learning, and noise consistency training. In the three-source case, gradual
geometric difficulty scaling and permutation-invariant evaluation are essential
for stable training.

The results demonstrate high accuracy, strong noise robustness, and reliable
generalization even in pathological geometries. The work highlights the
capability of deep learning surrogates to solve challenging inverse problems
with speed and stability surpassing classical methods.

---

## 2. Project Structure

```
inverse-source-em/
    data/
        surrogate/
        classification/
    models/
    notebooks/
    src/inverse_source_em/
        surrogate/
        data/
        physics/
        training/
        utils/
    README.md
    pyproject.toml
    requirements.txt
```

---

## 3. Surrogate Pipeline (Canonical)

The surrogate pipeline is **complete and frozen**. It provides:

- `SurrogateEM` — unified EM surrogate model  
- `SurrogateMLP` — configurable MLP surrogate  
- `SurrogateWrapper` — PhysicsTM-compatible interface  
- Sampling and dataset generation utilities  

These components form the **reference baseline** for all subsequent pipelines.

---

## 4. Classification Pipeline

The classification pipeline predicts the number of internal sources (1–5) from
surface field intensities.

It includes:

- Custom sampling (`sampling_classification.py`)
- Dataset generation (`generator_classification.py`)
- PyTorch dataset wrapper (`dataset_classification.py`)
- ResNet1D classifier (`classification_model.py`)
- Training script (`train_classification.py`)
- Evaluation notebook (`classification_evaluation.ipynb`)

**Baseline performance:**

- Validation accuracy: ~0.984  
- Test accuracy: ~0.985  
- Strong robustness to additive noise  

---

## 5. Installation

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

## 6. Usage Examples

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

## 7. Notebooks

- `physics_vs_surrogate.ipynb` — comparison of analytical and surrogate models  
- `classification_evaluation.ipynb` — evaluation of the classification pipeline  

All notebooks can be placed inside a `notebooks/` folder and will run normally
as long as `src/` is added to the Python path.

---

## 8. Reproducibility

- Surrogate pipeline is **canonical** and does not change.  
- Classification dataset generation is deterministic.  
- Training scripts use fixed seeds where appropriate.  
- All models and datasets are versioned.  

---

## 9. License

This project is licensed under the **MIT License**.

---

## 10. Citation

If you use this project in academic work, please cite:

**Papadopoulos, T. D.**  
*Master Thesis in Applied Mathematics*  
Hellenic Open University  

---

## 11. Roadmap

- [x] Surrogate pipeline (canonical)  
- [x] Classification pipeline (baseline)  
- [ ] Localization pipeline (1–3 sources)  
- [ ] GitHub release v0.9.0  
- [ ] pip packaging  
- [ ] Experiments (LeakyReLU, GELU, transformers, curriculum learning)  
- [ ] v1.0.0 stable release  
