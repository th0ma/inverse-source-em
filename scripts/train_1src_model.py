#!/usr/bin/env python
# coding: utf-8

"""
Train the Regression I (single-source localization) model.

This script is a thin wrapper around the main training pipeline implemented in
`inverse_source_em.training.train_1src`.

The model learns to predict the source location
$\left(\rho_s,\\,\phi_s\right)$
and source strength $I$ from boundary field measurements.

Run:
    python train_1src_model.py
"""


from inverse_source_em.training.train_1src import main


if __name__ == "__main__":
    main()
