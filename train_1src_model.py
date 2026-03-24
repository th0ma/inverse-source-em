#!/usr/bin/env python
# coding: utf-8

"""
Train the Regression I (single-source localization) model.

This script is a thin wrapper that simply calls the training
pipeline implemented inside the package:

    inverse_source_em.training.train_1src

Run:
    python train_1src_model.py
"""

from inverse_source_em.training.train_1src import main


if __name__ == "__main__":
    main()
