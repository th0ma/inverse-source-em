"""
Script entry point for training the 3‑source regression model
across all geometry stages (1..8).
"""

import os
from inverse_source_em.training.train_3src import train_full_curriculum


def main():
    # Project root = parent of this scripts/ directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_dir = os.path.join(root_dir, "data", "regression_3src")
    ckpt_dir = os.path.join(root_dir, "models", "regression_3src")

    print("Training 3‑source model")
    print("  data_dir :", data_dir)
    print("  ckpt_dir :", ckpt_dir)

    train_full_curriculum(
        data_dir=data_dir,
        ckpt_dir=ckpt_dir,
        stages=(1, 2, 3, 4, 5, 6, 7, 8),
    )


if __name__ == "__main__":
    main()
