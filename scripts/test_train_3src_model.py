"""
Fast smoke test for the 3‑source training pipeline.

It does NOT train on real data.
It:
- builds a tiny synthetic dataset
- runs train_stage for 1 epoch
- checks that a checkpoint is written
"""

import os
import shutil
import torch
from torch.utils.data import TensorDataset, DataLoader

from inverse_source_em.training.model_3src import ThreeSourceMultiHeadBig
from inverse_source_em.training.train_3src import train_stage, DEVICE


class DummyScaler:
    """Identity scaler with the same API as sklearn's MinMaxScaler."""
    def inverse_transform(self, arr):
        return arr


def test_train_3src_one_epoch():

    # Tiny synthetic dataset: 64 samples, 120 features, 6 targets
    batch_size = 16
    n_samples = 64
    input_dim = 120
    output_dim = 6

    X = torch.randn(n_samples, input_dim, dtype=torch.float32)
    y = torch.randn(n_samples, output_dim, dtype=torch.float32)

    train_ds = TensorDataset(X, y)
    test_ds = TensorDataset(X, y)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = ThreeSourceMultiHeadBig(input_dim).to(DEVICE)

    # Checkpoint directory (test-only)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_dir = os.path.join(root_dir, "models", "regression_3src_test")

    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    scaler_y = DummyScaler()

    # Train for a single epoch, minimal patience
    model = train_stage(
        stage_index=1,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        scaler_y=scaler_y,
        ckpt_dir=ckpt_dir,
        epochs=1,
        lr=5e-4,
        patience=1,
        block=1,
    )

    ckpt_path = os.path.join(ckpt_dir, "best_model_stage_1.pt")
    assert os.path.isfile(ckpt_path), "Checkpoint was not created."
