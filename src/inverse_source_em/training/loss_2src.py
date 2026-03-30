"""
loss_2src.py

Structured loss for the two-source inverse EM regression task.

This module implements a composite loss function tailored for the
two-source localization problem. It includes:

1. TailWeightScheduler
   Dynamically adjusts the weight of a loss term based on its p99
   (99th percentile) error. This helps stabilize training by increasing
   emphasis on tail errors when they exceed a target threshold.

2. angle_loss
   Computes the angular difference between predicted and true 2D vectors.

3. area_constraint
   Encourages geometric consistency by matching the parallelogram area
   spanned by the predicted and true source vectors.

4. structured_loss
   Combines:
       - distance errors for source A and B
       - angular errors for source A and B
       - area constraint
       - dynamic tail-weight scheduling
       - full logging of p99 metrics and weights

The output is a dictionary of weighted loss components and diagnostics,
allowing fine-grained monitoring during training.
"""

import torch


class TailWeightScheduler:
    """
    Dynamically adjusts the weight of a loss term based on its p99 value.

    If the 99th percentile error exceeds a target threshold, the weight
    increases gradually according to a linear schedule:

        w = min_w + (p99 - target_p99) / delta

    clipped to [min_w, max_w].

    Parameters
    ----------
    config : dict
        Must contain:
            - "target_p99" : float
            - "min_w"      : float
            - "max_w"      : float
            - "delta"      : float

    Notes
    -----
    - This scheduler is typically used for distance and angle terms.
    - It helps prevent the model from ignoring difficult tail cases.
    """

    def __init__(self, config):
        self.target_p99 = config["target_p99"]
        self.min_w = config["min_w"]
        self.max_w = config["max_w"]
        self.delta = config["delta"]

    def __call__(self, current_p99: float) -> float:
        """
        Compute the dynamic weight based on the current p99 value.

        Parameters
        ----------
        current_p99 : float
            The observed 99th percentile error for a loss component.

        Returns
        -------
        float
            Updated weight in [min_w, max_w].
        """
        excess = max(0.0, current_p99 - self.target_p99)
        steps = excess / self.delta
        return min(self.min_w + steps, self.max_w)


def angle_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the angular difference between two 2D vectors.

    Uses cosine similarity with numerical stability, then applies arccos.

    Parameters
    ----------
    pred : torch.Tensor of shape (N, 2)
        Predicted 2D vectors.
    target : torch.Tensor of shape (N, 2)
        Ground-truth 2D vectors.

    Returns
    -------
    torch.Tensor of shape (N,)
        Angular error in radians.
    """
    dot = torch.sum(pred * target, dim=1)
    norm_pred = torch.norm(pred, dim=1)
    norm_target = torch.norm(target, dim=1)

    cos_theta = dot / (norm_pred * norm_target + 1e-8)
    cos_theta = torch.clamp(cos_theta, -1 + 1e-6, 1 - 1e-6)

    return torch.acos(cos_theta)


def area_constraint(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Enforce geometric consistency via parallelogram area preservation.

    For two 2D vectors (v1, v2), the area of the parallelogram they span is:

        area = |x1*y2 - y1*x2|

    This term encourages the predicted pair of vectors to preserve the
    relative area spanned by the true pair.

    Parameters
    ----------
    preds : torch.Tensor of shape (N, 4)
        Predicted coordinates [x1, y1, x2, y2].
    targets : torch.Tensor of shape (N, 4)
        Ground-truth coordinates [x1, y1, x2, y2].

    Returns
    -------
    torch.Tensor (scalar)
        Mean absolute area difference.
    """
    pred_area = torch.abs(preds[:, 0] * preds[:, 3] - preds[:, 1] * preds[:, 2])
    target_area = torch.abs(targets[:, 0] * targets[:, 3] - targets[:, 1] * targets[:, 2])
    return torch.mean(torch.abs(pred_area - target_area))


def structured_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    config: dict,
    sched_distA: TailWeightScheduler | None = None,
    sched_distB: TailWeightScheduler | None = None,
    sched_angleA: TailWeightScheduler | None = None,
    sched_angleB: TailWeightScheduler | None = None,
):
    """
    Compute the full structured loss for the two-source regression task.

    The loss combines:
        - distance errors for source A and B
        - angular errors for source A and B
        - area constraint
        - dynamic tail-weight scheduling
        - full logging of p99 metrics and weights

    Parameters
    ----------
    preds : torch.Tensor of shape (N, 4)
        Predicted normalized Cartesian coordinates:
            [x1/R, y1/R, x2/R, y2/R]
    targets : torch.Tensor of shape (N, 4)
        Ground-truth normalized Cartesian coordinates.
    config : dict
        Must contain:
            "LAMBDA_AREA", "LAMBDA_DISTA", "LAMBDA_DISTB",
            "LAMBDA_ANGLEA", "LAMBDA_ANGLEB"
    sched_distA, sched_distB : TailWeightScheduler or None
        Optional dynamic schedulers for distance terms.
    sched_angleA, sched_angleB : TailWeightScheduler or None
        Optional dynamic schedulers for angle terms.

    Returns
    -------
    dict
        Dictionary containing:
            - weighted loss components
            - dynamic weights
            - p99 diagnostics

        Example keys:
            "area", "distA", "distB", "angleA", "angleB",
            "w_distA", "w_distB", "w_angleA", "w_angleB",
            "p99_distA", "p99_distB", "p99_angleA", "p99_angleB"

    Notes
    -----
    - This function does NOT sum the losses; it returns a dictionary.
      The training loop is expected to sum the components as needed.
    - p99 metrics provide robustness diagnostics for tail performance.
    """
    # Distances
    distA = torch.norm(preds[:, 0:2] - targets[:, 0:2], dim=1)
    distB = torch.norm(preds[:, 2:4] - targets[:, 2:4], dim=1)

    # Angles
    angA = angle_loss(preds[:, 0:2], targets[:, 0:2])
    angB = angle_loss(preds[:, 2:4], targets[:, 2:4])

    # Area
    area_val = area_constraint(preds, targets)

    # Tail metrics (p99)
    p99_distA = distA.quantile(0.99).item()
    p99_distB = distB.quantile(0.99).item()
    p99_angA = angA.quantile(0.99).item()
    p99_angB = angB.quantile(0.99).item()

    # Dynamic weights
    w_distA = sched_distA(p99_distA) if sched_distA else 1.0
    w_distB = sched_distB(p99_distB) if sched_distB else 1.0
    w_angleA = sched_angleA(p99_angA) if sched_angleA else 1.0
    w_angleB = sched_angleB(p99_angB) if sched_angleB else 1.0

    # Weighted components
    loss_area = config["LAMBDA_AREA"] * area_val
    loss_distA = config["LAMBDA_DISTA"] * w_distA * distA.mean()
    loss_distB = config["LAMBDA_DISTB"] * w_distB * distB.mean()
    loss_angleA = config["LAMBDA_ANGLEA"] * w_angleA * angA.mean()
    loss_angleB = config["LAMBDA_ANGLEB"] * w_angleB * angB.mean()

    loss_dict = {
        "area": loss_area,
        "distA": loss_distA,
        "distB": loss_distB,
        "angleA": loss_angleA,
        "angleB": loss_angleB,

        # Logging weights
        "w_distA": w_distA,
        "w_distB": w_distB,
        "w_angleA": w_angleA,
        "w_angleB": w_angleB,

        # Logging p99 metrics
        "p99_distA": p99_distA,
        "p99_distB": p99_distB,
        "p99_angleA": p99_angA,
        "p99_angleB": p99_angB,
    }

    return loss_dict
