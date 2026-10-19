"""Minimal training utilities and a deterministic synthetic smoke dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor

from .losses import LossBreakdown, slam_loss
from .models import AllSNN_SLAM, SLAMOutput


@dataclass
class TrainBatch:
    events: Tensor
    poses: Optional[Tensor] = None
    depth: Optional[Tensor] = None
    occupancy: Optional[Tensor] = None


def train_step(
    model: AllSNN_SLAM,
    batch: TrainBatch,
    optimizer: torch.optim.Optimizer,
    grad_clip: Optional[float] = 1.0,
) -> LossBreakdown:
    """Run one truncated-BPTT update and return detached loss metrics."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(batch.events, reset=True)
    breakdown = slam_loss(output, batch.poses, batch.depth, batch.occupancy)
    breakdown.total.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return LossBreakdown(*(value.detach() for value in breakdown.as_dict().values()))


@torch.no_grad()
def evaluate(model: AllSNN_SLAM, batch: TrainBatch) -> LossBreakdown:
    model.eval()
    output = model(batch.events, reset=True)
    breakdown = slam_loss(output, batch.poses, batch.depth, batch.occupancy)
    return LossBreakdown(*(value.detach() for value in breakdown.as_dict().values()))


def synthetic_batch(
    batch_size: int = 2,
    steps: int = 8,
    height: int = 64,
    width: int = 64,
    device: str | torch.device = "cpu",
) -> TrainBatch:
    """Generate a small moving-edge batch for CI and first-run validation.

    It intentionally has no external dataset dependency.  The targets are
    physically consistent transforms and constant depth, so users can verify
    gradients before adapting a real event-camera dataloader.
    """

    device = torch.device(device)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing="ij",
    )
    events = torch.zeros(batch_size, steps, 2, height, width, device=device)
    poses = torch.eye(4, device=device).expand(batch_size, steps, 4, 4).clone()
    for t in range(steps):
        edge = ((xx - (2 * t / max(steps - 1, 1) - 1.0)).abs() < 0.04).float()
        events[:, t, 0] = edge
        events[:, t, 1] = torch.roll(edge, shifts=1, dims=-1)
        poses[:, t, 0, 3] = 0.02 * t
        poses[:, t, 1, 3] = 0.01 * torch.sin(torch.tensor(float(t), device=device))
    depth = torch.full((batch_size, steps, 1, height, width), 4.0, device=device)
    return TrainBatch(events=events, poses=poses, depth=depth)


__all__ = ["TrainBatch", "train_step", "evaluate", "synthetic_batch"]

