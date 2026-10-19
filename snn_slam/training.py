"""Minimal training utilities and a deterministic synthetic smoke dataset."""

from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
from typing import Dict, Iterable, Optional

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
    depth_mask: Optional[Tensor] = None


def move_batch(batch: TrainBatch, device: str | torch.device) -> TrainBatch:
    """Move every tensor in a batch while preserving optional labels."""

    target = torch.device(device)
    return TrainBatch(
        events=batch.events.to(target),
        poses=batch.poses.to(target) if batch.poses is not None else None,
        depth=batch.depth.to(target) if batch.depth is not None else None,
        occupancy=batch.occupancy.to(target) if batch.occupancy is not None else None,
        depth_mask=batch.depth_mask.to(target) if batch.depth_mask is not None else None,
    )


def seed_everything(seed: int) -> None:
    """Seed Python and Torch for repeatable short experiments."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: str | Path,
    model: AllSNN_SLAM,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    """Save model, optimizer and experiment metadata in one portable file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "epoch": int(epoch),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, target)


def load_checkpoint(
    path: str | Path,
    model: AllSNN_SLAM,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, object]:
    """Restore a checkpoint and return its metadata.

    ``strict=True`` is intentional: silently accepting a changed architecture
    would produce a plausible but invalid SLAM experiment.
    """

    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if "model" not in checkpoint:
        raise ValueError(f"checkpoint has no model state: {path}")
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


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
    breakdown = slam_loss(output, batch.poses, batch.depth, batch.occupancy, depth_mask=batch.depth_mask)
    breakdown.total.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return LossBreakdown(*(value.detach() for value in breakdown.as_dict().values()))


@torch.no_grad()
def evaluate(model: AllSNN_SLAM, batch: TrainBatch) -> LossBreakdown:
    model.eval()
    output = model(batch.events, reset=True)
    breakdown = slam_loss(output, batch.poses, batch.depth, batch.occupancy, depth_mask=batch.depth_mask)
    return LossBreakdown(*(value.detach() for value in breakdown.as_dict().values()))


def _mean_metrics(metrics: Iterable[LossBreakdown]) -> Dict[str, float]:
    values = list(metrics)
    if not values:
        raise ValueError("cannot aggregate an empty metric sequence")
    keys = values[0].as_dict().keys()
    return {key: sum(float(item.as_dict()[key]) for item in values) / len(values) for key in keys}


def train_epoch(
    model: AllSNN_SLAM,
    loader,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device = "cpu",
    grad_clip: Optional[float] = 1.0,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Train over a DataLoader of :class:`TrainBatch` objects."""

    metrics = []
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        metrics.append(train_step(model, move_batch(batch, device), optimizer, grad_clip))
    return _mean_metrics(metrics)


@torch.no_grad()
def evaluate_epoch(
    model: AllSNN_SLAM,
    loader,
    device: str | torch.device = "cpu",
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate a DataLoader of :class:`TrainBatch` objects."""

    metrics = []
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        metrics.append(evaluate(model, move_batch(batch, device)))
    return _mean_metrics(metrics)


@torch.no_grad()
def evaluate_slam_epoch(
    model: AllSNN_SLAM,
    loader,
    device: str | torch.device = "cpu",
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate trajectory and depth metrics over a labelled DataLoader."""

    from .metrics import evaluate_output

    aggregate: Dict[str, list[float]] = {}
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = move_batch(batch, device)
        if batch.poses is None:
            continue
        output = model(batch.events, reset=True)
        values = evaluate_output(output, batch.poses, batch.depth, batch.depth_mask)
        for key, value in values.items():
            aggregate.setdefault(key, []).append(float(value))
    if not aggregate:
        raise ValueError("trajectory evaluation requires pose-labelled batches")
    return {key: sum(values) / len(values) for key, values in aggregate.items()}


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


__all__ = [
    "TrainBatch",
    "move_batch",
    "seed_everything",
    "save_checkpoint",
    "load_checkpoint",
    "train_step",
    "evaluate",
    "train_epoch",
    "evaluate_epoch",
    "evaluate_slam_epoch",
    "synthetic_batch",
]

