"""Losses for supervised and self-supervised training of the SNN SLAM model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .models import SLAMOutput
from .se3 import pose_geodesic_loss


@dataclass
class LossBreakdown:
    total: Tensor
    pose: Tensor
    depth: Tensor
    occupancy: Tensor
    smoothness: Tensor
    sparsity: Tensor

    def as_dict(self):
        return {
            "total": self.total,
            "pose": self.pose,
            "depth": self.depth,
            "occupancy": self.occupancy,
            "smoothness": self.smoothness,
            "sparsity": self.sparsity,
        }


def _zero_like(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def _masked_l1(predicted: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    if predicted.shape != target.shape:
        target = F.interpolate(
            target.reshape(-1, *target.shape[-3:]),
            size=predicted.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape_as(predicted)
    error = (predicted - target).abs()
    if mask is not None:
        if mask.shape != error.shape:
            mask = F.interpolate(mask.float().reshape(-1, 1, *mask.shape[-2:]), size=error.shape[-2:], mode="nearest")
            mask = mask.reshape_as(error)
        error = error * mask
        return error.sum() / mask.sum().clamp_min(1.0)
    return error.mean()


def slam_loss(
    output: SLAMOutput,
    target_poses: Optional[Tensor] = None,
    target_depth: Optional[Tensor] = None,
    target_occupancy: Optional[Tensor] = None,
    depth_mask: Optional[Tensor] = None,
    pose_weight: float = 1.0,
    depth_weight: float = 1.0,
    occupancy_weight: float = 0.5,
    smoothness_weight: float = 0.02,
    sparsity_weight: float = 1e-4,
) -> LossBreakdown:
    """Compute a training objective from any subset of available labels.

    With no labels the smoothness and spike-rate terms still provide a useful
    self-supervised warm start.  Pose targets should be homogeneous matrices
    with shape ``[B,T,4,4]``; depth/occupancy labels may be at another spatial
    resolution and are resized by the loss.
    """

    reference = output.twists
    pose = _zero_like(reference)
    depth = _zero_like(reference)
    occupancy = _zero_like(reference)
    if target_poses is not None:
        pose = pose_geodesic_loss(output.poses, target_poses.to(output.poses))
    if target_depth is not None:
        depth = _masked_l1(output.depth, target_depth.to(output.depth), depth_mask)
    if target_occupancy is not None:
        target = target_occupancy.to(output.occupancy)
        if target.shape != output.occupancy.shape:
            target = F.interpolate(
                target.reshape(-1, *target.shape[-3:]),
                size=output.occupancy.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).reshape_as(output.occupancy)
        occupancy = F.binary_cross_entropy(output.occupancy.clamp(1e-4, 1 - 1e-4), target.clamp(0, 1))
    if output.twists.shape[1] > 1:
        smoothness = (output.twists[:, 1:] - output.twists[:, :-1]).abs().mean()
    else:
        smoothness = _zero_like(reference)
    if output.spikes is None:
        sparsity = _zero_like(reference)
    else:
        sparsity = output.spikes.float().mean()
    total = (
        pose_weight * pose
        + depth_weight * depth
        + occupancy_weight * occupancy
        + smoothness_weight * smoothness
        + sparsity_weight * sparsity
    )
    return LossBreakdown(total, pose, depth, occupancy, smoothness, sparsity)


__all__ = ["LossBreakdown", "slam_loss"]

