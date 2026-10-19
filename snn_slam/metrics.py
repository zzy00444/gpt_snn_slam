"""Lightweight trajectory and depth metrics for sequence smoke evaluation."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor

from .models import SLAMOutput
from .se3 import inverse


def _as_batched(poses: Tensor) -> Tensor:
    if poses.ndim == 3:
        poses = poses.unsqueeze(0)
    if poses.ndim != 4 or poses.shape[-2:] != (4, 4):
        raise ValueError("poses must have shape [T,4,4] or [B,T,4,4]")
    return poses


def absolute_trajectory_error(predicted: Tensor, target: Tensor) -> Tensor:
    """Translation RMSE after aligning both trajectories at their first pose."""

    predicted, target = _as_batched(predicted), _as_batched(target)
    if predicted.shape != target.shape:
        raise ValueError("predicted and target trajectories must have equal shape")
    predicted = inverse(predicted[:, :1]) @ predicted
    target = inverse(target[:, :1]) @ target
    error = predicted[..., :3, 3] - target[..., :3, 3]
    return error.square().sum(-1).mean().sqrt()


def relative_pose_error(predicted: Tensor, target: Tensor) -> Tensor:
    """Mean translation error between consecutive relative motions."""

    predicted, target = _as_batched(predicted), _as_batched(target)
    if predicted.shape != target.shape:
        raise ValueError("predicted and target trajectories must have equal shape")
    if predicted.shape[1] < 2:
        return predicted.sum() * 0.0
    pred_delta = inverse(predicted[:, :-1]) @ predicted[:, 1:]
    target_delta = inverse(target[:, :-1]) @ target[:, 1:]
    error = pred_delta[..., :3, 3] - target_delta[..., :3, 3]
    return error.square().sum(-1).mean().sqrt()


def depth_metrics(predicted: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
    """Return valid-pixel AbsRel and RMSE for metric depth predictions."""

    if target.ndim == predicted.ndim - 1:
        target = target.unsqueeze(-3)
    if predicted.shape != target.shape:
        raise ValueError("predicted and target depth tensors must have equal shape")
    valid = (target > 0) if mask is None else mask.bool()
    if valid.ndim == predicted.ndim - 1:
        valid = valid.unsqueeze(-3)
    valid = valid.expand_as(predicted)
    count = valid.sum().clamp_min(1)
    difference = (predicted - target).abs()
    abs_rel = (difference / target.abs().clamp_min(1e-3) * valid).sum() / count
    rmse = (((predicted - target).square() * valid).sum() / count).sqrt()
    return {"abs_rel": abs_rel, "rmse": rmse}


def evaluate_output(
    output: SLAMOutput,
    target_poses: Tensor,
    target_depth: Optional[Tensor] = None,
    depth_mask: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Calculate trajectory metrics and optional depth metrics."""

    result = {
        "ate_rmse": absolute_trajectory_error(output.poses, target_poses),
        "rpe_translation": relative_pose_error(output.poses, target_poses),
    }
    if target_depth is not None:
        result.update({f"depth_{key}": value for key, value in depth_metrics(output.depth, target_depth, depth_mask).items()})
    return result


__all__ = ["absolute_trajectory_error", "relative_pose_error", "depth_metrics", "evaluate_output"]

