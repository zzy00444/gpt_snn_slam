"""Differentiable SE(3) utilities for neural odometry."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def skew(vector: Tensor) -> Tensor:
    """Return the 3x3 cross-product matrix for vectors ``[...,3]``."""

    if vector.shape[-1] != 3:
        raise ValueError("vector must end in 3")
    x, y, z = vector.unbind(-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        (
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ),
        dim=-1,
    ).reshape(vector.shape[:-1] + (3, 3))


def so3_exp(rotation_vector: Tensor) -> Tensor:
    """Rodrigues exponential map from an axis-angle vector."""

    if rotation_vector.shape[-1] != 3:
        raise ValueError("rotation_vector must end in 3")
    theta2 = (rotation_vector * rotation_vector).sum(-1, keepdim=True)
    theta = theta2.sqrt()
    omega_hat = skew(rotation_vector)
    eye = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    eye = eye.expand(rotation_vector.shape[:-1] + (3, 3))
    # Taylor series makes gradients finite around zero.
    a = torch.where(
        theta2 < 1e-8,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(theta) / theta.clamp_min(1e-8),
    )
    b = torch.where(
        theta2 < 1e-8,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-8),
    )
    return eye + a[..., None] * omega_hat + b[..., None] * (omega_hat @ omega_hat)


def se3_exp(twist: Tensor) -> Tensor:
    """Exponential map for twists ``[tx, ty, tz, rx, ry, rz]``."""

    if twist.shape[-1] != 6:
        raise ValueError("twist must end in 6")
    translation = twist[..., :3]
    rotation_vector = twist[..., 3:]
    rotation = so3_exp(rotation_vector)
    omega_hat = skew(rotation_vector)
    theta2 = (rotation_vector * rotation_vector).sum(-1, keepdim=True)
    theta = theta2.sqrt()
    eye = torch.eye(3, dtype=twist.dtype, device=twist.device)
    eye = eye.expand(twist.shape[:-1] + (3, 3))
    a = torch.where(
        theta2 < 1e-8,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-8),
    )
    b = torch.where(
        theta2 < 1e-8,
        1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0,
        (theta - torch.sin(theta)) / theta.clamp_min(1e-8).pow(3),
    )
    v_matrix = eye + a[..., None] * omega_hat + b[..., None] * (omega_hat @ omega_hat)
    transformed_translation = (v_matrix @ translation.unsqueeze(-1)).squeeze(-1)
    result = torch.zeros(twist.shape[:-1] + (4, 4), dtype=twist.dtype, device=twist.device)
    result[..., :3, :3] = rotation
    result[..., :3, 3] = transformed_translation
    result[..., 3, 3] = 1.0
    return result


def compose(first: Tensor, second: Tensor) -> Tensor:
    """Compose homogeneous transforms (apply ``second`` after ``first``)."""

    return first @ second


def inverse(transform: Tensor) -> Tensor:
    """Stable inverse of a rigid transform."""

    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    result = torch.zeros_like(transform)
    rotation_t = rotation.transpose(-1, -2)
    result[..., :3, :3] = rotation_t
    result[..., :3, 3] = -(rotation_t @ translation.unsqueeze(-1)).squeeze(-1)
    result[..., 3, 3] = 1.0
    return result


def integrate_twists(twists: Tensor, initial: Tensor | None = None) -> Tensor:
    """Integrate ``[B,T,6]`` twists into world poses ``[B,T,4,4]``."""

    if twists.ndim < 2 or twists.shape[-1] != 6:
        raise ValueError("twists must have shape [..., T, 6]")
    increments = se3_exp(twists)
    if initial is None:
        initial = torch.eye(4, dtype=twists.dtype, device=twists.device)
    if initial.ndim == 2:
        initial = initial.expand(twists.shape[:-2] + (4, 4))
    poses = []
    current = initial
    for increment in increments.unbind(-3):
        current = current @ increment
        poses.append(current)
    if not poses:
        return twists.new_empty(twists.shape[:-1] + (4, 4))
    return torch.stack(poses, dim=-3)


def pose_translation(poses: Tensor) -> Tensor:
    return poses[..., :3, 3]


def pose_geodesic_loss(predicted: Tensor, target: Tensor) -> Tensor:
    """A smooth translation + rotation loss for pose supervision."""

    if predicted.shape != target.shape:
        raise ValueError("predicted and target pose tensors must have equal shape")
    translation = (predicted[..., :3, 3] - target[..., :3, 3]).abs().mean()
    r_rel = predicted[..., :3, :3].transpose(-1, -2) @ target[..., :3, :3]
    trace = r_rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    # 1-cos(theta) is smooth at identity (acos(trace) has infinite gradient
    # there) and is monotonic for the practical [0, pi] pose-error range.
    rotation = ((3.0 - trace) / 2.0).clamp_min(0.0)
    return translation + rotation.mean()


__all__ = [
    "skew",
    "so3_exp",
    "se3_exp",
    "compose",
    "inverse",
    "integrate_twists",
    "pose_translation",
    "pose_geodesic_loss",
]

