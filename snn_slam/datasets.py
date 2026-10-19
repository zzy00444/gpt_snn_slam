"""Adapters for frame-based TUM RGB-D sequences.

TUM RGB-D is not an event-camera dataset.  This module therefore provides an
explicit frame-to-pseudo-event bridge for development and regression testing:
positive/negative channels contain thresholded log-intensity changes between
successive RGB frames.  The adapter keeps the original depth and ground-truth
trajectory so the SNN SLAM model can be trained and evaluated on real motion
before an event-camera dataset is available.
"""

from __future__ import annotations

import bisect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .training import TrainBatch


@dataclass(frozen=True)
class TUMFrame:
    timestamp: float
    rgb_path: Path
    depth_path: Optional[Path]
    pose: Optional[Tensor]


def _parse_tum_file(path: Path, fields: int) -> List[Tuple[float, List[str]]]:
    records: List[Tuple[float, List[str]]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = line.split()
            if len(values) >= fields:
                try:
                    records.append((float(values[0]), values[1:]))
                except ValueError:
                    continue
    return records


def _nearest(records: Sequence[Tuple[float, object]], timestamp: float, max_delta: float) -> object | None:
    if not records:
        return None
    timestamps = [item[0] for item in records]
    index = bisect.bisect_left(timestamps, timestamp)
    candidates = []
    if index < len(records):
        candidates.append(records[index])
    if index:
        candidates.append(records[index - 1])
    nearest = min(candidates, key=lambda item: abs(item[0] - timestamp))
    return nearest[1] if abs(nearest[0] - timestamp) <= max_delta else None


def _quaternion_pose(values: Sequence[str]) -> Tensor:
    if len(values) < 7:
        raise ValueError("TUM pose must contain tx ty tz qx qy qz qw")
    tx, ty, tz, qx, qy, qz, qw = (float(item) for item in values[:7])
    quaternion = torch.tensor([qx, qy, qz, qw], dtype=torch.float32)
    quaternion = quaternion / quaternion.norm().clamp_min(1e-8)
    x, y, z, w = quaternion
    rotation = torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        )
    ).reshape(3, 3)
    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, :3] = rotation
    pose[:3, 3] = torch.tensor([tx, ty, tz])
    return pose


def _inverse_pose(pose: Tensor) -> Tensor:
    rotation = pose[:3, :3]
    result = torch.eye(4, dtype=pose.dtype)
    result[:3, :3] = rotation.transpose(0, 1)
    result[:3, 3] = -(rotation.transpose(0, 1) @ pose[:3, 3])
    return result


def _resolve_sequence_root(root: os.PathLike[str] | str) -> Path:
    root_path = Path(root)
    if (root_path / "rgb").is_dir():
        return root_path
    nested = [candidate for candidate in root_path.iterdir() if candidate.is_dir() and (candidate / "rgb").is_dir()]
    if len(nested) == 1:
        return nested[0]
    raise FileNotFoundError(f"Could not find a TUM sequence root containing rgb/: {root_path}")


class TUMRGBDWindowDataset:
    """Real-frame smoke/training adapter for one TUM RGB-D sequence.

    Each item returns an unbatched :class:`TrainBatch` with ``events [T,2,H,W]``,
    relative ground-truth poses, metric depth and a valid-depth mask.  The
    model can consume ``item.events.unsqueeze(0)`` directly.
    """

    def __init__(
        self,
        root: os.PathLike[str] | str,
        window: int = 8,
        stride: int = 1,
        image_size: Tuple[int, int] = (64, 96),
        event_threshold: float = 0.08,
        depth_scale: float = 5000.0,
        association_delta: float = 0.03,
        require_depth: bool = True,
        require_pose: bool = True,
        max_frames: Optional[int] = None,
    ) -> None:
        if window < 2 or stride <= 0:
            raise ValueError("window must be >= 2 and stride must be positive")
        self.root = _resolve_sequence_root(root)
        self.window = int(window)
        self.stride = int(stride)
        self.image_size = tuple(int(x) for x in image_size)
        self.event_threshold = float(event_threshold)
        self.depth_scale = float(depth_scale)
        self.association_delta = float(association_delta)
        if min(self.image_size) <= 0 or self.event_threshold <= 0 or self.depth_scale <= 0:
            raise ValueError("image size, event threshold, and depth scale must be positive")
        self.frames = self._load_index()
        if require_depth:
            self.frames = [frame for frame in self.frames if frame.depth_path is not None]
        if require_pose:
            self.frames = [frame for frame in self.frames if frame.pose is not None]
        if max_frames is not None:
            self.frames = self.frames[: int(max_frames)]
        if len(self.frames) < self.window:
            raise ValueError(f"sequence contains {len(self.frames)} usable frames, window={self.window}")

    def _load_index(self) -> List[TUMFrame]:
        rgb_records = _parse_tum_file(self.root / "rgb.txt", fields=2)
        depth_records = _parse_tum_file(self.root / "depth.txt", fields=2)
        pose_records = [(stamp, _quaternion_pose(values)) for stamp, values in _parse_tum_file(self.root / "groundtruth.txt", 8)]
        if not rgb_records:
            raise FileNotFoundError(f"No RGB records found in {self.root / 'rgb.txt'}")
        frames: List[TUMFrame] = []
        for timestamp, values in rgb_records:
            rgb_path = self.root / values[0]
            if not rgb_path.exists():
                continue
            depth_record = _nearest(depth_records, timestamp, self.association_delta)
            pose = _nearest(pose_records, timestamp, self.association_delta)
            if isinstance(depth_record, list) and depth_record:
                depth_path = self.root / depth_record[0]
            else:
                depth_path = None
            frames.append(TUMFrame(timestamp, rgb_path, depth_path, pose))
        return frames

    def __len__(self) -> int:
        return 1 + (len(self.frames) - self.window) // self.stride

    def _load_image(self, path: Path, mode: str) -> Tensor:
        try:
            import numpy as np
            from PIL import Image
        except ImportError as error:  # pragma: no cover - depends on optional data extra
            raise ImportError("TUM adapter requires Pillow and numpy; install `pip install -e .[data]`") from error
        height, width = self.image_size
        with Image.open(path) as image:
            image = image.convert(mode)
            resampling = Image.Resampling.NEAREST if mode == "I" else Image.Resampling.BILINEAR
            image = image.resize((width, height), resampling)
            array = np.asarray(image).copy()
        return torch.from_numpy(array).float()

    def _load_depth(self, path: Optional[Path]) -> Optional[Tensor]:
        if path is None or not path.exists():
            return None
        depth = self._load_image(path, "I") / self.depth_scale
        return depth.unsqueeze(0)

    def __getitem__(self, index: int) -> TrainBatch:
        start = int(index) * self.stride
        if index < 0 or start + self.window > len(self.frames):
            raise IndexError(index)
        selected = self.frames[start : start + self.window]
        grayscale = [self._load_image(frame.rgb_path, "L") / 255.0 for frame in selected]
        events = torch.zeros(self.window, 2, *self.image_size, dtype=torch.float32)
        for time in range(1, self.window):
            delta = torch.log(grayscale[time].clamp_min(1e-3)) - torch.log(grayscale[time - 1].clamp_min(1e-3))
            events[time, 0] = (delta / self.event_threshold).clamp(0.0, 1.0)
            events[time, 1] = (-delta / self.event_threshold).clamp(0.0, 1.0)

        poses: Optional[Tensor] = None
        if all(frame.pose is not None for frame in selected):
            reference = _inverse_pose(selected[0].pose)  # type: ignore[arg-type]
            poses = torch.stack([reference @ frame.pose for frame in selected])  # type: ignore[operator]

        depths = [self._load_depth(frame.depth_path) for frame in selected]
        if all(depth is not None for depth in depths):
            depth = torch.stack([item for item in depths if item is not None])
            depth_mask = (depth > 0).float()
        else:
            depth = None
            depth_mask = None
        return TrainBatch(events=events, poses=poses, depth=depth, depth_mask=depth_mask)


def collate_tum_batches(samples: Sequence[TrainBatch]) -> TrainBatch:
    """Collate fixed-size TUM windows while preserving optional labels."""

    if not samples:
        raise ValueError("cannot collate an empty batch")
    def stack_optional(name: str) -> Optional[Tensor]:
        values = [getattr(sample, name) for sample in samples]
        return torch.stack(values) if all(value is not None for value in values) else None
    return TrainBatch(
        events=torch.stack([sample.events for sample in samples]),
        poses=stack_optional("poses"),
        depth=stack_optional("depth"),
        depth_mask=stack_optional("depth_mask"),
    )


__all__ = ["TUMFrame", "TUMRGBDWindowDataset", "collate_tum_batches"]

