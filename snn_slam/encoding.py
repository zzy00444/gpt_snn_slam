"""Event-camera encoders used by the all-SNN front end."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import Tensor


def _as_event_tensor(events: Union[Tensor, "object"], device=None) -> Tensor:
    """Convert an ``N x 4`` event array (x, y, timestamp, polarity) to torch."""

    if isinstance(events, Tensor):
        result = events.to(device=device) if device is not None else events
    else:
        # numpy arrays and array-like objects are intentionally accepted
        # without making numpy a runtime dependency.
        result = torch.as_tensor(events, device=device)
    if result.ndim != 2 or result.shape[-1] < 4:
        raise ValueError("events must have shape [N, 4] = (x, y, timestamp, polarity)")
    return result[:, :4]


def events_to_voxel_grid(
    events: Union[Tensor, "object"],
    bins: int,
    height: int,
    width: int,
    time_range: Optional[Tuple[float, float]] = None,
    normalize_time: bool = True,
    polarity_channels: bool = True,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Accumulate asynchronous events into a differentiable voxel tensor.

    Args:
        events: ``[N,4]`` values ``(x, y, timestamp, polarity)``. Polarity can
            be ``{-1, +1}`` or ``{0, 1}``.
        bins: Number of temporal bins.
        height, width: Sensor resolution.
        time_range: Optional ``(start, end)``. If omitted, the first and last
            event timestamps define the range.
        normalize_time: Divide each bin by its absolute event count.
        polarity_channels: Return ``[bins,2,H,W]`` (positive/negative), else
            signed ``[bins,1,H,W]``.

    Empty input is valid and returns an all-zero tensor. Events outside the
    image or requested time range are ignored rather than causing indexing
    errors, which is useful when processing overlapping camera packets.
    """

    if bins <= 0 or height <= 0 or width <= 0:
        raise ValueError("bins, height, and width must be positive")
    raw = _as_event_tensor(events, device=device)
    out_channels = 2 if polarity_channels else 1
    output = torch.zeros((bins, out_channels, height, width), dtype=dtype, device=raw.device)
    if raw.numel() == 0:
        return output
    x = raw[:, 0].long()
    y = raw[:, 1].long()
    timestamp = raw[:, 2].to(torch.float32)
    polarity = raw[:, 3].to(torch.float32)
    if time_range is None:
        start = float(timestamp.min().item())
        end = float(timestamp.max().item())
    else:
        start, end = map(float, time_range)
    duration = max(end - start, 1e-12)
    # Place an event exactly at end in the final bin, avoiding bins == bins.
    position = ((timestamp - start) / duration * bins).floor().long().clamp(0, bins - 1)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if time_range is not None:
        valid = valid & (timestamp >= start) & (timestamp <= end)
    if not bool(valid.any()):
        return output
    x, y, position, polarity = x[valid], y[valid], position[valid], polarity[valid]
    if polarity_channels:
        channel = (polarity >= 0).long()
        # Convert zero polarity to negative to keep the two channels exclusive.
        channel = torch.where(polarity == 0, torch.zeros_like(channel), channel)
        flat = (((position * 2 + channel) * height + y) * width + x).long()
        output.view(-1).scatter_add_(0, flat, torch.ones_like(flat, dtype=dtype))
    else:
        flat = ((position * height + y) * width + x).long()
        signed = torch.where(polarity >= 0, torch.ones_like(polarity), -torch.ones_like(polarity))
        output.view(-1).scatter_add_(0, flat, signed.to(dtype))
    if normalize_time:
        denom = output.abs().sum(dim=(-1, -2), keepdim=True).clamp_min(1.0)
        output = output / denom
    return output


def events_to_frames(
    events: Union[Tensor, "object"],
    frames: int,
    height: int,
    width: int,
    time_range: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> Tensor:
    """Alias returning a temporal frame sequence ``[T,C,H,W]``."""

    return events_to_voxel_grid(
        events,
        bins=frames,
        height=height,
        width=width,
        time_range=time_range,
        **kwargs,
    )


def event_rate(frames: Tensor) -> Tensor:
    """Per-frame event activity used for confidence weighting."""

    if frames.ndim < 3:
        raise ValueError("frames must have shape [T,...]")
    return frames.abs().flatten(1).mean(-1)


class EventVoxelEncoder:
    """Callable configuration object for dataloaders and streaming clients."""

    def __init__(
        self,
        bins: int,
        height: int,
        width: int,
        polarity_channels: bool = True,
        normalize_time: bool = True,
    ) -> None:
        self.bins = bins
        self.height = height
        self.width = width
        self.polarity_channels = polarity_channels
        self.normalize_time = normalize_time

    def __call__(self, events, time_range=None, device=None) -> Tensor:
        return events_to_voxel_grid(
            events,
            self.bins,
            self.height,
            self.width,
            time_range=time_range,
            polarity_channels=self.polarity_channels,
            normalize_time=self.normalize_time,
            device=device,
        )

