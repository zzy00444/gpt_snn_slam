"""End-to-end event-to-SLAM spiking model.

The model has one computation graph from event voxels to odometry, inverse
depth and a differentiable local occupancy map.  Every learned hidden block is
an LIF block; the only analogue operations are the explicit sensor encoding,
SE(3) integration and output decoders required by a physical SLAM interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .se3 import integrate_twists
from .snn import LIFNeuron, SNNBase, SpikingConv2d, SpikingLinear


@dataclass
class SLAMOutput:
    """Outputs produced by :class:`AllSNN_SLAM`.

    Shapes are ``twists [B,T,6]``, ``poses [B,T,4,4]``, ``depth [B,T,1,H,W]``
    and ``occupancy [B,T,1,M,M]``.  ``spikes`` contains the final encoder
    feature spikes and can be used to calculate energy/sparsity metrics.
    """

    twists: Tensor
    poses: Tensor
    depth: Tensor
    occupancy: Tensor
    spikes: Optional[Tensor] = None

    def as_dict(self) -> Dict[str, Tensor]:
        result = {
            "twists": self.twists,
            "poses": self.poses,
            "depth": self.depth,
            "occupancy": self.occupancy,
        }
        if self.spikes is not None:
            result["spikes"] = self.spikes
        return result


class SpikingBackbone(nn.Module):
    """Three-layer spatial feature extractor with explicit LIF state."""

    def __init__(self, in_channels: int, feature_channels: int = 64) -> None:
        super().__init__()
        c1 = max(16, feature_channels // 4)
        c2 = max(24, feature_channels // 2)
        self.layer1 = SpikingConv2d(in_channels, c1, kernel_size=5, stride=2)
        self.layer2 = SpikingConv2d(c1, c2, kernel_size=3, stride=2)
        self.layer3 = SpikingConv2d(c2, feature_channels, kernel_size=3, stride=1)

    def forward_step(
        self, x: Tensor, states: Optional[List[Optional[Tensor]]] = None
    ) -> Tuple[Tensor, List[Tensor]]:
        if states is None:
            states = [None, None, None]
        x, s1 = self.layer1(x, states[0])
        x, s2 = self.layer2(x, states[1])
        x, s3 = self.layer3(x, states[2])
        return x, [s1, s2, s3]


class DifferentiableOccupancyMap:
    """A tiny local map updated with differentiable spatial writes.

    The map is not a detached numpy buffer: gradients flow from an occupancy
    or map-consistency loss through the pose and event feature spikes.  The
    map uses the XY translation of the current pose to place the local write;
    extending this class with camera intrinsics/depth projection is possible
    without changing the SNN front-end API.
    """

    def __init__(self, size: int, extent_m: float = 20.0, decay: float = 0.97) -> None:
        if size <= 0 or extent_m <= 0:
            raise ValueError("map size and extent must be positive")
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.size = int(size)
        self.extent_m = float(extent_m)
        self.decay = float(decay)

    def initial(self, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(batch, 1, self.size, self.size, device=device, dtype=dtype)

    def _shift(self, value: Tensor, translation: Tensor) -> Tensor:
        # affine_grid uses normalized coordinates in [-1, 1]. XY pose is in
        # metres, therefore one metre corresponds to 2/extent_m.
        dx = translation[:, 0] * (2.0 / self.extent_m)
        dy = translation[:, 1] * (2.0 / self.extent_m)
        theta = torch.zeros(value.shape[0], 2, 3, device=value.device, dtype=value.dtype)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, 0, 2] = -dx
        theta[:, 1, 2] = -dy
        grid = F.affine_grid(theta, value.shape, align_corners=False)
        return F.grid_sample(value, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def update(self, current: Tensor, write: Tensor, translation: Tensor) -> Tensor:
        if write.shape[-2:] != current.shape[-2:]:
            write = F.interpolate(write, size=current.shape[-2:], mode="bilinear", align_corners=False)
        shifted = self._shift(write, translation)
        return (current * self.decay + shifted * (1.0 - self.decay)).clamp(0.0, 1.0)


class AllSNN_SLAM(SNNBase):
    """End-to-end all-SNN visual/event SLAM network.

    ``forward`` accepts ``[B,T,C,H,W]`` event voxels or a single frame
    ``[B,C,H,W]``.  The network can process a complete sequence with
    ``reset=True`` (the default) or be called frame by frame with
    ``reset=False``.  For stateful streaming use :meth:`step` and
    :meth:`reset_state`.
    """

    def __init__(
        self,
        in_channels: int = 2,
        feature_channels: int = 64,
        map_size: int = 64,
        map_extent_m: float = 20.0,
        depth_min_m: float = 0.1,
        depth_max_m: float = 20.0,
        pose_scale: Tuple[float, float] = (0.25, 0.25),
    ) -> None:
        super().__init__()
        if depth_min_m <= 0 or depth_max_m <= depth_min_m:
            raise ValueError("depth range must satisfy 0 < min < max")
        self.in_channels = int(in_channels)
        self.feature_channels = int(feature_channels)
        self.depth_min_m = float(depth_min_m)
        self.depth_range_m = float(depth_max_m - depth_min_m)
        self.pose_scale = (float(pose_scale[0]), float(pose_scale[1]))
        self.backbone = SpikingBackbone(in_channels, feature_channels)
        # A pooled spike vector is fed only to LIF synapses.
        hidden = max(32, feature_channels)
        self.pose_hidden = SpikingLinear(feature_channels, hidden)
        self.pose_output = SpikingLinear(hidden, 6)
        # Spatial inverse-depth readout remains fully spiking at the pixel level.
        self.depth_hidden = SpikingConv2d(feature_channels, max(16, feature_channels // 2), 3)
        self.depth_output = SpikingConv2d(max(16, feature_channels // 2), 1, 1, padding=0)
        self.map = DifferentiableOccupancyMap(map_size, map_extent_m)
        self._stream_states: Dict[str, object] = {}

    def reset_state(self) -> None:
        self._stream_states = {}

    def detach_state(self) -> None:
        for key, state in list(self._stream_states.items()):
            if isinstance(state, list):
                self._stream_states[key] = [x.detach() if isinstance(x, Tensor) else x for x in state]
            elif isinstance(state, Tensor):
                self._stream_states[key] = state.detach()

    @staticmethod
    def _normalise_input(events: Tensor, channels: int) -> Tensor:
        if events.ndim == 4:
            events = events.unsqueeze(1)
        if events.ndim != 5:
            raise ValueError("events must have shape [B,T,C,H,W] or [B,C,H,W]")
        if events.shape[2] != channels:
            if channels == 1 and events.shape[2] == 2:
                events = events.sum(dim=2, keepdim=True)
            else:
                raise ValueError(f"expected {channels} event channels, got {events.shape[2]}")
        return events.float()

    def _step(
        self,
        frame: Tensor,
        states: Dict[str, object],
        map_state: Tensor,
        pose: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, object], Tensor]:
        features, backbone_state = self.backbone.forward_step(frame, states.get("backbone"))
        pooled = features.mean(dim=(-2, -1))
        pose_hidden, pose_hidden_state = self.pose_hidden(pooled, states.get("pose_hidden"))
        pose_spikes, pose_state = self.pose_output(pose_hidden, states.get("pose_output"))
        # Read the membrane state as a bounded analogue velocity. It is the
        # explicit decoder of an otherwise binary spike computation.
        pose_mem = pose_state
        translation_scale, rotation_scale = self.pose_scale
        scales = pose_mem.new_tensor([translation_scale] * 3 + [rotation_scale] * 3)
        twist = (pose_mem / (1.0 + pose_mem.abs())) * scales
        increment = integrate_twists(twist.unsqueeze(1))[:, 0]
        next_pose = pose @ increment

        depth_hidden, depth_hidden_state = self.depth_hidden(features, states.get("depth_hidden"))
        depth_spikes, depth_state = self.depth_output(depth_hidden, states.get("depth_output"))
        # A spike rate of 0..1 maps directly to a metric depth range.
        depth_spikes = F.interpolate(
            depth_spikes, size=frame.shape[-2:], mode="bilinear", align_corners=False
        )
        depth = self.depth_min_m + depth_spikes * self.depth_range_m
        map_state = self.map.update(map_state, depth_spikes, next_pose[:, :3, 3])
        next_states = {
            "backbone": backbone_state,
            "pose_hidden": pose_hidden_state,
            "pose_output": pose_state,
            "depth_hidden": depth_hidden_state,
            "depth_output": depth_state,
        }
        return twist, next_pose, depth, map_state, next_states, features

    def forward(self, events: Tensor, reset: bool = True) -> SLAMOutput:
        events = self._normalise_input(events, self.in_channels)
        batch, steps = events.shape[:2]
        if reset:
            states: Dict[str, object] = {}
            pose = torch.eye(4, dtype=events.dtype, device=events.device).expand(batch, 4, 4).clone()
            map_state = self.map.initial(batch, events.device, events.dtype)
        else:
            states = self._stream_states.get("states", {})
            pose = self._stream_states.get("pose")
            map_state = self._stream_states.get("map")
            if (
                not isinstance(pose, Tensor)
                or not isinstance(map_state, Tensor)
                or pose.shape[0] != batch
                or map_state.shape[0] != batch
                or pose.device != events.device
                or pose.dtype != events.dtype
                or map_state.dtype != events.dtype
            ):
                states = {}
                pose = torch.eye(4, dtype=events.dtype, device=events.device).expand(batch, 4, 4).clone()
                map_state = self.map.initial(batch, events.device, events.dtype)
        twists: List[Tensor] = []
        poses: List[Tensor] = []
        depths: List[Tensor] = []
        maps: List[Tensor] = []
        feature_spikes: List[Tensor] = []
        for frame in events.unbind(1):
            twist, pose, depth, map_state, states, features = self._step(
                frame, states, map_state, pose
            )
            twists.append(twist)
            poses.append(pose)
            depths.append(depth)
            maps.append(map_state)
            feature_spikes.append(features)
        if not twists:
            raise ValueError("events sequence must contain at least one timestep")
        if not reset:
            self._stream_states = {"states": states, "pose": pose, "map": map_state}
        return SLAMOutput(
            twists=torch.stack(twists, dim=1),
            poses=torch.stack(poses, dim=1),
            depth=torch.stack(depths, dim=1),
            occupancy=torch.stack(maps, dim=1),
            spikes=torch.stack(feature_spikes, dim=1),
        )

    def step(self, frame: Tensor) -> SLAMOutput:
        """Process one ``[B,C,H,W]`` frame while retaining SNN/map state."""

        return self.forward(frame.unsqueeze(1) if frame.ndim == 4 else frame, reset=False)


SNN_SLAM = AllSNN_SLAM

__all__ = ["SLAMOutput", "AllSNN_SLAM", "SNN_SLAM", "DifferentiableOccupancyMap"]

