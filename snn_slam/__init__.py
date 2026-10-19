"""End-to-end all-SNN event SLAM.

The public API intentionally stays small::

    from snn_slam import AllSNN_SLAM, EventVoxelEncoder, slam_loss

``AllSNN_SLAM`` consumes event voxel tensors and returns differentiable pose,
depth and occupancy-map predictions.
"""

from .encoding import EventVoxelEncoder, event_rate, events_to_frames, events_to_voxel_grid
from .losses import LossBreakdown, slam_loss
from .models import AllSNN_SLAM, DifferentiableOccupancyMap, SNN_SLAM, SLAMOutput
from .se3 import integrate_twists, inverse, pose_geodesic_loss, se3_exp, so3_exp
from .snn import LIFNeuron, SNNBase, SpikingConv2d, SpikingLinear, mean_rate, spike_function

__all__ = [
    "AllSNN_SLAM",
    "SNN_SLAM",
    "SLAMOutput",
    "DifferentiableOccupancyMap",
    "EventVoxelEncoder",
    "events_to_voxel_grid",
    "events_to_frames",
    "event_rate",
    "slam_loss",
    "LossBreakdown",
    "LIFNeuron",
    "SpikingConv2d",
    "SpikingLinear",
    "SNNBase",
    "spike_function",
    "mean_rate",
    "se3_exp",
    "so3_exp",
    "integrate_twists",
    "inverse",
    "pose_geodesic_loss",
]

