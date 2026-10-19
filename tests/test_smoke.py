import torch

from snn_slam import AllSNN_SLAM, events_to_voxel_grid, se3_exp, slam_loss, so3_exp


def test_event_encoder_handles_empty_and_boundaries():
    empty = events_to_voxel_grid(torch.empty(0, 4), bins=3, height=8, width=8)
    assert empty.shape == (3, 2, 8, 8)
    events = torch.tensor([[0, 0, 0.0, -1], [7, 7, 1.0, 1], [99, 99, 0.5, 1]])
    voxels = events_to_voxel_grid(events, bins=3, height=8, width=8)
    assert voxels.shape == (3, 2, 8, 8)
    assert torch.isfinite(voxels).all()


def test_all_snn_forward_and_gradient():
    model = AllSNN_SLAM(feature_channels=32, map_size=16)
    events = torch.rand(1, 3, 2, 24, 24)
    output = model(events)
    assert output.twists.shape == (1, 3, 6)
    assert output.poses.shape == (1, 3, 4, 4)
    assert output.depth.shape == (1, 3, 1, 24, 24)
    assert output.occupancy.shape == (1, 3, 1, 16, 16)
    losses = slam_loss(output, target_depth=torch.ones_like(output.depth))
    losses.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_streaming_step_matches_interface():
    model = AllSNN_SLAM(feature_channels=32, map_size=16)
    model.reset_state()
    output = model.step(torch.rand(2, 2, 24, 24))
    assert output.twists.shape == (2, 1, 6)


def test_se3_exp_is_finite_at_identity():
    rotation = torch.zeros(3, requires_grad=True)
    transform_twist = torch.zeros(6, requires_grad=True)
    (so3_exp(rotation).sum() + se3_exp(transform_twist).sum()).backward()
    assert torch.isfinite(rotation.grad).all()
    assert torch.isfinite(transform_twist.grad).all()

