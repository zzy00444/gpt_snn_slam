"""Small, dependency-free spiking neural-network building blocks.

The implementation deliberately keeps membrane states explicit.  This makes a
network usable both in truncated BPTT training and in a streaming robot
process, where :meth:`SNNBase.reset_state` can be called at a frame boundary.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor, nn


class _SurrogateSpike(torch.autograd.Function):
    """Binary spike in the forward pass with a smooth backward pass.

    The rectangular forward function is what gives the network event-like
    activations.  The fast-sigmoid derivative is stable for both float32 and
    mixed-precision training and does not require a third-party SNN library.
    """

    @staticmethod
    def forward(ctx, voltage: Tensor, slope: float) -> Tensor:
        ctx.save_for_backward(voltage)
        ctx.slope = float(slope)
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        (voltage,) = ctx.saved_tensors
        # d sigmoid(slope*x) / dx, written in a bounded form.
        derivative = ctx.slope / (2.0 + ctx.slope * voltage.abs()).square()
        return grad_output * derivative, None


def spike_function(voltage: Tensor, slope: float = 10.0) -> Tensor:
    """Return a hard spike while retaining a surrogate gradient."""

    return _SurrogateSpike.apply(voltage, slope)


class LIFNeuron(nn.Module):
    """Leaky integrate-and-fire neuron with soft reset.

    ``state`` is passed in and returned rather than hidden in the module.  A
    caller can therefore process a sequence in chunks without accidentally
    retaining a graph from a previous optimisation step.
    """

    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_slope: float = 10.0,
        learnable_decay: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must be in (0, 1)")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        # Parameterise beta through a logit so it remains in (0, 1).
        beta_logit = torch.logit(torch.tensor(float(beta)))
        if learnable_decay:
            self.beta_logit = nn.Parameter(beta_logit)
        else:
            self.register_buffer("beta_logit", beta_logit)
        self.threshold = float(threshold)
        self.surrogate_slope = float(surrogate_slope)

    @property
    def beta(self) -> Tensor:
        return self.beta_logit.sigmoid()

    def initial_state(self, reference: Tensor) -> Tensor:
        return torch.zeros_like(reference)

    def forward(
        self, current: Tensor, state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        if state is None:
            state = self.initial_state(current)
        voltage = self.beta.to(current.dtype) * state + current
        spikes = spike_function(voltage - self.threshold, self.surrogate_slope)
        # Soft reset avoids a discontinuity in the membrane dynamics while
        # preserving the binary output consumed by downstream layers.
        next_state = voltage - spikes * self.threshold
        return spikes, next_state


class SpikingConv2d(nn.Module):
    """Convolution followed by a LIF neuron."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        beta: float = 0.9,
        groups: int = 1,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=True,
        )
        # GroupNorm is affine synaptic normalisation; unlike BatchNorm it is
        # well-defined for one-frame streaming inference and tiny batches.
        groups_norm = max(1, min(8, out_channels))
        while out_channels % groups_norm != 0:
            groups_norm -= 1
        self.norm = nn.GroupNorm(groups_norm, out_channels)
        self.neuron = LIFNeuron(beta=beta)

    def forward(
        self, x: Tensor, state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        return self.neuron(self.norm(self.conv(x)), state)


class SpikingLinear(nn.Module):
    """Linear synapse followed by a LIF neuron."""

    def __init__(self, in_features: int, out_features: int, beta: float = 0.9) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.neuron = LIFNeuron(beta=beta)

    def forward(
        self, x: Tensor, state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        return self.neuron(self.linear(x), state)


class SNNBase(nn.Module):
    """Mixin for stateful SNN modules.

    State dictionaries use ordinary attribute names and are intentionally kept
    out of ``state_dict``.  ``reset_state`` is safe to call between sequences,
    while ``detach_state`` is useful for truncated BPTT.
    """

    def reset_state(self) -> None:
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "_stream_state"):
                module._stream_state = None

    def detach_state(self) -> None:
        for module in self.modules():
            state = getattr(module, "_stream_state", None)
            if isinstance(state, Tensor):
                module._stream_state = state.detach()


def run_lif_sequence(
    neuron: LIFNeuron,
    currents: Tensor,
    state: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Apply one LIF neuron to ``[T, ...]`` currents.

    Returns ``(spikes, final_state)`` with spikes retaining the time axis.
    """

    if currents.ndim < 1:
        raise ValueError("currents must have a time dimension")
    outputs = []
    for current in currents.unbind(0):
        spike, state = neuron(current, state)
        outputs.append(spike)
    if not outputs:
        return currents, state if state is not None else torch.zeros_like(currents)
    return torch.stack(outputs, dim=0), state


def mean_rate(spikes: Tensor, dim: int = 0) -> Tensor:
    """Decode a spike train into a rate in [0, 1]."""

    if not spikes.is_floating_point():
        spikes = spikes.float()
    return spikes.mean(dim=dim)


def count_spikes(modules: Iterable[nn.Module]) -> int:
    """Count LIF neurons, useful for reporting model size and sparsity."""

    return sum(isinstance(module, LIFNeuron) for module in modules)

