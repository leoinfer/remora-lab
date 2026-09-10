from __future__ import annotations

import torch
from torch import nn


def _compose_affine(
    left: tuple[torch.Tensor, torch.Tensor],
    right: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose elementwise affine state updates.

    An update is represented as ``s -> decay * s + injection``.  The
    composition is associative, which permits a tree scan over the token
    dimension.
    """

    decay_left, injection_left = left
    decay_right, injection_right = right
    return (
        decay_right * decay_left,
        decay_right * injection_left + injection_right,
    )


def _reference_affine_scan(
    decay: torch.Tensor, injection: torch.Tensor, state: torch.Tensor
) -> torch.Tensor:
    states = []
    current = state
    for index in range(decay.size(1)):
        current = decay[:, index] * current + injection[:, index]
        states.append(current)
    return torch.stack(states, dim=1)


def _associative_affine_scan(
    decay: torch.Tensor, injection: torch.Tensor, state: torch.Tensor
) -> torch.Tensor | None:
    try:
        from torch._higher_order_ops import associative_scan
    except ImportError:  # pragma: no cover - depends on torch version
        return None
    try:
        prefix_decay, prefix_injection = associative_scan(
            _compose_affine,
            (decay, injection),
            dim=1,
        )
    except (RuntimeError, NotImplementedError):
        # Keep older/driver-specific installations usable.  The caller still
        # gets the exact recurrence, just without the tree-scan speedup.
        return None
    return prefix_decay * state[:, None, :] + prefix_injection


def _reverse_adjoint_scan(decay: torch.Tensor, grad_states: torch.Tensor) -> torch.Tensor:
    """Reverse recurrence for the custom affine-scan backward pass."""

    batch, time, _ = decay.shape
    transition = torch.zeros_like(decay)
    if time > 1:
        # At token t, the future adjoint is multiplied by decay[t + 1].
        transition[:, :-1, :] = decay[:, 1:, :]
    reversed_states = _associative_affine_scan(
        transition.flip(1),
        grad_states.flip(1),
        torch.zeros(batch, decay.size(-1), device=decay.device, dtype=decay.dtype),
    )
    if reversed_states is not None:
        return reversed_states.flip(1)

    adjoint = torch.zeros_like(grad_states)
    future = torch.zeros_like(grad_states[:, 0])
    for index in range(time - 1, -1, -1):
        future = grad_states[:, index] + future * transition[:, index]
        adjoint[:, index] = future
    return adjoint


class _DifferentiableAffineScan(torch.autograd.Function):
    """CUDA tree scan with an explicit backward for the affine recurrence.

    PyTorch's higher-order associative scan currently rejects gradients for
    lifted/trainable inputs.  Wrapping its inference forward and supplying the
    elementary reverse recurrence preserves the optimization during training
    while keeping gradients mathematically equivalent to the reference loop.
    """

    @staticmethod
    def forward(ctx, decay: torch.Tensor, injection: torch.Tensor, state: torch.Tensor):
        with torch.no_grad():
            states = _associative_affine_scan(decay, injection, state)
            if states is None:
                states = _reference_affine_scan(decay, injection, state)
        ctx.save_for_backward(decay, states, state)
        return states

    @staticmethod
    def backward(ctx, grad_states: torch.Tensor):
        decay, states, initial_state = ctx.saved_tensors
        with torch.no_grad():
            previous = torch.cat((initial_state[:, None, :], states[:, :-1, :]), dim=1)
            adjoint = _reverse_adjoint_scan(decay, grad_states)
            grad_decay = adjoint * previous
            grad_injection = adjoint
            grad_initial_state = adjoint[:, 0, :] * decay[:, 0, :]
        return grad_decay, grad_injection, grad_initial_state


class GatedDeltaState(nn.Module):
    """A causal recurrent state path with bounded gated delta updates."""

    def __init__(self, d_model: int, state_dim: int, parallel_scan: bool = True):
        super().__init__()
        self.state_dim = state_dim
        self.parallel_scan = parallel_scan
        self.key = nn.Linear(d_model, state_dim)
        self.value = nn.Linear(d_model, state_dim)
        self.gate = nn.Linear(d_model, state_dim)
        self.query = nn.Linear(d_model, state_dim)
        self.out = nn.Linear(state_dim, d_model)

    _compose_affine = staticmethod(_compose_affine)

    def forward(
        self, x: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        key = torch.sigmoid(self.key(x))
        value = torch.tanh(self.value(x))
        gate = torch.sigmoid(self.gate(x))
        query = torch.tanh(self.query(x))
        if state is None:
            state = torch.zeros(b, self.state_dim, device=x.device, dtype=x.dtype)

        update = gate * key
        decay = 1.0 - update
        # The current PyTorch associative-scan implementation does not
        # support autograd through lifted/trainable inputs on this runtime.
        # Keep the fast tree scan for inference, but use the exact reference
        # recurrence whenever gradients are enabled so training remains
        # genuinely differentiable instead of failing during compilation.
        if self.parallel_scan and x.is_cuda:
            injection = update * value
            if torch.is_grad_enabled():
                states = _DifferentiableAffineScan.apply(decay, injection, state)
            else:
                # PyTorch's pointwise associative scan is CUDA/XPU-only. CPU
                # and older installations use the exact reference recurrence.
                states = _associative_affine_scan(decay, injection, state)
                if states is None:
                    states = _reference_affine_scan(decay, injection, state)
            return self.out(states * query), states[:, -1, :]

        states = _reference_affine_scan(decay, update * value, state)
        return self.out(states * query), states[:, -1, :]
