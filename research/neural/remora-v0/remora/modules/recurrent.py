from __future__ import annotations

import torch
from torch import nn


class GatedDeltaState(nn.Module):
    """A causal recurrent state path with bounded gated delta updates."""

    def __init__(self, d_model: int, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.key = nn.Linear(d_model, state_dim)
        self.value = nn.Linear(d_model, state_dim)
        self.gate = nn.Linear(d_model, state_dim)
        self.query = nn.Linear(d_model, state_dim)
        self.out = nn.Linear(state_dim, d_model)

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
        outputs = []
        for i in range(t):
            state = state + gate[:, i] * key[:, i] * (value[:, i] - state)
            outputs.append(state * query[:, i])
        return self.out(torch.stack(outputs, dim=1)), state
