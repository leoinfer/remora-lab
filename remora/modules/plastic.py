from __future__ import annotations

import torch
from torch import nn


class FastPlasticAdapter(nn.Module):
    """A low-rank parameter island initialized as a no-op."""

    interface_version = "plastic-adapter-v1"

    def __init__(self, d_model: int, adapter_dim: int, scale: float = 1.0):
        super().__init__()
        self.down = nn.Linear(d_model, adapter_dim, bias=False)
        self.up = nn.Linear(adapter_dim, d_model, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.up(torch.tanh(self.down(x)))

    def interface_signature(self) -> dict:
        return {"version": self.interface_version, "input_dim": self.down.in_features, "output_dim": self.up.out_features}
