from __future__ import annotations

import torch
from torch import nn


class ExpertMLP(nn.Module):
    interface_version = "expert-v1"

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int | None = None):
        super().__init__()
        output_dim = output_dim or input_dim
        self.up = nn.Linear(input_dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.gelu(self.up(x)))

    def interface_signature(self) -> dict:
        return {"version": self.interface_version, "input_dim": self.up.in_features, "output_dim": self.down.out_features}


class SwiGLUExpert(nn.Module):
    interface_version = "expert-v1"

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int | None = None):
        super().__init__()
        output_dim = output_dim or input_dim
        self.value = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.value(x))

    def interface_signature(self) -> dict:
        return {"version": self.interface_version, "input_dim": self.value.in_features, "output_dim": self.down.out_features}


class ModularExperts(nn.Module):
    def __init__(self, bus_dim: int, hidden_dim: int, n_experts: int):
        super().__init__()
        self.router = nn.Linear(bus_dim, n_experts)
        self.experts = nn.ModuleList(
            [ExpertMLP(bus_dim, hidden_dim, bus_dim) for _ in range(n_experts)]
        )
        self.n_experts = n_experts
        self.last_load: torch.Tensor | None = None

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.router(latent), dim=-1)
        outputs = torch.stack([expert(latent) for expert in self.experts], dim=-2)
        mixed = torch.sum(outputs * weights.unsqueeze(-1), dim=-2)
        self.last_load = weights.detach().mean(dim=(0, 1))
        return mixed, weights

    def replace_expert(self, index: int, expert: nn.Module) -> nn.Module:
        if not 0 <= index < self.n_experts:
            raise IndexError(index)
        old = self.experts[index]
        self.experts[index] = expert
        return old
