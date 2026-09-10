from __future__ import annotations

"""Remora-facing wrapped graft for the extracted Qwen shared expert."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .payload import QwenSharedExpertOrgan


class LowRankPort(nn.Module):
    """A narrow trainable port whose rank is explicit in the experiment ledger."""

    def __init__(self, input_dim: int, output_dim: int, rank: int = 8):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.rank = int(rank)
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, output_dim, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        # A donor-width port followed by a second port otherwise attenuates a
        # foreign organ twice and makes real/shuffled cores numerically
        # indistinguishable at the Remora residual scale.  This is an
        # initialization scale, not an extra learned pathway; it is recorded
        # in the graft experiment so the ablation remains auditable.
        nn.init.normal_(self.up.weight, mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))

    @property
    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))


class QwenSharedExpertGraft(nn.Module):
    """Wrap the frozen Qwen organ behind low-rank Remora bus ports.

    The Qwen buffers remain the computation core.  Only the two ports are
    trainable by default; the caller can use ``trainable_port_parameters`` to
    make that scope explicit and auditable.
    """

    interface_version = "expert-v1/qwen-shared-expert-organ-v1"

    def __init__(
        self,
        donor_tensors: Mapping[str, torch.Tensor],
        *,
        bus_dim: int,
        donor_dim: int = 2560,
        rank: int = 8,
        donor_variant: str = "actual",
        variant_seed: int = 0,
    ):
        super().__init__()
        if bus_dim <= 0 or donor_dim <= 0:
            raise ValueError("bus_dim and donor_dim must be positive")
        if donor_variant not in {"actual", "shuffled", "random", "zero"}:
            raise ValueError(donor_variant)
        variant = make_donor_variant(donor_tensors, donor_variant, seed=variant_seed)
        self.donor_variant = donor_variant
        self.donor_variant_seed = int(variant_seed)
        self.bus_dim = int(bus_dim)
        self.donor_dim = int(donor_dim)
        self.rank = int(rank)
        self.input_port = LowRankPort(bus_dim, donor_dim, rank)
        self.output_port = LowRankPort(donor_dim, bus_dim, rank)
        self.organ = QwenSharedExpertOrgan(variant)
        self.donor_payload_parameter_count = int(self.organ.donor_parameter_count)
        self.port_parameter_count = int(sum(parameter.numel() for parameter in self.port_parameters()))
        for parameter in self.organ.parameters():
            parameter.requires_grad = False

    def port_parameters(self):
        yield from self.input_port.parameters()
        yield from self.output_port.parameters()

    def trainable_port_parameters(self) -> list[nn.Parameter]:
        return list(self.port_parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.bus_dim:
            raise ValueError(f"expected Remora bus width {self.bus_dim}, got {x.shape[-1]}")
        donor_input = self.input_port(x)
        donor_output = self.organ(donor_input)
        return self.output_port(donor_output)

    def interface_signature(self) -> dict[str, Any]:
        return {
            "version": self.interface_version,
            "input": ["batch", "time", self.bus_dim],
            "donor_core": ["batch", "time", self.donor_dim],
            "output": ["batch", "time", self.bus_dim],
            "port_rank": self.rank,
            "donor_core_frozen": True,
            "donor_variant": self.donor_variant,
        }

    def donor_buffers(self) -> dict[str, torch.Tensor]:
        return {name: buffer for name, buffer in self.organ.named_buffers()}


def make_donor_variant(
    tensors: Mapping[str, torch.Tensor], variant: str, *, seed: int = 0
) -> dict[str, torch.Tensor]:
    """Create matched controls without changing the extracted bundle."""

    if variant not in {"actual", "shuffled", "random", "zero"}:
        raise ValueError(variant)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result: dict[str, torch.Tensor] = {}
    for name in sorted(tensors):
        source = tensors[name].detach().cpu().contiguous()
        if variant == "actual":
            value = source.clone()
        elif variant == "zero":
            value = torch.zeros_like(source)
        elif variant == "random":
            std = float(source.float().std(unbiased=False))
            value = torch.randn(source.shape, generator=generator, dtype=torch.float32) * max(std, 1e-6)
            value = value.to(source.dtype)
        else:
            flat = source.reshape(-1)
            permutation = torch.randperm(flat.numel(), generator=generator)
            value = flat[permutation].reshape(source.shape).clone()
        result[name] = value
    return result
