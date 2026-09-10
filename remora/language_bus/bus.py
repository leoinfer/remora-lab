from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class BusPacket:
    latent: torch.Tensor
    confidence: torch.Tensor
    version: str
    producer: str


class SharedLanguageBus(nn.Module):
    """The shared, versioned semantic interface used by all Remora blocks."""

    def __init__(self, d_model: int, bus_dim: int, version: str = "language-v1"):
        super().__init__()
        self.version = version
        self.to_latent = nn.Linear(d_model, bus_dim)
        self.latent_norm = nn.LayerNorm(bus_dim)
        self.confidence_head = nn.Sequential(nn.Linear(bus_dim, bus_dim // 2), nn.Tanh(), nn.Linear(bus_dim // 2, 1))
        self.from_latent = nn.Linear(bus_dim, d_model)

    def encode(self, x: torch.Tensor, producer: str = "unknown") -> BusPacket:
        z = self.latent_norm(self.to_latent(x))
        confidence = torch.sigmoid(self.confidence_head(z))
        return BusPacket(z, confidence, self.version, producer)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.from_latent(latent)

    def forward(self, x: torch.Tensor, producer: str = "shared-bus") -> BusPacket:
        return self.encode(x, producer=producer)

    def interface_signature(self) -> dict:
        return {
            "version": self.version,
            "input": ["batch", "time", self.to_latent.in_features],
            "latent": ["batch", "time", self.to_latent.out_features],
            "output": ["batch", "time", self.from_latent.out_features],
            "metadata": ["confidence", "producer", "version"],
        }


class DirectLanguageBus(nn.Module):
    """A parameter-free direct coupling ablation with the same tensor port.

    This is deliberately austere: it exposes the first ``bus_dim`` channels
    of the block state and pads them back to ``d_model`` on decode.  There is
    no learned semantic projection, normalization, confidence head, or
    trainable bus state.  Keeping the packet/geometry contract intact lets an
    experiment test whether the learned shared bus contributes beyond merely
    transporting a fixed-width hidden slice.
    """

    def __init__(self, d_model: int, bus_dim: int, version: str = "language-direct-v1"):
        super().__init__()
        if bus_dim > d_model:
            raise ValueError("direct bus requires bus_dim <= d_model")
        self.d_model = int(d_model)
        self.bus_dim = int(bus_dim)
        self.version = version

    def encode(self, x: torch.Tensor, producer: str = "direct-bus") -> BusPacket:
        latent = x[..., : self.bus_dim]
        confidence = torch.full(
            (*latent.shape[:-1], 1),
            0.5,
            device=latent.device,
            dtype=latent.dtype,
        )
        return BusPacket(latent, confidence, self.version, producer)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.size(-1) != self.bus_dim:
            raise ValueError(f"expected latent width {self.bus_dim}, got {latent.size(-1)}")
        if self.bus_dim == self.d_model:
            return latent
        return torch.cat(
            (latent, torch.zeros(*latent.shape[:-1], self.d_model - self.bus_dim, device=latent.device, dtype=latent.dtype)),
            dim=-1,
        )

    def forward(self, x: torch.Tensor, producer: str = "direct-bus") -> BusPacket:
        return self.encode(x, producer=producer)

    def interface_signature(self) -> dict:
        return {
            "version": self.version,
            "input": ["batch", "time", self.d_model],
            "latent": ["batch", "time", self.bus_dim],
            "output": ["batch", "time", self.d_model],
            "metadata": ["confidence", "producer", "version"],
            "trainable_parameters": 0,
            "coupling": "fixed_channel_slice_and_zero_pad",
        }
