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
