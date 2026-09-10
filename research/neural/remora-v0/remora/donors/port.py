from __future__ import annotations

"""Differentiable, versioned ports for importing donor representations.

The adapter is intentionally separate from any donor implementation.  A
Transformers model, llama.cpp server, or another local runtime can provide a
compact hidden-state batch; Remora owns only the adapter and the resulting
module lineage.  Donor output is detached by the caller unless a deliberate
joint-training experiment is authorized.
"""

import torch
from torch import nn

from ..language_bus.bus import BusPacket


class TeacherPortAdapter(nn.Module):
    """Map a donor hidden representation into the common Remora bus."""

    interface_version = "donor-port-v1"

    def __init__(self, teacher_dim: int, bus_dim: int, bottleneck: int | None = None):
        super().__init__()
        hidden = bottleneck or max(bus_dim, min(teacher_dim, bus_dim * 2))
        self.teacher_dim = teacher_dim
        self.bus_dim = bus_dim
        self.down = nn.Linear(teacher_dim, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.up = nn.Linear(hidden, bus_dim)
        self.confidence = nn.Sequential(nn.Linear(bus_dim, max(1, bus_dim // 2)), nn.Tanh(), nn.Linear(max(1, bus_dim // 2), 1))
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, teacher_features: torch.Tensor, producer: str = "donor") -> BusPacket:
        if teacher_features.size(-1) != self.teacher_dim:
            raise ValueError(f"expected donor feature width {self.teacher_dim}, got {teacher_features.size(-1)}")
        # Donor runtimes commonly expose BF16/FP16 activations while the
        # trainable port is kept in FP32 for stable local adaptation. Make the
        # conversion an explicit interface operation rather than relying on a
        # backend-specific matmul cast.
        if teacher_features.dtype != self.down.weight.dtype:
            teacher_features = teacher_features.to(dtype=self.down.weight.dtype)
        latent = self.up(torch.tanh(self.norm(self.down(teacher_features))))
        latent = torch.sigmoid(self.gate) * latent
        confidence = torch.sigmoid(self.confidence(latent))
        return BusPacket(latent=latent, confidence=confidence, version=self.interface_version, producer=producer)

    def interface_signature(self) -> dict[str, object]:
        return {
            "version": self.interface_version,
            "input": ["batch", "time", self.teacher_dim],
            "latent": ["batch", "time", self.bus_dim],
            "metadata": ["confidence", "producer", "version"],
            "gradient_policy": "donor frozen by default; adapter trainable",
        }


def teacher_logit_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """KL loss for donors with an aligned tokenizer and vocabulary.

    It intentionally rejects shape mismatches.  Qwen3.8-Flash-Next and the
    character-token Remora-v0 do not satisfy this contract, so they must use
    response or hidden-state distillation rather than silently comparing
    unrelated vocabulary coordinates.
    """

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("logit distillation requires the same token positions and vocabulary shape")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    t = float(temperature)
    return nn.functional.kl_div(
        nn.functional.log_softmax(student_logits / t, dim=-1),
        nn.functional.softmax(teacher_logits.detach() / t, dim=-1),
        reduction="batchmean",
    ) * (t * t)
