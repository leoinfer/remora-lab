from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn

from ..memory import EvidenceStore


class RuleWorldModel(nn.Module):
    """Small structured world model with frozen inherited and local experience paths."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 32, n_classes: int = 2):
        super().__init__()
        self.inherited = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, n_classes)
        )
        self.experience_adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, n_classes)
        )
        self.consolidated = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, n_classes)
        )
        self._zero_module(self.experience_adapter)
        self._zero_module(self.consolidated)

    @staticmethod
    def _zero_module(module: nn.Module) -> None:
        last = [m for m in module.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, features: torch.Tensor, use_experience: bool = True) -> torch.Tensor:
        logits = self.inherited(features)
        if use_experience:
            logits = logits + self.experience_adapter(features)
        return logits

    def consolidated_forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.inherited(features) + self.consolidated(features)

    @staticmethod
    def features(z: torch.Tensor, environment: torch.Tensor, measurement: torch.Tensor | None = None) -> torch.Tensor:
        if measurement is None:
            measurement = torch.zeros_like(z)
        return torch.stack([z, environment, measurement], dim=-1).float()

    @staticmethod
    def predict_label(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)

    def memory_prediction(self, store: EvidenceStore, z: int, environment: int = 0) -> dict:
        return store.separated_posterior({"z": z, "environment": environment})

    def consolidate_from_adapter(self) -> None:
        self.consolidated.load_state_dict(self.experience_adapter.state_dict())


@dataclass
class Hypothesis:
    hypothesis_id: str
    statement: str
    confidence: float
    supporting_evidence: list[str] = field(default_factory=list)
    conflicting_evidence: list[str] = field(default_factory=list)
    status: str = "ACTIVE"


class HypothesisTracker:
    def __init__(self, hypotheses: list[Hypothesis] | None = None):
        self.hypotheses = {x.hypothesis_id: x for x in (hypotheses or [])}

    def add(self, hypothesis: Hypothesis) -> None:
        self.hypotheses[hypothesis.hypothesis_id] = hypothesis

    def observe(self, evidence_id: str, predicted: dict[str, str], observed: str) -> None:
        for h in self.hypotheses.values():
            prediction = predicted.get(h.hypothesis_id)
            if prediction is None:
                continue
            if prediction == observed:
                h.confidence = min(0.99, h.confidence + 0.05)
                h.supporting_evidence.append(evidence_id)
            else:
                # Contradiction weakens but does not immediately delete a dormant
                # explanation; this is intentionally non-collapsing.
                h.confidence = max(0.01, h.confidence - 0.03)
                h.conflicting_evidence.append(evidence_id)

    def to_dict(self) -> dict[str, Any]:
        return {"hypotheses": [asdict(x) for x in self.hypotheses.values()]}
