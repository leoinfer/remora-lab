from __future__ import annotations

"""Small uncertainty-aware surprise tracker used by the v0 experiments."""

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass
class SurpriseObservation:
    observed: int
    cluster_id: str
    nll: float
    brier: float
    provenance: dict[str, Any]


class SurpriseTracker:
    """Aggregate prediction errors by evidence cluster, not raw repetitions."""

    def __init__(self, low_threshold: float = 0.25, high_threshold: float = 1.5):
        self.low_threshold = float(low_threshold)
        self.high_threshold = float(high_threshold)
        self.observations: list[SurpriseObservation] = []

    @staticmethod
    def signal(probabilities: torch.Tensor, observed: torch.Tensor | int) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = probabilities.float()
        if probabilities.ndim == 1:
            probabilities = probabilities.unsqueeze(0)
        if (probabilities < 0).any() or (probabilities.sum(-1) <= 0).any():
            raise ValueError("probabilities must be non-negative and have positive row sums")
        probabilities = probabilities / probabilities.sum(-1, keepdim=True)
        observed_tensor = torch.as_tensor(observed, device=probabilities.device).long().reshape(-1)
        if observed_tensor.numel() != probabilities.size(0):
            raise ValueError("observed labels must match probability rows")
        nll = -probabilities[torch.arange(probabilities.size(0), device=probabilities.device), observed_tensor].clamp_min(1e-8).log()
        one_hot = torch.nn.functional.one_hot(observed_tensor, num_classes=probabilities.size(-1)).float()
        brier = (probabilities - one_hot).square().mean(-1)
        return nll, brier

    def observe(
        self,
        probabilities: torch.Tensor,
        observed: int,
        cluster_id: str,
        provenance: dict[str, Any] | None = None,
    ) -> SurpriseObservation:
        nll, brier = self.signal(probabilities, observed)
        item = SurpriseObservation(int(observed), str(cluster_id), float(nll[0]), float(brier[0]), provenance or {})
        self.observations.append(item)
        return item

    def summary(self) -> dict[str, Any]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for item in self.observations:
            grouped[item.cluster_id].append(item.nll)
        cluster_means = {key: sum(values) / len(values) for key, values in sorted(grouped.items())}
        mean_surprise = sum(cluster_means.values()) / len(cluster_means) if cluster_means else 0.0
        max_surprise = max(cluster_means.values(), default=0.0)
        effective_clusters = len(cluster_means)
        if mean_surprise <= self.low_threshold and max_surprise <= self.low_threshold:
            allocation = 1
            level = "LOW"
        elif effective_clusters >= 3 and mean_surprise >= self.high_threshold:
            allocation = 8
            level = "PERSISTENT_HIGH"
        elif max_surprise >= self.high_threshold:
            allocation = 4
            level = "ISOLATED_OR_CORRELATED_HIGH"
        else:
            allocation = 2
            level = "MODERATE"
        return {
            "raw_observations": len(self.observations),
            "effective_independent_clusters": effective_clusters,
            "cluster_mean_nll": cluster_means,
            "mean_surprise": mean_surprise,
            "max_surprise": max_surprise,
            "level": level,
            "compute_allocation": allocation,
            "thresholds": {"low": self.low_threshold, "high": self.high_threshold},
        }

    def to_dict(self) -> dict[str, Any]:
        return {"observations": [asdict(item) for item in self.observations], "summary": self.summary()}


def scalar_surprise(probability_of_observation: float) -> float:
    """Convenience NLL for logging externally produced categorical events."""

    return -math.log(max(1e-8, min(1.0, float(probability_of_observation))))
