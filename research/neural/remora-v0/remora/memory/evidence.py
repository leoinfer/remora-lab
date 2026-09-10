from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Evidence:
    evidence_id: str
    source_kind: str  # inherited or experienced
    lineage_id: str
    cluster_id: str
    condition: dict[str, Any]
    label: str
    confidence: float
    provenance: dict[str, Any]


class EvidenceStore:
    """Evidence accounting with bounded support per correlated lineage cluster."""

    def __init__(self, prior_log_odds: float = 0.0):
        self.prior_log_odds = prior_log_odds
        self.items: list[Evidence] = []

    def add(
        self,
        source_kind: str,
        lineage_id: str,
        cluster_id: str,
        condition: dict[str, Any],
        label: str,
        confidence: float = 1.0,
        provenance: dict[str, Any] | None = None,
        evidence_id: str | None = None,
    ) -> Evidence:
        if source_kind not in {"inherited", "experienced"}:
            raise ValueError(source_kind)
        if label not in {"X", "Y"}:
            raise ValueError(label)
        payload = f"{source_kind}|{lineage_id}|{cluster_id}|{condition}|{label}|{len(self.items)}"
        evidence_id = evidence_id or hashlib.sha256(payload.encode()).hexdigest()[:16]
        item = Evidence(
            evidence_id=evidence_id,
            source_kind=source_kind,
            lineage_id=lineage_id,
            cluster_id=cluster_id,
            condition=dict(condition),
            label=label,
            confidence=max(0.0, min(1.0, float(confidence))),
            provenance=provenance or {},
        )
        self.items.append(item)
        return item

    @staticmethod
    def _matches(item: Evidence, query: dict[str, Any]) -> bool:
        return all(item.condition.get(k, "any") in {v, "any"} for k, v in query.items())

    def matching(self, query: dict[str, Any], source_kind: str | None = None) -> list[Evidence]:
        return [
            x
            for x in self.items
            if self._matches(x, query) and (source_kind is None or x.source_kind == source_kind)
        ]

    def _cluster_contributions(
        self, query: dict[str, Any], source_kind: str | None = None
    ) -> dict[str, float]:
        groups: dict[str, list[Evidence]] = {}
        for item in self.matching(query, source_kind=source_kind):
            # A cluster is the unit of independence. Repeated items in it are averaged,
            # not summed, so duplicate replays cannot manufacture certainty.
            groups.setdefault(item.cluster_id, []).append(item)
        contributions = {}
        for cluster, items in groups.items():
            signed = [1.0 if x.label == "X" else -1.0 for x in items]
            contributions[cluster] = sum(s * x.confidence for s, x in zip(signed, items)) / len(items)
        return contributions

    def posterior(self, query: dict[str, Any], source_kind: str | None = None) -> dict:
        contributions = self._cluster_contributions(query, source_kind)
        log_odds = self.prior_log_odds + sum(contributions.values())
        probability_x = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, log_odds))))
        return {
            "query": query,
            "source_kind": source_kind or "combined",
            "probability_x": probability_x,
            "probability_y": 1.0 - probability_x,
            "effective_independent_clusters": len(contributions),
            "cluster_contributions": contributions,
            "raw_observations": len(self.matching(query, source_kind)),
        }

    def separated_posterior(self, query: dict[str, Any]) -> dict[str, dict]:
        return {
            "inherited": self.posterior(query, "inherited"),
            "experienced": self.posterior(query, "experienced"),
            "combined": self.posterior(query, None),
        }

    def to_dict(self) -> dict:
        return {"prior_log_odds": self.prior_log_odds, "items": [asdict(x) for x in self.items]}

    def save(self, path: str | Path) -> None:
        with Path(path).open("w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")


class EpisodicArchive:
    """Slow, append-only provenance store for raw lifetime observations."""

    def __init__(self):
        self.records: list[dict[str, Any]] = []

    def append(self, observation: dict[str, Any]) -> str:
        record = dict(observation)
        record.setdefault("episode_id", f"episode-{len(self.records):05d}")
        record.setdefault("timestamp_index", len(self.records))
        self.records.append(record)
        return record["episode_id"]

    def to_dict(self) -> dict:
        return {"records": self.records}

    def save(self, path: str | Path) -> None:
        with Path(path).open("w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")
