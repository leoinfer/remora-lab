from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Candidate:
    candidate_id: str
    status: str
    expected_upside: float
    uncertainty: float
    novelty: float
    test_cost: float
    failure_state: dict[str, Any]
    failure_reason: str
    history: list[dict[str, Any]] = field(default_factory=list)


class ResurrectionQueue:
    def __init__(self):
        self.candidates: dict[str, Candidate] = {}

    def add(self, candidate: Candidate) -> None:
        self.candidates[candidate.candidate_id] = candidate

    @staticmethod
    def _context_change(failure_state: dict[str, Any], current_state: dict[str, Any]) -> float:
        keys = set(failure_state) | set(current_state)
        if not keys:
            return 0.0
        changed = sum(failure_state.get(k) != current_state.get(k) for k in keys)
        return changed / len(keys)

    def priorities(self, current_state: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for c in self.candidates.values():
            context_change = self._context_change(c.failure_state, current_state)
            score = (
                c.expected_upside
                * c.uncertainty
                * (0.5 + 0.5 * c.novelty)
                * (1.0 + context_change)
                / max(c.test_cost, 1e-6)
            )
            rows.append({"candidate_id": c.candidate_id, "priority": score, "context_change": context_change, "status": c.status})
        return sorted(rows, key=lambda x: x["priority"], reverse=True)

    def to_dict(self) -> dict:
        return {"candidates": [asdict(x) for x in self.candidates.values()]}
