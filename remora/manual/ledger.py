from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModuleRecord:
    module_id: str
    version: str
    inputs: list[str]
    outputs: list[str]
    state_owner: str
    parameter_count: int
    ancestry: list[str] = field(default_factory=list)
    status: str = "ACTIVE"
    dependencies: list[str] = field(default_factory=list)
    benchmarks: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    replacements: list[str] = field(default_factory=list)
    content_hash: str = ""


class AssemblyLedger:
    """Factual graph; answers are computed only from registered records/edges."""

    def __init__(self, generation: int = 0):
        self.generation = generation
        self.modules: dict[str, ModuleRecord] = {}
        self.edges: list[dict[str, str]] = []
        self.experiments: dict[str, dict[str, Any]] = {}

    def register_module(self, record: ModuleRecord) -> None:
        if not record.content_hash:
            record.content_hash = hashlib.sha256(
                json.dumps(asdict(record), sort_keys=True).encode()
            ).hexdigest()
        self.modules[record.module_id] = record

    def add_dependency(self, upstream: str, downstream: str, kind: str = "data") -> None:
        if upstream not in self.modules or downstream not in self.modules:
            raise KeyError((upstream, downstream))
        self.edges.append({"upstream": upstream, "downstream": downstream, "kind": kind})
        if upstream not in self.modules[downstream].dependencies:
            self.modules[downstream].dependencies.append(upstream)

    def add_experiment(self, experiment_id: str, record: dict[str, Any]) -> None:
        self.experiments[experiment_id] = dict(record)

    def dependents(self, module_id: str) -> list[str]:
        found: set[str] = set()
        frontier = [module_id]
        while frontier:
            current = frontier.pop()
            for edge in self.edges:
                if edge["upstream"] == current and edge["downstream"] not in found:
                    found.add(edge["downstream"])
                    frontier.append(edge["downstream"])
        return sorted(found)

    def ancestry(self, module_id: str) -> list[str]:
        if module_id not in self.modules:
            return []
        return list(self.modules[module_id].ancestry)

    def minimum_affected_neighborhood(self, module_id: str) -> list[str]:
        return [module_id] + self.dependents(module_id)

    def factual_query(self, query: str, module_id: str | None = None) -> dict[str, Any]:
        if query == "dependents" and module_id is not None:
            return {"module_id": module_id, "dependents": self.dependents(module_id), "source": "graph"}
        if query == "ancestry" and module_id is not None:
            return {"module_id": module_id, "ancestry": self.ancestry(module_id), "source": "ledger"}
        raise ValueError(f"unsupported factual query: {query}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "modules": {k: asdict(v) for k, v in self.modules.items()},
            "edges": self.edges,
            "experiments": self.experiments,
        }

    def save(self, path: str | Path) -> None:
        with Path(path).open("w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")
