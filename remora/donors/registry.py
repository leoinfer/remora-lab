from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DONOR_STATES = {"DONOR_OBSERVED", "CANDIDATE_EXTRACTED", "FROZEN_EVALUATED", "PROMOTED", "REJECTED", "DORMANT"}


@dataclass
class DonorSourceRecord:
    donor_id: str
    path: str
    repository: str | None
    revision: str | None
    license_name: str | None
    manifest_digest: str
    shard_count: int
    total_bytes: int
    tensor_count: int
    state: str = "DONOR_OBSERVED"
    metadata_only: bool = True


@dataclass
class DonorCandidateRecord:
    candidate_id: str
    donor_id: str
    import_mode: str
    module_id: str
    interface_version: str
    state: str = "CANDIDATE_EXTRACTED"
    external_decision: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)


class DonorRegistry:
    """Factual donor/candidate state; promotion requires an external decision."""

    def __init__(self):
        self.sources: dict[str, DonorSourceRecord] = {}
        self.candidates: dict[str, DonorCandidateRecord] = {}

    def register_manifest(self, manifest: dict[str, Any]) -> str:
        summary = {
            "path": manifest.get("path"),
            "source": manifest.get("source", {}),
            "license": manifest.get("license", {}),
            "files": manifest.get("files", {}),
            "tensor_index": manifest.get("tensor_index", {}),
        }
        digest = hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest()
        donor_id = f"donor-{digest[:12]}"
        source = manifest.get("source", {})
        license_info = manifest.get("license", {})
        files = manifest.get("files", {})
        self.sources[donor_id] = DonorSourceRecord(
            donor_id=donor_id,
            path=str(manifest.get("path", "")),
            repository=source.get("repository"),
            revision=source.get("revision"),
            license_name=license_info.get("name"),
            manifest_digest=digest,
            shard_count=int(files.get("shard_count", 0)),
            total_bytes=int(files.get("total_shard_bytes", 0)),
            tensor_count=int(manifest.get("headers", {}).get("parsed_tensor_count", manifest.get("tensor_index", {}).get("indexed_tensor_count", 0))),
        )
        return donor_id

    def propose_candidate(
        self,
        donor_id: str,
        candidate_id: str,
        import_mode: str,
        module_id: str,
        interface_version: str,
        provenance: dict[str, Any] | None = None,
    ) -> DonorCandidateRecord:
        if donor_id not in self.sources:
            raise KeyError(donor_id)
        candidate = DonorCandidateRecord(
            candidate_id=candidate_id,
            donor_id=donor_id,
            import_mode=import_mode,
            module_id=module_id,
            interface_version=interface_version,
            provenance=provenance or {},
        )
        self.candidates[candidate_id] = candidate
        return candidate

    def decide(
        self,
        candidate_id: str,
        state: str,
        metrics: dict[str, Any],
        reason: str = "",
        external_decision: bool = False,
    ) -> DonorCandidateRecord:
        if state not in DONOR_STATES - {"DONOR_OBSERVED", "CANDIDATE_EXTRACTED"}:
            raise ValueError(state)
        if candidate_id not in self.candidates:
            raise KeyError(candidate_id)
        if state == "PROMOTED" and not external_decision:
            raise PermissionError("donor promotion requires an external experiment-harness decision")
        candidate = self.candidates[candidate_id]
        candidate.state = state
        candidate.metrics = dict(metrics)
        candidate.failure_reason = reason
        candidate.external_decision = bool(external_decision)
        return candidate

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "remora-donor-registry-v1",
            "sources": {key: asdict(value) for key, value in self.sources.items()},
            "candidates": {key: asdict(value) for key, value in self.candidates.items()},
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
