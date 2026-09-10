from __future__ import annotations

"""Provenance-checked response records for black-box donor distillation."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DonorResponseRecord:
    record_id: str
    donor_id: str
    prompt: str
    response: str
    prompt_sha256: str
    response_sha256: str
    lineage_key: str
    runtime: dict[str, Any]
    verifier: dict[str, Any]
    accepted: bool

    @classmethod
    def create(
        cls,
        record_id: str,
        donor_id: str,
        prompt: str,
        response: str,
        *,
        lineage_key: str,
        runtime: dict[str, Any],
        verifier: dict[str, Any],
        accepted: bool,
    ) -> "DonorResponseRecord":
        return cls(
            record_id=record_id,
            donor_id=donor_id,
            prompt=prompt,
            response=response,
            prompt_sha256=_sha256_text(prompt),
            response_sha256=_sha256_text(response),
            lineage_key=lineage_key,
            runtime=dict(runtime),
            verifier=dict(verifier),
            accepted=bool(accepted),
        )

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "DonorResponseRecord":
        record = cls(**row)
        record.verify_integrity()
        return record

    def verify_integrity(self) -> None:
        if self.prompt_sha256 != _sha256_text(self.prompt):
            raise ValueError(f"prompt hash mismatch for {self.record_id}")
        if self.response_sha256 != _sha256_text(self.response):
            raise ValueError(f"response hash mismatch for {self.record_id}")
        if not self.lineage_key:
            raise ValueError(f"missing lineage key for {self.record_id}")
        if not self.runtime.get("runtime_id"):
            raise ValueError(f"missing runtime identity for {self.record_id}")
        if not self.verifier.get("verifier_id"):
            raise ValueError(f"missing verifier identity for {self.record_id}")


def write_response_records(path: str | Path, records: Iterable[DonorResponseRecord]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w") as handle:
        for record in records:
            record.verify_integrity()
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
            count += 1
    return count


def load_response_records(
    path: str | Path,
    *,
    accepted_only: bool = True,
    verifier: Callable[[DonorResponseRecord], bool] | None = None,
) -> list[DonorResponseRecord]:
    records: list[DonorResponseRecord] = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = DonorResponseRecord.from_dict(json.loads(line))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid donor response at line {line_number}: {exc}") from exc
            if accepted_only and not record.accepted:
                continue
            if verifier is not None and not verifier(record):
                continue
            records.append(record)
    return records


def response_contract_signature() -> dict[str, Any]:
    return {
        "version": "donor-response-v1",
        "required": [
            "record_id",
            "donor_id",
            "prompt",
            "response",
            "prompt_sha256",
            "response_sha256",
            "lineage_key",
            "runtime",
            "verifier",
            "accepted",
        ],
        "policy": "verify hashes and external verifier before distillation; donor responses are evidence, not ground truth by themselves",
    }
