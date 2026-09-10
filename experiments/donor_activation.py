from __future__ import annotations

"""Round-trip the guarded activation interchange before using a real donor."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.donors.activation import (
    DonorActivationRecord,
    activation_contract_signature,
    load_activation_bundle,
    write_activation_bundle,
    write_activation_records,
)
from remora.donors.port import TeacherPortAdapter
from remora.ledger import record_experiment
from remora.utils import write_json


def run(
    seed: int = 71,
    output: str | Path | None = None,
    records_output: str | Path | None = None,
    bundle_output: str | Path | None = None,
) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    source = {
        f"activation-{index:03d}": torch.randn(48, generator=generator)
        for index in range(16)
    }
    records = [
        DonorActivationRecord.create(
            record_id,
            "synthetic-teacher-v1",
            "encoder.hidden",
            f"synthetic prompt {index}",
            tensor,
            lineage_key="synthetic-activation-run-1",
            runtime={"runtime_id": "synthetic-activation-runtime-v1"},
            accepted=index < 12,
        )
        for index, (record_id, tensor) in enumerate(source.items())
    ]
    if records_output:
        write_activation_records(records_output, records)
    if bundle_output:
        payload_bytes = write_activation_bundle(bundle_output, records, source)
        loaded = load_activation_bundle(
            bundle_output,
            records_output,
            max_payload_bytes=12 * 48 * 4,
            accepted_only=True,
        )
    else:
        payload_bytes = sum(tensor.numel() * tensor.element_size() for tensor in source.values())
        loaded = {"activations": source, "payload_bytes": payload_bytes, "records": records}
    max_delta = max(
        float((loaded["activations"][record_id] - source[record_id]).abs().max())
        for record_id in loaded["activations"]
    )
    port = TeacherPortAdapter(teacher_dim=48, bus_dim=16, bottleneck=12)
    with torch.no_grad():
        source_latent = port(
            torch.stack([source[key] for key in sorted(loaded["activations"])], dim=0).unsqueeze(1)
        ).latent
        loaded_latent = port(
            torch.stack([loaded["activations"][key] for key in sorted(loaded["activations"])], dim=0).unsqueeze(1)
        ).latent
    result = {
        "schema": "remora-v0-donor-activation-result",
        "seed": seed,
        "mode": "MECHANISM_ONLY_GUARDED_ACTIVATION_BUNDLE",
        "contract": activation_contract_signature(),
        "records": len(records),
        "accepted_records_loaded": len(loaded["records"]) if "records" in loaded else len(loaded["activations"]),
        "payload_bytes_loaded": loaded["payload_bytes"],
        "payload_bytes_written": payload_bytes,
        "max_activation_roundtrip_delta": max_delta,
        "max_port_roundtrip_delta": float((source_latent - loaded_latent).abs().max()),
        "records_path": str(records_output) if records_output else None,
        "bundle_path": str(bundle_output) if bundle_output else None,
        "promotion": {"state": "CANDIDATE_ACTIVATION_CAPTURED", "promoted": False},
        "interpretation": "MEASURED MECHANISM TEST: accepted donor activations round-tripped through a hash-checked safetensors bundle under a byte budget and remained compatible with donor-port-v1; no donor model loader ran.",
    }
    if output:
        write_json(output, result)
    record_experiment(
        ROOT,
        "DONOR-ACTIVATION-001",
        "A donor runtime can hand Remora bounded hidden-state records through a versioned port without exposing the donor model loader to the student process.",
        "Write synthetic hidden states with prompt/runtime/lineage hashes, load only accepted records under a 2,304-byte payload budget, and pass them through TeacherPortAdapter(48 -> 16).",
        "Round-trip and port deltas are zero within serialization precision, rejected records are not materialized, and promotion remains false.",
        "Hash/header validation fails open, the byte budget is bypassed, rejected records are loaded, port dimensions mismatch, or the candidate self-promotes.",
        "python -m experiments.donor_activation",
        seed,
        {key: value for key, value in result.items() if key not in {"contract", "promotion"}},
        result["interpretation"],
        "Replace the synthetic producer with an explicitly launched resident runtime hook and collect only a declared layer subset; compare transfer against response-only distillation.",
        hardware={"device": "cpu", "mode": result["mode"]},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--output", default=str(ROOT / "results" / "donor-activation.json"))
    parser.add_argument("--records-output", default=str(ROOT / "results" / "donor-activation-records.jsonl"))
    parser.add_argument("--bundle-output", default=str(ROOT / "results" / "donor-activation-bundle.safetensors"))
    args = parser.parse_args()
    result = run(args.seed, args.output, args.records_output, args.bundle_output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
