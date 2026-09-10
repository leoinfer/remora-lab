from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.config import ModelConfig
from remora.donors.manifest import inspect_resident_model
from remora.donors.registry import DonorRegistry
from remora.ledger import record_experiment
from remora.utils import write_json


def run(
    donor_path: str | Path,
    output: str | Path = ROOT / "results" / "donor-inspection.json",
    target_config_path: str | Path = ROOT / "configs" / "v0_tiny.json",
    include_tensor_headers: bool = True,
) -> dict:
    target = ModelConfig.from_json(target_config_path).to_dict() if target_config_path else None
    manifest = inspect_resident_model(donor_path, target, include_tensor_headers)
    registry = DonorRegistry()
    donor_id = registry.register_manifest(manifest)
    registry.propose_candidate(
        donor_id,
        candidate_id=f"{donor_id}-port-candidate",
        import_mode=manifest["compatibility"]["recommended_import"],
        module_id="donor-port-candidate",
        interface_version="donor-port-v1",
        provenance={"source_manifest_digest": registry.sources[donor_id].manifest_digest},
    )
    summary = {
        "schema": manifest["schema"],
        "donor_path": manifest["path"],
        "source": manifest["source"],
        "license": manifest["license"],
        "inspection": manifest["inspection"],
        "file_summary": manifest["files"],
        "tensor_summary": {
            "indexed": manifest["tensor_index"]["indexed_tensor_count"],
            "parsed_headers": manifest["headers"]["parsed_tensor_count"],
            "payload_bytes_from_headers": manifest["headers"]["parsed_payload_bytes"],
            "header_errors": manifest["headers"]["errors"],
            "index_reconciliation": manifest["headers"]["index_name_reconciliation"],
            "dtype_counts": manifest["headers"]["dtype_counts"],
            "class_counts": manifest["headers"]["class_counts"],
        },
        "compatibility": manifest["compatibility"],
        "local_acquisition": manifest["local_acquisition"],
        "registry": registry.to_dict(),
        "full_manifest": manifest,
    }
    write_json(output, summary)
    metrics = {
        "shard_count": manifest["files"]["shard_count"],
        "shard_bytes": manifest["files"]["total_shard_bytes"],
        "indexed_tensor_count": manifest["tensor_index"]["indexed_tensor_count"],
        "parsed_tensor_count": manifest["headers"]["parsed_tensor_count"],
        "header_error_count": len(manifest["headers"]["errors"]),
        "weights_materialized": manifest["inspection"]["weights_materialized"],
        "direct_graft_status": manifest["compatibility"]["status"],
    }
    record_experiment(
        ROOT,
        "DONOR-INSPECTION-002",
        "A resident open-weight model can be assessed for provenance and port compatibility without loading its tensor values.",
        "Parse local metadata, the safetensors index, and safetensors headers only; do not instantiate a donor model or call get_tensor.",
        "The manifest is complete enough to identify architecture mismatches, tensor classes, shard cost, and license review state while reporting zero materialized weights.",
        "Any tensor payload is materialized, the index/header reconciliation fails without a retained diagnostic, or the compatibility result is inferred from filenames alone.",
        f"python -m experiments.donor_inspection --path {donor_path} --output {output}",
        0,
        metrics,
        "MEASURED: header-only donor inspection completed; this does not claim that donor knowledge has been transferred.",
        "Run an explicitly budgeted frozen-teacher or response-distillation experiment after selecting a license-compliant runtime and held-out evaluator.",
        hardware={"mode": "header_only"},
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a resident donor model without loading weights.")
    parser.add_argument("--path", required=True)
    parser.add_argument("--output", default=str(ROOT / "results" / "donor-inspection.json"))
    parser.add_argument("--target-config", default=str(ROOT / "configs" / "v0_tiny.json"))
    parser.add_argument("--no-tensor-headers", action="store_true")
    args = parser.parse_args()
    result = run(args.path, args.output, args.target_config, not args.no_tensor_headers)
    print(json.dumps({
        "schema": result["schema"],
        "donor_path": result["donor_path"],
        "inspection": result["inspection"],
        "tensor_summary": result["tensor_summary"],
        "compatibility": result["compatibility"],
    }, indent=2))


if __name__ == "__main__":
    main()
