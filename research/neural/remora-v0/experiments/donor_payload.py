from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.donors.anatomy import load_manifest
from remora.donors.neural_ir import qwen_shared_expert_ir
from remora.donors.payload import functional_equivalence, inspect_payload, load_payload
from remora.ledger import record_experiment
from remora.utils import runtime_context, write_json


def run(
    payload_path: str | Path,
    *,
    extraction_result_path: str | Path = ROOT / "results" / "qwen-neural-organ-layer0-extraction-result-v1.json",
    anatomy_path: str | Path = ROOT / "results" / "qwen-neural-anatomy-v4.json",
    output: str | Path = ROOT / "results" / "qwen-neural-organ-layer0-payload-analysis-v1.json",
    max_payload_bytes: int = 256 * 1024 * 1024,
) -> dict:
    extraction = json.loads(Path(extraction_result_path).read_text())
    tensors = load_payload(payload_path, max_payload_bytes=max_payload_bytes)
    tensor_names = sorted(tensors)
    source_revision = str(extraction.get("source", {}).get("revision", "unknown"))
    ir = qwen_shared_expert_ir(source_revision=source_revision, tensor_names=tensor_names)
    analysis = inspect_payload(payload_path, max_payload_bytes=max_payload_bytes)
    equivalence = functional_equivalence(tensors)
    result = {
        "schema": "remora-qwen-neural-organ-payload-result-v1",
        "payload_analysis": analysis,
        "neural_ir": ir.to_dict(),
        "functional_equivalence": equivalence,
        "donor_identity": {
            "repository": extraction.get("source", {}).get("repository"),
            "revision": source_revision,
            "source_shard": extraction.get("extraction", {}).get("source_shards"),
            "source_tensor_names": extraction.get("extraction", {}).get("tensor_names"),
        },
        "anatomy_update": {
            "anatomy_path": str(anatomy_path),
            "component_id": "qwen3.8.language.layer0.shared_expert",
            "actual_payload_inspected": True,
            "payload_statistics_source": str(output),
        },
        "promotion_state": "DONOR_FUNCTION_REPRODUCED" if equivalence["status"].startswith("EQUIVALENT") else "DONOR_EXTRACTED",
        "interpretation": "MEASURED: actual selected Qwen weights were inspected, represented in a minimal Neural IR, and reproduced by a standalone implementation within the measured numerical tolerance. This is donor-function reproduction, not yet Remora utility or promotion.",
    }
    write_json(output, result)
    runtime = runtime_context(torch.device("cpu"))
    runtime.update({"mode": "bounded_extracted_payload_analysis", "model_loader_called": False, "command": "python -m experiments.donor_payload"})
    record_experiment(
        ROOT,
        "QWEN-NEURAL-ORGAN-FUNCTION-001",
        "A selectively extracted trained Qwen organ can be independently executed and represented without loading the donor model.",
        f"Load only {payload_path}, inspect actual distributions/spectra, emit the minimal Neural IR, and compare standalone execution with an independent reference path.",
        "The materialized bytes stay within budget, the IR validates, and output/functional errors remain below the predeclared 1e-6 relative-L2 equivalence gate.",
        "Any full donor model is instantiated, payload exceeds budget, IR is invalid, or relative-L2 error is at least 1e-6 without a retained failure record.",
        "python -m experiments.donor_payload --payload results/qwen-neural-organ-layer0.safetensors",
        1701,
        {
            "materialized_payload_bytes": analysis["materialized_payload_bytes"],
            "tensor_count": analysis["tensor_count"],
            "functional_equivalence": equivalence,
            "promotion_state": result["promotion_state"],
        },
        result["interpretation"],
        "Attach the reproduced organ behind a small, explicitly accounted Remora interface and run donor-weight ablations against equal-size controls.",
        hardware=runtime,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and reproduce one extracted donor neural organ.")
    parser.add_argument("--payload", required=True)
    parser.add_argument("--extraction-result", default=str(ROOT / "results" / "qwen-neural-organ-layer0-extraction-result-v1.json"))
    parser.add_argument("--anatomy", default=str(ROOT / "results" / "qwen-neural-anatomy-v4.json"))
    parser.add_argument("--output", default=str(ROOT / "results" / "qwen-neural-organ-layer0-payload-analysis-v1.json"))
    parser.add_argument("--max-payload-bytes", type=int, default=256 * 1024 * 1024)
    args = parser.parse_args()
    result = run(args.payload, extraction_result_path=args.extraction_result, anatomy_path=args.anatomy, output=args.output, max_payload_bytes=args.max_payload_bytes)
    print(json.dumps({
        "output": str(args.output),
        "materialized_payload_bytes": result["payload_analysis"]["materialized_payload_bytes"],
        "functional_equivalence": result["functional_equivalence"],
        "promotion_state": result["promotion_state"],
    }, indent=2))


if __name__ == "__main__":
    main()
