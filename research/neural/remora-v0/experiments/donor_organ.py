from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.config import ModelConfig
from remora.donors.anatomy import build_anatomy_graph
from remora.donors.extract import extract_selected_tensors, write_extraction_receipt
from remora.donors.manifest import inspect_resident_model
from remora.donors.organ import select_anatomy_component
from remora.ledger import record_experiment
from remora.utils import runtime_context, write_json


def _runtime(mode: str, command: str) -> dict:
    result = runtime_context(torch.device("cpu"))
    result.update({"mode": mode, "command": command, "model_loader_called": False})
    return result


def run(
    donor_path: str | Path,
    *,
    component_id: str = "qwen3.8.language.layer0.shared_expert",
    header_manifest_output: str | Path = ROOT / "results" / "qwen-neural-header-manifest-v1.json",
    anatomy_path: str | Path = ROOT / "results" / "qwen-neural-anatomy-v3.json",
    selection_output: str | Path = ROOT / "results" / "qwen-neural-organ-layer0-selection.json",
    result_output: str | Path = ROOT / "results" / "qwen-neural-organ-layer0-selection-result.json",
    payload_output: str | Path | None = None,
    extraction_result_output: str | Path | None = None,
    allow_payload: bool = False,
    max_payload_bytes: int = 256 * 1024 * 1024,
    target_config_path: str | Path = ROOT / "configs" / "v0_tiny.json",
) -> dict:
    command = "python -m experiments.donor_organ"
    target = ModelConfig.from_json(target_config_path).to_dict() if target_config_path else None
    manifest = inspect_resident_model(donor_path, target, include_tensor_headers=True)
    graph = build_anatomy_graph(manifest, target)
    selection = select_anatomy_component(graph, component_id, max_payload_bytes=max_payload_bytes)
    write_json(header_manifest_output, manifest)
    write_json(anatomy_path, {"manifest_digest": "recomputed_by_header_pass", "anatomy": graph})
    write_json(selection_output, selection)

    base_result = {
        "schema": "remora-qwen-neural-organ-selection-result-v1",
        "source": manifest.get("source", {}),
        "header_manifest": str(header_manifest_output),
        "anatomy": str(anatomy_path),
        "selection": str(selection_output),
        "component_id": component_id,
        "selection_is_value_free": True,
        "payload_materialized": False,
        "model_loader_called": False,
        "extraction": None,
        "interpretation": "MEASURED: selection was made from config/index/header geometry and byte accounting only. No donor payload was materialized and no transplant promotion occurred.",
    }
    write_json(result_output, base_result)
    record_experiment(
        ROOT,
        "QWEN-NEURAL-ORGAN-SELECT-001",
        "A bounded foreign neural organ can be selected reproducibly without using its trained values.",
        f"Re-inspect Qwen headers, rebuild anatomy, and select {component_id} under a {max_payload_bytes}-byte budget.",
        "The selected tensor names, shards, geometry, and payload bytes are complete and value-free.",
        "Selection depends on payload statistics, exceeds the declared budget, names an absent tensor, or silently loads the model.",
        command,
        0,
        {
            "component_id": component_id,
            "selected_payload_bytes": selection["selected_payload_bytes"],
            "tensor_count": len(selection["selected"][0]["tensor_names"]),
            "weights_materialized": False,
            "selection_is_value_free": True,
        },
        base_result["interpretation"],
        "Run the same selection with explicit payload opt-in and verify the standalone organ before any Remora attachment.",
        hardware=_runtime("value_free_organ_selection", command),
    )

    if not allow_payload:
        return base_result
    if payload_output is None or extraction_result_output is None:
        raise ValueError("payload_output and extraction_result_output are required with allow_payload=True")
    receipt = extract_selected_tensors(
        donor_path,
        manifest,
        selection,
        payload_output,
        allow_payload=True,
        max_payload_bytes=max_payload_bytes,
    )
    extraction_result = {
        "schema": "remora-qwen-neural-organ-extraction-result-v1",
        "source": manifest.get("source", {}),
        "header_manifest": str(header_manifest_output),
        "anatomy": str(anatomy_path),
        "selection": str(selection_output),
        "component_id": component_id,
        "selection_is_value_free": True,
        "payload_materialized": True,
        "model_loader_called": False,
        "extraction": receipt,
        "interpretation": "MEASURED: only the selected Qwen organ payload was materialized. This establishes extraction, not functional equivalence, Remora utility, or promotion.",
    }
    write_extraction_receipt(extraction_result_output, extraction_result)
    record_experiment(
        ROOT,
        "QWEN-NEURAL-ORGAN-EXTRACT-001",
        "A selected Qwen neural organ can be materialized into a standalone payload without loading the donor model.",
        f"Extract only {component_id} after the value-free selection and byte-budget gate.",
        "Observed payload bytes equal header accounting, source shards are explicit, output is outside the donor tree, and the donor candidate remains unpromoted.",
        "Any unselected tensor is read, the byte accounting differs, the donor tree is written, or a capability/promotion claim is made before equivalence and ablation.",
        f"{command} --allow-payload --component-id {component_id}",
        0,
        receipt,
        extraction_result["interpretation"],
        "Build an independent standalone shared-expert harness and compare its output to a reference implementation before wrapping it for Remora.",
        hardware=_runtime("bounded_selective_payload", f"{command} --allow-payload"),
    )
    return extraction_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Select and optionally extract one bounded Qwen neural organ.")
    parser.add_argument("--path", required=True)
    parser.add_argument("--component-id", default="qwen3.8.language.layer0.shared_expert")
    parser.add_argument("--header-manifest-output", default=str(ROOT / "results" / "qwen-neural-header-manifest-v1.json"))
    parser.add_argument("--anatomy", default=str(ROOT / "results" / "qwen-neural-anatomy-v3.json"))
    parser.add_argument("--selection-output", default=str(ROOT / "results" / "qwen-neural-organ-layer0-selection.json"))
    parser.add_argument("--result-output", default=str(ROOT / "results" / "qwen-neural-organ-layer0-selection-result.json"))
    parser.add_argument("--payload-output")
    parser.add_argument("--extraction-result-output")
    parser.add_argument("--allow-payload", action="store_true")
    parser.add_argument("--max-payload-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--target-config", default=str(ROOT / "configs" / "v0_tiny.json"))
    args = parser.parse_args()
    result = run(
        args.path,
        component_id=args.component_id,
        header_manifest_output=args.header_manifest_output,
        anatomy_path=args.anatomy,
        selection_output=args.selection_output,
        result_output=args.result_output,
        payload_output=args.payload_output,
        extraction_result_output=args.extraction_result_output,
        allow_payload=args.allow_payload,
        max_payload_bytes=args.max_payload_bytes,
        target_config_path=args.target_config,
    )
    print(json.dumps({
        "component_id": result["component_id"],
        "payload_materialized": result["payload_materialized"],
        "selection": result["selection"],
        "extraction": result.get("extraction"),
    }, indent=2))


if __name__ == "__main__":
    main()
