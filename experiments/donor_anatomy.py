from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.config import ModelConfig
from remora.donors.anatomy import build_anatomy_graph, manifest_digest, write_anatomy_graph
from remora.donors.manifest import inspect_resident_model
from remora.ledger import record_experiment


def run(
    donor_path: str | Path,
    output: str | Path = ROOT / "results" / "qwen-neural-anatomy-v1.json",
    target_config_path: str | Path = ROOT / "configs" / "v0_tiny.json",
) -> dict:
    target = ModelConfig.from_json(target_config_path).to_dict() if target_config_path else None
    manifest = inspect_resident_model(donor_path, target, include_tensor_headers=True)
    graph = build_anatomy_graph(manifest, target)
    result = {
        "schema": "remora-qwen-neural-anatomy-result-v1",
        "manifest_digest": manifest_digest(manifest),
        "manifest_source": manifest["path"],
        "anatomy": graph,
        "interpretation": "MEASURED: architecture, index, and safetensors headers were reconstructed without materializing tensor payloads. ESTIMATED: candidate rank and graft difficulty. No transplant or capability claim is made.",
    }
    write_anatomy_graph(output, result)
    selected = graph["candidate_ranking"][0] if graph["candidate_ranking"] else None
    record_experiment(
        ROOT,
        "QWEN-NEURAL-ANATOMY-001",
        "A foreign open-weight checkpoint can expose bounded, meaningful neural component boundaries before any payload is loaded.",
        "Read the local Qwen config, safetensors index, and all shard headers; group tensors into layer/component boundaries and compare geometry with Remora-v0.",
        "The graph reconciles the header inventory, records tensor geometry/bytes/shards/dependencies/state/interface assumptions, and identifies a bounded first candidate.",
        "Any tensor payload is materialized, the full model is instantiated, component boundaries are inferred from model-name folklore alone, or ranking is reported as measured utility.",
        f"python -m experiments.donor_anatomy --path {donor_path} --output {output}",
        0,
        {
            "component_count": graph["component_count"],
            "parsed_tensor_count": graph["inspection"]["parsed_tensor_count"],
            "weights_materialized": graph["inspection"]["weights_materialized"],
            "first_ranked_candidate": selected,
        },
        result["interpretation"],
        "Use the ranked graph only to choose a value-free extraction manifest; inspect one actual bounded organ next.",
        hardware={"mode": "header_only_anatomy", "model_loader_called": False},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a value-free anatomy graph for a resident donor model.")
    parser.add_argument("--path", required=True)
    parser.add_argument("--output", default=str(ROOT / "results" / "qwen-neural-anatomy-v1.json"))
    parser.add_argument("--target-config", default=str(ROOT / "configs" / "v0_tiny.json"))
    args = parser.parse_args()
    result = run(args.path, args.output, args.target_config)
    graph = result["anatomy"]
    print(json.dumps({
        "output": str(args.output),
        "component_count": graph["component_count"],
        "parsed_tensor_count": graph["inspection"]["parsed_tensor_count"],
        "weights_materialized": graph["inspection"]["weights_materialized"],
        "candidate_ranking": graph["candidate_ranking"][:8],
    }, indent=2))


if __name__ == "__main__":
    main()
