from __future__ import annotations

"""Add donor-assimilation accounting to an immutable graft result.

The original graft run predates the explicit donor-cost schema.  This
postprocessor derives the missing accounting without rerunning or modifying
that result.  It deliberately refuses to invent the Qwen pretraining bill:
the downloaded checkpoint has no optimizer/data/compute history.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.ledger import record_experiment
from remora.utils import write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"values": [], "mean": None, "std": None}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
    return {"values": values, "mean": mean, "std": math.sqrt(variance)}


def _extraction_accounting(extraction_path: Path, header_path: Path) -> dict[str, Any]:
    extraction = json.loads(extraction_path.read_text())
    header = json.loads(header_path.read_text())
    tensors = extraction["extraction"]
    source_payload_bytes = int(tensors["observed_payload_bytes"])
    donor_parameters = int(source_payload_bytes // 2)  # selected source tensors are BF16
    parsed_total = int(header.get("headers", {}).get("parsed_payload_bytes", 0))
    return {
        "source_model": extraction["source"]["repository"],
        "source_revision": extraction["source"]["revision"],
        "source_shards": tensors["source_shards"],
        "source_payload_bytes_selected": source_payload_bytes,
        "source_payload_sha256_selected_bundle": tensors["output_sha256"],
        "source_model_payload_bytes_from_headers": parsed_total,
        "selected_payload_fraction_of_source_payload": source_payload_bytes / max(parsed_total, 1),
        "donor_parameter_count": donor_parameters,
        "source_dtype": "BF16",
        "runtime_dtype": "FP32",
        "runtime_storage_bytes": donor_parameters * 4,
        "storage_conversion": "BF16 payload represented as FP32 buffers in the standalone Remora organ; no learned numerical transformation was applied.",
    }


def run(
    input_path: str | Path = ROOT / "results" / "qwen-neural-graft-v1.json",
    extraction_path: str | Path = ROOT / "results" / "qwen-neural-organ-layer0-extraction-result-v1.json",
    header_path: str | Path = ROOT / "results" / "qwen-neural-header-manifest-v1.json",
    output: str | Path = ROOT / "results" / "qwen-neural-graft-analysis-v1.json",
) -> dict[str, Any]:
    input_path = Path(input_path)
    extraction_path = Path(extraction_path)
    header_path = Path(header_path)
    raw = json.loads(input_path.read_text())
    extraction = _extraction_accounting(extraction_path, header_path)

    actual_records = [run_item["arms"]["donor_actual_local"] for run_item in raw["runs"]]
    preservation = []
    assimilation = []
    causal_ablation = {name: [] for name in ("shuffled", "random", "zero")}
    matched_arm_advantage = {name: [] for name in ("shuffled", "random")}
    for record in actual_records:
        donor = record["donor"]
        training = record["training"]
        donor_parameters = int(donor["donor_payload_parameter_count"])
        port_parameters = int(donor["port_parameter_count"])
        new_tokens = int(training["new_task_tokens"])
        rehearsal_tokens = int(training["rehearsal_tokens"])
        total_tokens = int(training.get("total_assimilation_tokens", new_tokens + rehearsal_tokens))
        trainable = int(record["trainable_parameters"])
        model_parameters_after = int(record["parameter_count"])
        preservation.append({
            "seed": int(record["seed"]),
            "donor_parameters_preserved_unchanged": donor_parameters,
            "donor_parameters_analytically_transformed": 0,
            "donor_parameters_discarded": 0,
            "newly_trained_parameters": trainable,
            "port_parameters": port_parameters,
            "donor_core_frozen": bool(donor["donor_core_frozen"]),
            "donor_functional_equivalence_status": "EQUIVALENT_WITHIN_FLOAT32_REFERENCE_TOLERANCE",
        })
        modeled_trainable_flops = 6 * trainable * total_tokens
        modeled_forward_flops = 6 * (model_parameters_after + donor_parameters) * total_tokens
        assimilation.append({
            "seed": int(record["seed"]),
            "gradient_steps": int(training["optimization_steps"]),
            "new_task_tokens": new_tokens,
            "rehearsal_tokens": rehearsal_tokens,
            "total_assimilation_tokens": total_tokens,
            "wall_seconds": float(training["wall_seconds"]),
            "modeled_trainable_update_flops": modeled_trainable_flops,
            "modeled_forward_flops_including_donor_core": modeled_forward_flops,
            "trainable_parameter_count": trainable,
            "donor_parameter_count": donor_parameters,
            "tokens_per_preserved_donor_parameter": total_tokens / max(donor_parameters, 1),
            "wall_seconds_per_preserved_donor_parameter": float(training["wall_seconds"]) / max(donor_parameters, 1),
        })
        actual_loss = float(donor["ablation"]["actual"]["target"]["loss"])
        for variant in causal_ablation:
            causal_ablation[variant].append(
                float(donor["ablation"]["controls"][variant]["target"]["loss"]) - actual_loss
            )

    # This compares separately trained, same-port arms.  It is deliberately
    # kept apart from the fixed-port donor-core ablation because the latter is
    # the stronger causal test.
    for run_item in raw["runs"]:
        actual_loss = float(run_item["arms"]["donor_actual_local"]["post"]["target"]["loss"])
        for variant in matched_arm_advantage:
            control_loss = float(run_item["arms"][f"donor_{variant}_local"]["post"]["target"]["loss"])
            matched_arm_advantage[variant].append(control_loss - actual_loss)

    original_compute = {
        "status": "UNMEASURED",
        "value": None,
        "unit": "FLOP",
        "reason": "The Qwen checkpoint contains trained tensors but not the original optimizer state, data-token count, hardware trace, or training FLOP ledger.",
    }
    result = {
        "schema": "remora-v1-donor-assimilation-analysis-v1",
        "experiment_family": "QWEN-NEURAL-ORGAN-003+",
        "raw_result": str(input_path),
        "raw_result_sha256": _sha256(input_path),
        "raw_result_immutable": True,
        "extraction": extraction,
        "donor_parameter_accounting": {
            "per_seed": preservation,
            "summary": {
                "preserved_parameters": _stats([row["donor_parameters_preserved_unchanged"] for row in preservation]),
                "analytically_transformed_parameters": _stats([row["donor_parameters_analytically_transformed"] for row in preservation]),
                "discarded_parameters": _stats([row["donor_parameters_discarded"] for row in preservation]),
                "newly_trained_parameters": _stats([row["newly_trained_parameters"] for row in preservation]),
            },
        },
        "assimilation_cost": {
            "per_seed": assimilation,
            "summary": {
                key: _stats([row[key] for row in assimilation])
                for key in (
                    "gradient_steps",
                    "new_task_tokens",
                    "rehearsal_tokens",
                    "total_assimilation_tokens",
                    "wall_seconds",
                    "modeled_trainable_update_flops",
                    "modeled_forward_flops_including_donor_core",
                    "tokens_per_preserved_donor_parameter",
                    "wall_seconds_per_preserved_donor_parameter",
                )
            },
        },
        "retained_donor_capability_evidence": {
            "functional_reproduction": {
                "source": "results/qwen-neural-organ-layer0-payload-analysis-v1.json",
                "status": "EQUIVALENT_WITHIN_FLOAT32_REFERENCE_TOLERANCE",
                "relative_l2_error": 0.0,
                "cosine_similarity": 1.0000004768371582,
                "scope": "standalone shared-expert computation, not full Qwen block behavior",
            },
            "fixed_port_core_ablation_target_loss_delta": {
                variant: _stats(values) for variant, values in causal_ablation.items()
            },
            "same_budget_retrained_arm_control_minus_actual_target_loss": {
                variant: _stats(values) for variant, values in matched_arm_advantage.items()
            },
            "interpretation": "The frozen donor core is causally responsible for a large held-out output difference when the repaired ports are held fixed, while its advantage over separately trained same-port controls is modest. This is preserved computation evidence, not proof of broad donor capability transfer.",
        },
        "original_donor_training_compute": original_compute,
        "original_compute_avoided_over_assimilation_compute": {
            "status": "NOT_COMPUTABLE",
            "value": None,
            "reason": "Original donor training compute is unmeasured; only assimilation cost is recorded here.",
        },
        "labels": {
            "MEASURED": ["payload bytes", "donor tensor count", "functional-equivalence errors", "ablation loss deltas", "gradient steps", "tokens", "wall seconds"],
            "DERIVED": ["parameter fractions", "tokens per preserved parameter", "control-minus-actual loss", "selected payload fraction"],
            "MODELED": ["trainable-update FLOPs", "forward FLOPs including donor buffers"],
            "UNMEASURED_EXTERNAL": ["original Qwen training compute"],
            "HYPOTHESIS": ["expensive donor capability can be retained at a small assimilation cost"],
        },
        "promotion_state": "CONTROLLED_EXPERIMENT_ONLY",
        "interpretation": "MEASURED/DERIVED: this artifact makes retained-weight reuse and assimilation cost explicit. The central compute-avoidance ratio remains unavailable until a donor training ledger or defensible external accounting is obtained; no ratio is fabricated.",
    }
    write_json(output, result)
    record_experiment(
        ROOT,
        "QWEN-NEURAL-ORGAN-003",
        "A useful donor organ should preserve trained parameters with very little local assimilation compute, but the avoided original training bill must be measured or explicitly left unknown.",
        "Postprocess the immutable aged Qwen shared-expert graft with explicit unchanged/transformed/discarded/new parameter accounting, assimilation tokens/steps/time, modeled FLOPs, and causal donor ablations.",
        "The report distinguishes true frozen donor computation from port repair and refuses to claim an original-training compute ratio without a donor ledger.",
        "Any parameter category is ambiguous, assimilation cost is omitted, or original donor compute is presented as measured when it is not available.",
        f"python -m experiments.analyze_donor_graft --input {input_path}",
        0,
        {
            "raw_result_sha256": result["raw_result_sha256"],
            "donor_parameters_preserved": extraction["donor_parameter_count"],
            "assimilation": result["assimilation_cost"]["summary"],
            "original_compute_status": original_compute["status"],
        },
        result["interpretation"],
        "Use the accounting to prioritize a lower-repair or function-preserving conversion; do not promote the graft until its utility beats a matched fresh organ under the same assimilation budget.",
        hardware={"mode": "immutable_donor_graft_postprocessing", "input": str(input_path), "output": str(output)},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "results" / "qwen-neural-graft-v1.json"))
    parser.add_argument("--extraction", default=str(ROOT / "results" / "qwen-neural-organ-layer0-extraction-result-v1.json"))
    parser.add_argument("--header-manifest", default=str(ROOT / "results" / "qwen-neural-header-manifest-v1.json"))
    parser.add_argument("--output", default=str(ROOT / "results" / "qwen-neural-graft-analysis-v1.json"))
    args = parser.parse_args()
    result = run(args.input, args.extraction, args.header_manifest, args.output)
    print(json.dumps({
        "interpretation": result["interpretation"],
        "donor_accounting": result["donor_parameter_accounting"],
        "assimilation_cost": result["assimilation_cost"]["summary"],
        "capability_evidence": result["retained_donor_capability_evidence"],
        "original_training_compute": result["original_donor_training_compute"],
    }, indent=2))


if __name__ == "__main__":
    main()
