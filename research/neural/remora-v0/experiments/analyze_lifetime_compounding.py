from __future__ import annotations

"""Post-process the immutable lifetime run without changing its raw record."""

import argparse
import hashlib
import json
import math
import statistics
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


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _bootstrap_mean_ci(values: list[float]) -> dict[str, Any]:
    """Exact small-n percentile bootstrap; paired inputs are already aligned."""

    if not values:
        return {"lower": None, "upper": None, "resamples": 0}
    resamples: list[float] = []
    count = len(values)
    for code in range(count**count):
        sample = []
        remaining = code
        for _ in range(count):
            sample.append(values[remaining % count])
            remaining //= count
        resamples.append(sum(sample) / count)
    return {"lower": _quantile(resamples, 0.025), "upper": _quantile(resamples, 0.975), "resamples": len(resamples)}


_T95 = {1: None, 2: 4.302652729911275, 3: 3.182446305284263, 4: 2.7764451051977987, 5: 2.570581835636305, 6: 2.446911851144, 7: 2.3646242515927844, 8: 2.306004135204166}


def _summary(values: list[float]) -> dict[str, Any]:
    mean = _mean(values)
    std = _std(values)
    t_critical = _T95.get(len(values))
    t_ci = {"lower": None, "upper": None, "confidence": 0.95, "method": "not available for n<2"}
    if mean is not None and std is not None and len(values) >= 2:
        half = float(t_critical or 1.96) * std / math.sqrt(len(values))
        t_ci = {"lower": mean - half, "upper": mean + half, "confidence": 0.95, "method": "two-sided t interval" if t_critical else "normal approximation"}
    return {"values": values, "mean": mean, "std": std, "t_ci": t_ci, "bootstrap_percentile_ci": _bootstrap_mean_ci(values)}


def _post_task(stage: dict[str, Any], task_id: str, *, when: str) -> dict[str, Any] | None:
    evaluations = stage.get("evaluations", [])
    matches = [item for item in evaluations if item.get("evaluation_kind") == when]
    if not matches:
        return None
    return matches[-1].get("tasks", {}).get(task_id)


def _stage_metrics(run: dict[str, Any], arm: str, stage_index: int) -> dict[str, Any]:
    stage = run["arms"][arm]["stage_results"][stage_index]
    current = stage["stage_id"]
    pre = _post_task(stage, current, when="pre_stage_all_tasks")
    after = _post_task(stage, current, when="post_stage_all_tasks")
    pre_accuracy = pre.get("valid", {}).get("teacher_forced_accuracy") if pre else stage.get("current_task_primary_accuracy_before")
    after_accuracy = after.get("valid", {}).get("teacher_forced_accuracy") if after else stage.get("current_task_primary_accuracy_after")
    gain = float(after_accuracy) - float(pre_accuracy)
    return {
        "seed": run["seed"],
        "arm": arm,
        "stage_id": current,
        "concept_id": stage["concept_id"],
        "current_task_accuracy_before": float(pre_accuracy),
        "current_task_accuracy_after": float(after_accuracy),
        "current_task_accuracy_gain": gain,
        "new_task_tokens": int(stage["new_task_tokens"]),
        "wall_seconds": float(stage["wall_seconds"]),
        "modeled_full_forward_training_flops": int(stage["modeled_full_forward_training_flops"]),
        "modeled_trainable_update_flops": int(stage["modeled_trainable_update_flops"]),
        "gain_per_new_token": gain / max(float(stage["new_task_tokens"]), 1.0),
        "gain_per_wall_second": gain / max(float(stage["wall_seconds"]), 1e-9),
        "gain_per_modeled_full_forward_flop": gain / max(float(stage["modeled_full_forward_training_flops"]), 1.0),
        "gain_per_trainable_parameter": gain / max(float(stage["trainable_parameters"]), 1.0),
        "threshold": float(stage["threshold"]),
        "threshold_step": stage.get("threshold_step"),
        "threshold_new_task_tokens": stage.get("threshold_new_task_tokens"),
        "base_valid_loss_before": stage.get("base_valid_loss_before", {}),
        "base_valid_loss_after": stage.get("base_valid_loss_after", {}),
        "old_task_primary_deltas": stage.get("old_task_primary_deltas", {}),
        "changed_fraction_total": stage["changed_parameters"]["changed_fraction"],
        "changed_fraction_selected": stage["changed_trainable_island"]["changed_fraction_of_selected"],
    }


def _retention_metrics(run: dict[str, Any], arm: str, stage_index: int) -> list[dict[str, Any]]:
    stage = run["arms"][arm]["stage_results"][stage_index]
    pre_tasks = _post_task(stage, "__all__", when="pre_stage_all_tasks")
    after_tasks = _post_task(stage, "__all__", when="post_stage_all_tasks")
    # ``_post_task`` is intentionally strict for normal task IDs; pull the
    # complete dictionaries here for the previous-task retention table.
    evaluations = stage.get("evaluations", [])
    pre = next((item.get("tasks", {}) for item in evaluations if item.get("evaluation_kind") == "pre_stage_all_tasks"), {})
    after = next((item.get("tasks", {}) for item in reversed(evaluations) if item.get("evaluation_kind") == "post_stage_all_tasks"), {})
    result = []
    for task_id in stage.get("rehearsal_task_ids", []):
        before = pre.get(task_id, {}).get("valid", {}).get("teacher_forced_accuracy")
        current = after.get(task_id, {}).get("valid", {}).get("teacher_forced_accuracy")
        if before is None or current is None:
            continue
        result.append({"task_id": task_id, "accuracy_before": before, "accuracy_after": current, "absolute_delta": current - before})
    return result


def run(
    input_path: str | Path = ROOT / "results" / "lifetime-compounding.json",
    output: str | Path = ROOT / "results" / "lifetime-compounding-analysis-v1.json",
) -> dict[str, Any]:
    input_path = Path(input_path)
    raw = json.loads(input_path.read_text())
    arms = ["remora_local_rehearsal", "baseline_lora_rehearsal", "remora_local_target_only", "baseline_lora_target_only", "remora_frozen", "baseline_frozen"]
    stage_ids = raw["curriculum"]["task_order"]
    per_stage: dict[str, dict[str, Any]] = {}
    for arm in arms:
        per_stage[arm] = {}
        for stage_index, stage_id in enumerate(stage_ids):
            metrics = [_stage_metrics(run, arm, stage_index) for run in raw["runs"]]
            retention = [entry for run in raw["runs"] for entry in _retention_metrics(run, arm, stage_index)]
            per_stage[arm][stage_id] = {
                "per_seed": metrics,
                "aggregate": {
                    key: _summary([float(item[key]) for item in metrics])
                    for key in ("current_task_accuracy_before", "current_task_accuracy_after", "current_task_accuracy_gain", "new_task_tokens", "wall_seconds", "gain_per_new_token", "gain_per_wall_second", "gain_per_modeled_full_forward_flop", "gain_per_trainable_parameter", "changed_fraction_total", "changed_fraction_selected")
                },
                "threshold_step": _summary([float(item["threshold_step"]) for item in metrics if item["threshold_step"] is not None]),
                "threshold_hit_rate": sum(item["threshold_step"] is not None for item in metrics) / max(len(metrics), 1),
                "previous_task_retention_per_observation": retention,
                "base_valid_loss_after": {domain: _summary([float(item["base_valid_loss_after"].get(domain, float("nan"))) for item in metrics]) for domain in ("text", "code")},
            }

    paired: dict[str, Any] = {}
    for left_arm, right_arm in (("remora_local_rehearsal", "baseline_lora_rehearsal"), ("remora_local_target_only", "baseline_lora_target_only")):
        key = f"{left_arm}_minus_{right_arm}"
        paired[key] = {}
        for stage_id in stage_ids:
            left = per_stage[left_arm][stage_id]["per_seed"]
            right = per_stage[right_arm][stage_id]["per_seed"]
            paired[key][stage_id] = {
                metric: _summary([float(a[metric]) - float(b[metric]) for a, b in zip(left, right)])
                for metric in ("current_task_accuracy_after", "current_task_accuracy_gain", "gain_per_new_token", "gain_per_wall_second", "gain_per_modeled_full_forward_flop", "gain_per_trainable_parameter", "wall_seconds", "changed_fraction_total")
            }
            paired[key][stage_id]["threshold_step_delta"] = _summary([
                float(a["threshold_step"]) - float(b["threshold_step"])
                for a, b in zip(left, right)
                if a["threshold_step"] is not None and b["threshold_step"] is not None
            ])

    related = [stage_id for stage_id, concept in zip(stage_ids, raw["curriculum"]["concept_order"]) if concept == "parity"]
    compounding = {}
    for arm in ("remora_local_rehearsal", "baseline_lora_rehearsal"):
        compounding[arm] = {
            "related_stage_ids": related,
            "per_stage_threshold_hit_rate": {stage_id: per_stage[arm][stage_id]["threshold_hit_rate"] for stage_id in related},
            "per_stage_gain_per_new_token": {stage_id: per_stage[arm][stage_id]["aggregate"]["gain_per_new_token"] for stage_id in related},
            "per_stage_gain_per_wall_second": {stage_id: per_stage[arm][stage_id]["aggregate"]["gain_per_wall_second"] for stage_id in related},
            "interpretation": "DERIVED: later-stage advantage is not inferred from a single endpoint; compare the complete stage curve and threshold hit rates.",
        }

    result = {
        "schema": "remora-v1-lifetime-compounding-analysis-v1",
        "raw_result": str(input_path),
        "raw_result_sha256": _sha256(input_path),
        "raw_result_immutable": True,
        "experiment_family": "LIFETIME-COMPOUNDING-001+",
        "seeds": raw["training"]["seeds"],
        "stage_ids": stage_ids,
        "per_stage": per_stage,
        "paired": paired,
        "compounding": compounding,
        "labels": {
            "MEASURED": ["raw accuracy/loss/wall/tokens/parameter values copied from immutable run"],
            "DERIVED": ["paired differences", "confidence intervals", "retention tables", "learning velocity normalizations", "threshold hit rates", "compounding curves"],
            "MODELED": ["full-forward and trainable-update FLOP denominators copied from declared model"],
            "ESTIMATED": [],
            "HYPOTHESIS": ["Remora advantage compounds with age"],
            "EXTERNAL": ["deterministic curriculum oracle"],
        },
        "interpretation": "DERIVED: this analysis distinguishes absolute retention from accuracy deltas and reports compute-normalized paired uncertainty; it does not upgrade the original small-sample run into a superiority claim.",
    }
    write_json(output, result)
    record_experiment(
        ROOT,
        "LIFETIME-COMPOUNDING-002",
        "The lifetime comparison should be interpreted from paired, compute-normalized stage curves rather than an ambiguous endpoint retention summary.",
        f"Read immutable {input_path}, derive per-seed stage gains/retention/velocity and paired bootstrap/t intervals, and write a separate analysis artifact.",
        "The analysis preserves the raw hash, exposes per-seed values, and makes threshold/compute normalization explicit.",
        "The raw result is modified, confidence intervals are omitted for the central paired metrics, or accuracy deltas are mislabeled as retention.",
        f"python -m experiments.analyze_lifetime_compounding --input {input_path} --output {output}",
        0,
        {"raw_result_sha256": result["raw_result_sha256"], "stage_count": len(stage_ids), "seed_count": len(raw["runs"])},
        result["interpretation"],
        "Use the adversarial reading to guide consolidation, aged surgery, bus ablation, and resurrection before any scaling decision.",
        hardware={"mode": "immutable_result_postprocessing", "input_result": str(input_path), "output_result": str(output)},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an immutable lifetime-compounding result.")
    parser.add_argument("--input", default=str(ROOT / "results" / "lifetime-compounding.json"))
    parser.add_argument("--output", default=str(ROOT / "results" / "lifetime-compounding-analysis-v1.json"))
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(json.dumps({"output": args.output, "paired": result["paired"], "compounding": result["compounding"]}, indent=2))


if __name__ == "__main__":
    main()
