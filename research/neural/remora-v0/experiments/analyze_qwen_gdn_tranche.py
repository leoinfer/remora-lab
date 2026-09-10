from __future__ import annotations

"""Create a small derived statistical summary from the donor tranche JSONs."""

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: list[float]) -> dict[str, Any]:
    return {
        "values": [float(value) for value in values],
        "mean": statistics.mean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
    }


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot quantile an empty list")
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _paired(a: list[float], b: list[float], *, bootstrap_samples: int = 20000) -> dict[str, Any]:
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must have equal non-zero length")
    differences = [float(left - right) for left, right in zip(a, b)]
    generator = random.Random(17031)
    boot = []
    for _ in range(bootstrap_samples):
        sample = [differences[generator.randrange(len(differences))] for _ in differences]
        boot.append(statistics.mean(sample))
    return {
        "difference_per_seed": differences,
        "difference": _stats(differences),
        "bootstrap_mean_difference_ci_95": [_quantile(boot, 0.025), _quantile(boot, 0.975)],
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_note": "descriptive paired seed-resampling interval; n=3 is too small for a strong significance claim",
    }


def run(
    path_result: str | Path,
    interface_result: str | Path,
    native_result: str | Path,
    *,
    output: str | Path = ROOT / "results" / "qwen-neural-gdn-tranche-analysis-v1.json",
) -> dict[str, Any]:
    path_result = Path(path_result)
    interface_result = Path(interface_result)
    native_result = Path(native_result)
    path = json.loads(path_result.read_text())
    interface = json.loads(interface_result.read_text())
    native = json.loads(native_result.read_text())
    actual = [path["task"]["per_seed"][str(seed)]["zero_shot"]["actual"]["core_to_actual_value"]["accuracy"] for seed in path["task"]["seeds"]]
    random_core = [path["task"]["per_seed"][str(seed)]["zero_shot"]["random"]["core_to_actual_value"]["accuracy"] for seed in path["task"]["seeds"]]
    shuffled = [path["task"]["per_seed"][str(seed)]["zero_shot"]["shuffled"]["core_to_actual_value"]["accuracy"] for seed in path["task"]["seeds"]]
    zero = [path["task"]["per_seed"][str(seed)]["zero_shot"]["zero"]["core_to_actual_value"]["accuracy"] for seed in path["task"]["seeds"]]
    shifted = [path["task"]["per_seed"][str(seed)]["zero_shot"]["actual"]["shifted_interface"]["accuracy"] for seed in path["task"]["seeds"]]
    fresh_thresholds = [path["task"]["per_seed"][str(seed)]["fresh_trainable_same_mechanism_threshold_steps"] for seed in path["task"]["seeds"]]

    interface_seed_names = [str(seed) for seed in interface["per_seed"]]
    interface_summary: dict[str, Any] = {}
    for name in ("actual", "random"):
        for rank in ("4", "96"):
            curves = [interface["per_seed"][seed]["curves"][name][rank]["steps"] for seed in interface_seed_names]
            final = [float(curve[-1]["accuracy"]) for curve in curves]
            best = [max(float(row["accuracy"]) for row in curve) for curve in curves]
            interface_summary[f"{name}_rank{rank}"] = {
                "final_accuracy": _stats(final),
                "best_accuracy": _stats(best),
                "threshold_steps": [interface["per_seed"][seed]["curves"][name][rank]["threshold_steps"] for seed in interface_seed_names],
            }

    result = {
        "schema": "remora-qwen-neural-gdn-tranche-analysis-v1",
        "classification": "DERIVED_FROM_MEASURED_RESULT_FILES",
        "inputs": {
            "path": {"path": str(path_result), "sha256": _sha256(path_result)},
            "interface": {"path": str(interface_result), "sha256": _sha256(interface_result)},
            "native": {"path": str(native_result), "sha256": _sha256(native_result)},
        },
        "aged_pathway": {
            "seeds": path["task"]["seeds"],
            "actual_zero_step_accuracy": _stats(actual),
            "random_zero_step_accuracy": _stats(random_core),
            "shuffled_zero_step_accuracy": _stats(shuffled),
            "zero_core_accuracy": _stats(zero),
            "shifted_actual_zero_step_accuracy": _stats(shifted),
            "actual_minus_random": _paired(actual, random_core),
            "actual_minus_zero": _paired(actual, zero),
            "fresh_same_mechanism_threshold_steps": fresh_thresholds,
            "fresh_curve_mean_accuracy": [
                statistics.mean(
                    float(path["task"]["per_seed"][str(seed)]["fresh_trainable_same_mechanism_curve"]["steps"][index]["accuracy"])
                    for seed in path["task"]["seeds"]
                )
                for index in range(len(path["task"]["per_seed"][str(path["task"]["seeds"][0])]["fresh_trainable_same_mechanism_curve"]["steps"]))
            ],
            "fresh_curve_steps": path["task"]["repair_steps"],
        },
        "shifted_interface_repair": interface_summary,
        "native_exact_reuse": {
            "actual_zero_step_accuracy": native["task"]["zero_step_actual_accuracy"],
            "random_zero_step_accuracy": native["task"]["zero_step_random_accuracy"],
            "actual_minus_random": native["task"]["zero_step_actual_minus_random"],
            "shifted_actual_zero_step_accuracy": native["task"]["zero_step_shifted_actual_accuracy"],
            "native_vs_compact": native["function_preservation"],
        },
        "labels": {
            "MEASURED_INPUTS": "the three source JSON artifacts",
            "DERIVED": "means, sample standard deviations, paired differences, and descriptive bootstrap intervals",
            "HYPOTHESIS": "a donor core can be useful before retraining, but the current interface is not transplant-tolerant",
            "UNMEASURED": "original Qwen training compute avoided and energy",
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize measured Qwen GDN donor tranche results.")
    parser.add_argument("--path-result", default=str(ROOT / "results" / "qwen-neural-gdn-remora-path-v5.json"))
    parser.add_argument("--interface-result", default=str(ROOT / "results" / "qwen-neural-gdn-interface-repair-v1.json"))
    parser.add_argument("--native-result", default=str(ROOT / "results" / "qwen-neural-gdn-native-core-v1.json"))
    parser.add_argument("--output", default=str(ROOT / "results" / "qwen-neural-gdn-tranche-analysis-v1.json"))
    args = parser.parse_args()
    result = run(args.path_result, args.interface_result, args.native_result, output=args.output)
    print(json.dumps({"output": args.output, "aged_pathway": result["aged_pathway"]["actual_minus_random"]}, indent=2))


if __name__ == "__main__":
    main()
