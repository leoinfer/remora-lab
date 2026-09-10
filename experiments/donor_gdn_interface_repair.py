from __future__ import annotations

"""Test whether the GDN donor's shifted-interface failure is locally repairable.

This is deliberately narrower than the aged-path experiment.  The donor core
and task are unchanged; only the fixed input interface is cyclically shifted.
Rank-4 and rank-96 learned socket repairs are compared for actual and random
cores at identical budgets.  Recovery by a large port is not semantic bus
success: the result records the port cost and donor/random separation.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.donors.gdn import QwenGatedDeltaCoreOrgan  # noqa: E402
from remora.utils import count_parameters, runtime_context, set_seed, write_json  # noqa: E402
from remora.ledger import record_experiment  # noqa: E402
from experiments.donor_gdn_head import (  # noqa: E402
    BUS_WIDTH,
    LAYER,
    _conversion_bases,
    _convert_payload,
    _extract_head,
)
from experiments.donor_gdn_remora_pathway import (  # noqa: E402
    CHECKPOINT_PATTERN,
    RECALL_THRESHOLD,
    SEEDS,
    TEST_SAMPLES,
    TRAIN_SAMPLES,
    _make_organ,
    _make_path_dataset,
    _make_task_host,
    _padded_values,
    _retrieval,
    _run_continuous,
)


REPAIR_RANKS = (4, 96)
REPAIR_STEPS = (0, 1, 2, 4, 8, 16, 32, 64)
SELECTED_VALUE_HEAD = 10
TARGET_LAYER = 1
TASK_D_MODEL = 192
MEMORY_ITEMS = 4
MEMORY_DELAY = 9
TASK_TIME = MEMORY_DELAY + 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: list[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    if not values:
        return {"values": [], "mean": None, "std": None}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
    return {"values": values, "mean": mean, "std": math.sqrt(variance)}


def _repair_state(branch: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in branch.named_parameters()
        if parameter.requires_grad
    }


def _load_repair_state(branch: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    named = dict(branch.named_parameters())
    for name, value in state.items():
        named[name].data.copy_(value)


def _curve(
    checkpoint: Path,
    organ: QwenGatedDeltaCoreOrgan,
    train: dict[str, Any],
    test: dict[str, Any],
    *,
    rank: int,
    seed: int,
    steps: tuple[int, ...],
) -> dict[str, Any]:
    set_seed(seed)
    template_model, template_branch, _source = _make_task_host(
        checkpoint, organ, repair_rank=rank, shifted=True
    )
    initial_state = _repair_state(template_branch)
    reference_train = _padded_values(train["values"])
    reference_test = _padded_values(test["values"])
    rows: list[dict[str, Any]] = []
    for budget in steps:
        model, branch, _source = _make_task_host(
            checkpoint, organ, repair_rank=rank, shifted=True
        )
        _load_repair_state(branch, initial_state)
        parameters = branch.port_parameters()
        optimizer = torch.optim.AdamW(parameters, lr=0.03, weight_decay=0.0)
        started = time.perf_counter()
        losses: list[float] = []
        model.train()
        for _ in range(int(budget)):
            trace = _run_continuous(model, train["hidden"])
            scores = (
                F.normalize(trace["path_output"][:, -1].float(), dim=-1)[:, None, :]
                * F.normalize(reference_train.float(), dim=-1)
            ).sum(dim=-1)
            loss = F.cross_entropy(scores * 10.0, train["labels"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            trace = _run_continuous(model, test["hidden"])
        metrics = _retrieval(trace["path_output"], test["labels"], reference_test)
        changed = sum(
            int((parameter.detach() - initial_state[name]).abs().gt(1e-12).sum())
            for name, parameter in branch.named_parameters()
            if name in initial_state
        )
        total_model = count_parameters(model)
        rows.append(
            {
                "gradient_steps": int(budget),
                "accuracy": metrics["accuracy"],
                "mean_target_cosine": metrics["mean_target_cosine"],
                "loss_last": losses[-1] if losses else None,
                "wall_seconds": time.perf_counter() - started,
                "tokens": int(budget * train["hidden"].shape[0] * train["hidden"].shape[1]),
                "trainable_parameters": int(sum(parameter.numel() for parameter in parameters)),
                "changed_parameters": int(changed),
                "changed_fraction_total_model": changed / max(total_model, 1),
            }
        )
        del model, branch
    del template_model, template_branch
    return {
        "interface": "cyclic bus permutation by 7 plus scale 0.9",
        "repair_rank": int(rank),
        "trainable_parameters": int(sum(value.numel() for value in initial_state.values())),
        "steps": rows,
        "threshold_steps": next(
            (int(row["gradient_steps"]) for row in rows if row["accuracy"] >= RECALL_THRESHOLD),
            None,
        ),
    }


def run(
    source_path: str | Path,
    *,
    checkpoint_pattern: str = CHECKPOINT_PATTERN,
    output: str | Path = ROOT / "results" / "qwen-neural-gdn-interface-repair-v1.json",
    seeds: tuple[int, ...] = SEEDS,
    repair_ranks: tuple[int, ...] = REPAIR_RANKS,
    repair_steps: tuple[int, ...] = REPAIR_STEPS,
    record_ledger: bool = True,
    experiment_id: str = "QWEN-DONOR-GDN-INTERFACE-REPAIR-001",
) -> dict[str, Any]:
    source_root = Path(source_path).expanduser().resolve()
    output = Path(output)
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    if not seeds or not repair_ranks or not repair_steps:
        raise ValueError("seeds, repair_ranks, and repair_steps must be non-empty")
    if tuple(sorted(set(repair_ranks))) != tuple(repair_ranks) or min(repair_ranks) <= 0:
        raise ValueError("repair_ranks must be sorted, unique, and positive")
    if tuple(sorted(set(repair_steps))) != tuple(repair_steps) or min(repair_steps) < 0:
        raise ValueError("repair_steps must be sorted, unique, and non-negative")

    selected_payload, selected_metadata = _extract_head(source_root, LAYER, SELECTED_VALUE_HEAD)
    input_basis, output_basis = _conversion_bases(selected_payload)
    actual_organ = _make_organ(selected_payload, input_basis, output_basis, "actual", 0)
    suite_by_seed: dict[str, Any] = {}
    for seed in seeds:
        set_seed(int(seed))
        checkpoint = Path(checkpoint_pattern.format(seed=int(seed)))
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        norm_model, _norm_branch, checkpoint_source = _make_task_host(checkpoint, actual_organ)
        norm = norm_model.blocks[TARGET_LAYER].norm
        train = _make_path_dataset(norm, actual_organ, samples=TRAIN_SAMPLES, seed=int(seed) + 1000)
        test = _make_path_dataset(norm, actual_organ, samples=TEST_SAMPLES, seed=int(seed) + 2000)
        del norm_model
        variants = {
            "actual": actual_organ,
            "random": _make_organ(
                selected_payload, input_basis, output_basis, "random", int(seed) + 5000
            ),
        }
        curves = {
            name: {
                str(rank): _curve(
                    checkpoint,
                    organ,
                    train,
                    test,
                    rank=rank,
                    seed=int(seed) + (9100 if name == "actual" else 9200) + rank,
                    steps=repair_steps,
                )
                for rank in repair_ranks
            }
            for name, organ in variants.items()
        }
        baseline = {
            name: _curve(
                checkpoint,
                organ,
                train,
                test,
                rank=repair_ranks[0],
                seed=int(seed) + (9300 if name == "actual" else 9400),
                steps=(0,),
            )["steps"][0]
            for name, organ in variants.items()
        }
        suite_by_seed[str(seed)] = {
            "checkpoint": {
                "path": str(checkpoint),
                "bytes": int(checkpoint.stat().st_size),
                "sha256": _sha256_file(checkpoint),
                "schema": checkpoint_source.get("schema"),
                "seed": checkpoint_source.get("seed"),
            },
            "dataset": {
                "train_samples": TRAIN_SAMPLES,
                "test_samples": TEST_SAMPLES,
                "memory_items": MEMORY_ITEMS,
                "memory_delay": MEMORY_DELAY,
                "time": TASK_TIME,
                "query_synthesis": "same donor-key analytic synthesis as QWEN-DONOR-GDN-REMORA-PATH-004; no donor training",
            },
            "shifted_zero_repair_baseline": baseline,
            "curves": curves,
        }

    actual_rank4 = [suite_by_seed[str(seed)]["curves"]["actual"][str(repair_ranks[0])]["threshold_steps"] for seed in seeds]
    random_rank4 = [suite_by_seed[str(seed)]["curves"]["random"][str(repair_ranks[0])]["threshold_steps"] for seed in seeds]
    actual_rank_max = [suite_by_seed[str(seed)]["curves"]["actual"][str(repair_ranks[-1])]["threshold_steps"] for seed in seeds]
    result = {
        "schema": "remora-qwen-neural-gdn-interface-repair-result-v1",
        "experiment_id": experiment_id,
        "experiment_family": f"{experiment_id}+",
        "source": {
            "repository": "Qwen/Qwen3.8-Flash-Next",
            "revision": "f5d08274bafd880402bd16f5e3e6c514136ec06c",
            "path": str(source_root),
            "layer": LAYER,
            "value_head": SELECTED_VALUE_HEAD,
            "source_shard": selected_metadata["source_shard"],
            "source_payload_bytes_bf16": selected_metadata["source_payload_bytes_bf16"],
            "selected_payload_materialized_bytes_fp32": selected_metadata["selected_payload_materialized_bytes_fp32"],
            "full_model_materialized": False,
            "model_loader_called": False,
        },
        "hypothesis": "The shifted-interface failure is a bounded port-compatibility problem: a sufficiently small local repair should recover the actual donor core faster than a random core under the same repair budget.",
        "predeclared_gates": {
            "against_hypothesis": [
                "rank-4 actual repair remains at chance while random is comparable",
                "rank-96 actual does not recover or does not separate from random",
                "recovery requires changing frozen donor weights",
                "port parameter cost dominates without donor/random separation",
            ],
            "interpretation_rule": "Any recovery with a large port is interface repair evidence, not common-language abstraction evidence; actual-vs-random and parameter cost must be reported together.",
        },
        "per_seed": suite_by_seed,
        "thresholds": {
            "actual_rank4": actual_rank4,
            "random_rank4": random_rank4,
            "actual_rank96": actual_rank_max,
        },
        "accounting": {
            "rank4_port_parameters": int(4 * (TASK_D_MODEL + BUS_WIDTH + 128 + TASK_D_MODEL)),
            "rank96_port_parameters": int(96 * (TASK_D_MODEL + BUS_WIDTH + 128 + TASK_D_MODEL)),
            "donor_core_parameters_resident": int(actual_organ.donor_parameter_count),
            "donor_core_parameters_modified": 0,
            "source_payload_bytes_bf16": int(selected_metadata["source_payload_bytes_bf16"]),
        },
        "labels": {
            "MEASURED": [
                "shifted-interface repair curves for actual and random frozen cores",
                "threshold records, changed parameters, tokens, and wall time",
                "bounded source extraction and checkpoint hashes",
            ],
            "DERIVED": ["threshold arrays and port parameter counts"],
            "MODELED": [],
            "UNMEASURED": ["semantic bus abstraction", "original donor training compute avoided", "energy"],
        },
        "interpretation": "The result distinguishes local compatibility repair from semantic interface transfer; promotion requires actual donor/random separation at a bounded port cost and does not follow from rank-96 recovery alone.",
        "runtime": {**runtime_context(torch.device("cpu")), "cwd": str(ROOT), "mode": "shifted_gdn_socket_repair_cpu"},
    }
    write_json(output, result)
    if record_ledger:
        record_experiment(
            ROOT,
            experiment_id,
            result["hypothesis"],
            "Keep the selected layer-17/value-head-10 GDN core frozen, shift only the input socket by a known cyclic permutation and scale, and compare rank-4 versus rank-96 learned repairs for actual and random cores.",
            "A small repair recovers the actual donor faster than the equal-budget random core, while frozen donor parameters remain unchanged; interpret large-rank recovery only as compatibility evidence.",
            "Actual repair remains at chance, rank-96 does not separate from random, or apparent recovery is possible only because donor weights were updated or the controls differ in port budget.",
            f"python -m experiments.donor_gdn_interface_repair --source {source_root} --checkpoint-pattern {checkpoint_pattern} --seeds {','.join(str(seed) for seed in seeds)} --repair-ranks {','.join(str(rank) for rank in repair_ranks)} --repair-steps {','.join(str(step) for step in repair_steps)} --experiment-id {experiment_id}",
            int(seeds[0]),
            {
                "actual_rank4_threshold_steps": actual_rank4,
                "random_rank4_threshold_steps": random_rank4,
                "actual_rank96_threshold_steps": actual_rank_max,
                "accounting": result["accounting"],
            },
            result["interpretation"],
            "If a small repair recovers actual but not random, test a learned transplant-tolerant bus condition; if only rank-96 recovers, preserve the result as expensive interface calibration and redesign the bus before scaling.",
            hardware={**runtime_context(torch.device("cpu")), "mode": "shifted_gdn_socket_repair_cpu", "source_model_materialized": False},
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure shifted Qwen GDN socket repair cost.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint-pattern", default=CHECKPOINT_PATTERN)
    parser.add_argument("--output", default=str(ROOT / "results" / "qwen-neural-gdn-interface-repair-v1.json"))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--repair-ranks", default=",".join(str(rank) for rank in REPAIR_RANKS))
    parser.add_argument("--repair-steps", default=",".join(str(step) for step in REPAIR_STEPS))
    parser.add_argument("--experiment-id", default="QWEN-DONOR-GDN-INTERFACE-REPAIR-001")
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()
    result = run(
        args.source,
        checkpoint_pattern=args.checkpoint_pattern,
        output=args.output,
        seeds=tuple(int(value) for value in args.seeds.split(",") if value),
        repair_ranks=tuple(int(value) for value in args.repair_ranks.split(",") if value),
        repair_steps=tuple(int(value) for value in args.repair_steps.split(",") if value),
        record_ledger=not args.no_ledger,
        experiment_id=args.experiment_id,
    )
    print(json.dumps({"output": str(args.output), "thresholds": result["thresholds"]}, indent=2))


if __name__ == "__main__":
    main()
