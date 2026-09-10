from __future__ import annotations

"""Common-language bus ablation at fixed Remora-v0 scale.

The shared bus is compared with a parameter-free fixed channel slice/zero-pad
coupling.  The direct arm keeps the packet geometry and routed expert socket,
but removes the learned projection, normalization, confidence head, and
shared semantic parameters.  Both arms are initialized from the same common
non-bus tensors and trained from scratch under the same budgets; a separate
aged-checkpoint intervention measures what happens when the learned bus is
removed after lifetime adaptation.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.lifetime_curriculum import build_lifetime_curriculum
from environments.transfer_suite import build_transfer_suite
from experiments.lifetime_compounding import (
    _all_task_evaluation,
    _base_evaluation,
    _checkpoint,
    _cpu_state,
    _run_arm,
    _task_width,
    _train_base,
)
from remora.config import ModelConfig
from remora.ledger import record_experiment
from remora.models import build_model
from remora.plot import write_line_svg
from remora.tokenizer import ByteTokenizer
from remora.utils import choose_device, count_parameters, runtime_context, set_seed, write_json


BUS_MODES = ("shared", "direct")


def _stats(values: list[float]) -> dict:
    if not values:
        return {"values": [], "mean": None, "std": None}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
    return {"values": values, "mean": mean, "std": math.sqrt(variance)}


def _common_state(source: dict[str, torch.Tensor], target: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return source tensors that exist in the target bus variant."""

    target_state = target.state_dict()
    return {
        name: value
        for name, value in source.items()
        if name in target_state and tuple(value.shape) == tuple(target_state[name].shape)
    }


def _build_from_state(
    cfg: ModelConfig,
    bus_mode: str,
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    model = build_model("remora", cfg, bus_mode=bus_mode)
    if bus_mode == "direct":
        common = _common_state(state, model)
        missing, unexpected = model.load_state_dict(common, strict=False)
        load_report = {
            "mode": "common_non_bus_state_transfer",
            "loaded_common_tensors": len(common),
            "missing_after_common_transfer": list(missing),
            "unexpected_source_tensors_ignored": [name for name in state if name not in common],
        }
    else:
        missing, unexpected = model.load_state_dict(state, strict=False)
        load_report = {
            "mode": "full_shared_state_transfer",
            "loaded_tensors": len(state) - len(unexpected),
            "missing": list(missing),
            "unexpected": list(unexpected),
        }
    return model.to(device), load_report


def _aged_bus_intervention(
    checkpoint: str | Path,
    tasks,
    tokenizer: ByteTokenizer,
    suite,
    device: torch.device,
    task_eval_count: int,
    free_running_count: int,
    batch_size: int,
    seq_len: int,
    max_eval_tokens: int,
) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**payload["config"])
    shared, shared_load = _build_from_state(cfg, "shared", payload["state_dict"], device)
    direct, direct_load = _build_from_state(cfg, "direct", payload["state_dict"], device)
    result = {}
    for mode, model, load_report in (("shared", shared, shared_load), ("direct", direct, direct_load)):
        result[mode] = {
            "load_report": load_report,
            "parameter_count": count_parameters(model),
            "tasks": _all_task_evaluation(
                model,
                tasks,
                tokenizer,
                device,
                task_eval_count=task_eval_count,
                free_running_count=free_running_count,
            ),
            "base": _base_evaluation(
                model,
                suite,
                device=device,
                batch_size=batch_size,
                seq_len=seq_len,
                max_eval_tokens=max_eval_tokens,
            ),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result["checkpoint"] = str(checkpoint)
    result["interpretation"] = (
        "MEASURED aged intervention: direct coupling reuses the aged non-bus "
        "parameters but removes the learned shared bus; this is a causal bus "
        "ablation, not a scratch-trained architecture comparison."
    )
    return result


def run(
    wiki_root: str | Path = "/home/leo/tmp/wikitext-2-raw",
    *,
    seeds: list[int] | tuple[int, ...] = (7, 19, 31),
    device_name: str = "auto",
    base_steps: int = 60,
    stage_steps: int = 40,
    base_eval_every: int = 60,
    stage_eval_every: int = 20,
    stage_count: int = 4,
    batch_size: int = 16,
    seq_len: int = 96,
    max_eval_tokens: int = 1536,
    task_eval_count: int = 16,
    curriculum_train_count: int = 64,
    curriculum_eval_count: int = 32,
    aged_checkpoint: str | Path = ROOT / "checkpoints" / "lifetime-compounding-remora_local_rehearsal-seed7.pt",
    output: str | Path | None = ROOT / "results" / "bus-ablation-v1.json",
    record_ledger: bool = True,
    save_checkpoints: bool = True,
) -> dict:
    if not seeds:
        raise ValueError("at least one seed is required")
    if stage_count < 1 or stage_count > 5:
        raise ValueError("stage_count must be between 1 and 5")
    if max_eval_tokens < batch_size * seq_len:
        raise ValueError("max_eval_tokens must cover one fixed stream batch")
    device = choose_device(device_name)
    cfg = ModelConfig.from_json(ROOT / "configs" / "v0_tiny.json")
    tokenizer = ByteTokenizer(cfg.vocab_size)
    suite = build_transfer_suite(ROOT, wiki_root, tokenizer=tokenizer)
    tasks, curriculum_metadata = build_lifetime_curriculum(
        train_count=curriculum_train_count,
        eval_count=curriculum_eval_count,
    )
    tasks = tasks[:stage_count]
    task_width = _task_width(tasks, tokenizer)
    runs: list[dict] = []

    for seed in seeds:
        # Align all non-bus initial tensors exactly between arms for each
        # seed.  The shared bus has its own learned state; direct has no
        # trainable bus state.  Rebuilding this pair per seed preserves the
        # intended multi-seed independence.
        set_seed(seed)
        shared_init = build_model("remora", cfg, bus_mode="shared")
        shared_init_state = _cpu_state(shared_init)
        direct_init_probe = build_model("remora", cfg, bus_mode="direct")
        direct_common_state = _common_state(shared_init_state, direct_init_probe)
        del shared_init, direct_init_probe
        seed_run = {"seed": seed, "bases": {}, "arms": {}, "bus_modes": list(BUS_MODES)}
        for mode in BUS_MODES:
            if mode == "shared":
                model, init_load = _build_from_state(cfg, "shared", shared_init_state, device)
            else:
                model, init_load = _build_from_state(cfg, "direct", direct_common_state, device)
            history, evaluations = _train_base(
                model,
                suite,
                seed=seed,
                device=device,
                steps=base_steps,
                batch_size=batch_size,
                seq_len=seq_len,
                eval_every=base_eval_every,
                max_eval_tokens=max_eval_tokens,
            )
            base_state = _cpu_state(model)
            base_checkpoint = ROOT / "checkpoints" / f"bus-ablation-base-{mode}-seed{seed}.pt"
            if save_checkpoints:
                _checkpoint(
                    model,
                    base_checkpoint,
                    metadata={
                        "schema": "remora-v1-bus-ablation-base-checkpoint",
                        "model_kind": "remora",
                        "bus_mode": mode,
                        "seed": seed,
                        "config": cfg.to_dict(),
                        "scratch_trained": True,
                    },
                )
            seed_run["bases"][mode] = {
                "bus_mode": mode,
                "init_load": init_load,
                "summary": {"parameters": count_parameters(model)},
                "history": history,
                "evaluations": evaluations,
                "checkpoint": str(base_checkpoint),
                "scratch_trained": True,
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            arm_checkpoint = ROOT / "checkpoints" / f"bus-ablation-lifetime-{mode}-seed{seed}.pt"
            seed_run["arms"][mode] = _run_arm(
                base_state=base_state,
                model_kind="remora",
                mode="remora_local_rehearsal",
                cfg=cfg,
                suite=suite,
                tasks=tasks,
                tokenizer=tokenizer,
                device=device,
                seed=seed,
                stage_steps=stage_steps,
                eval_every=stage_eval_every,
                batch_size=batch_size,
                seq_len=seq_len,
                max_eval_tokens=max_eval_tokens,
                task_eval_count=task_eval_count,
                rehearsal_weight=1.0,
                rehearsal_examples=16,
                task_width=task_width,
                stage_count=stage_count,
                free_running_count=0,
                save_path=arm_checkpoint if save_checkpoints else None,
                remora_bus_mode=mode,
            )
        runs.append(seed_run)

    paired = []
    for stage_index, task in enumerate(tasks):
        shared = [run_item["arms"]["shared"]["stage_results"][stage_index] for run_item in runs]
        direct = [run_item["arms"]["direct"]["stage_results"][stage_index] for run_item in runs]
        paired.append({
            "stage_id": task.stage_id,
            "shared_minus_direct": {
                "primary_accuracy": _stats([
                    a["current_task_primary_accuracy_after"] - b["current_task_primary_accuracy_after"]
                    for a, b in zip(shared, direct)
                ]),
                "shifted_accuracy": _stats([
                    a["shifted_accuracy_after"] - b["shifted_accuracy_after"]
                    for a, b in zip(shared, direct)
                ]),
                "unseen_accuracy": _stats([
                    a["unseen_accuracy_after"] - b["unseen_accuracy_after"]
                    for a, b in zip(shared, direct)
                ]),
                "wall_seconds": _stats([a["wall_seconds"] - b["wall_seconds"] for a, b in zip(shared, direct)]),
                "base_text_loss": _stats([
                    a["base_valid_loss_after"]["text"] - b["base_valid_loss_after"]["text"]
                    for a, b in zip(shared, direct)
                ]),
            },
        })

    aged = None
    if Path(aged_checkpoint).exists():
        aged = _aged_bus_intervention(
            aged_checkpoint,
            build_lifetime_curriculum(train_count=curriculum_train_count, eval_count=curriculum_eval_count)[0],
            tokenizer,
            suite,
            device,
            task_eval_count,
            0,
            batch_size,
            seq_len,
            max_eval_tokens,
        )

    parameter_counts = {
        mode: count_parameters(build_model("remora", cfg, bus_mode=mode)) for mode in BUS_MODES
    }
    result = {
        "schema": "remora-v1-bus-ablation-result",
        "experiment_family": "BUS-ABLATION-001+",
        "mode": "FIXED_SCALE_SCRATCH_SHARED_BUS_VS_DIRECT_COUPLING",
        "device": str(device),
        "config": cfg.to_dict(),
        "training": {
            "seeds": list(seeds),
            "base_steps": base_steps,
            "stage_steps": stage_steps,
            "stage_count": stage_count,
            "base_eval_every": base_eval_every,
            "stage_eval_every": stage_eval_every,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "task_width": task_width,
            "rehearsal_weight": 1.0,
            "rehearsal_examples": 16,
        },
        "parameter_accounting": {
            "by_bus_mode": parameter_counts,
            "shared_bus_parameters": parameter_counts["shared"] - parameter_counts["direct"],
            "mismatch_statement": "Direct coupling removes the learned shared bus parameters; non-bus tensors were initialized identically, and the parameter mismatch is reported rather than hidden.",
        },
        "curriculum": curriculum_metadata,
        "suite": suite.metadata,
        "runs": runs,
        "paired": paired,
        "aged_checkpoint_intervention": aged,
        "labels": {
            "MEASURED": ["loss", "accuracy", "wall_seconds", "changed_parameters", "tokens", "parameter_counts"],
            "DERIVED": ["shared_minus_direct", "retention", "transfer_gaps"],
            "MODELED": ["training_flops"],
            "HYPOTHESIS": ["learned_common_bus_improves_abstraction_beyond_fixed_transport"],
        },
        "promotion": {"state": "CONTROLLED_EXPERIMENT_ONLY", "promoted": False},
        "interpretation": (
            "MEASURED ADVERSARIAL RESULT: the shared learned bus and a fixed direct "
            "coupling were compared from aligned scratch initialization and after "
            "aging; no abstraction claim is made unless shifted/unseen transfer and "
            "retention exceed the direct control under the reported cost mismatch."
        ),
    }
    if output:
        output_path = Path(output)
        write_json(output_path, result)
        write_line_svg(
            output_path.with_name(output_path.stem + "-primary.svg"),
            {
                mode: [
                    _stats([item["arms"][mode]["stage_results"][i]["current_task_primary_accuracy_after"] for item in runs])["mean"] or 0.0
                    for i in range(stage_count)
                ]
                for mode in BUS_MODES
            },
            title="BUS-ABLATION-001 primary exact accuracy",
        )
        write_line_svg(
            output_path.with_name(output_path.stem + "-shifted.svg"),
            {
                mode: [
                    _stats([item["arms"][mode]["stage_results"][i]["shifted_accuracy_after"] for item in runs])["mean"] or 0.0
                    for i in range(stage_count)
                ]
                for mode in BUS_MODES
            },
            title="BUS-ABLATION-001 shifted-interface accuracy",
        )
    if record_ledger:
        record_experiment(
            ROOT,
            "BUS-ABLATION-001",
            "The learned versioned common bus should improve cross-interface transfer and retention beyond a fixed direct hidden-state coupling at the same Remora scale.",
            "Train shared-bus and parameter-free direct-coupling Remora arms from aligned non-bus initialization, then remove the bus from an aged checkpoint and evaluate all lifetime interfaces.",
            "Shared bus improves shifted/unseen transfer or retention enough to justify its measured parameter/runtime cost, and the aged intervention identifies whether the learned bus carried useful information.",
            "Direct coupling matches or beats shared bus on transfer/retention after cost accounting, or the aged bus removal is behaviorally neutral; either result is evidence against the common-language claim.",
            f"python -m experiments.bus_ablation --seeds {' '.join(str(seed) for seed in seeds)} --base-steps {base_steps} --stage-steps {stage_steps}",
            0,
            {
                "parameter_accounting": result["parameter_accounting"],
                "paired": paired,
                "aged_checkpoint": str(aged_checkpoint),
            },
            result["interpretation"],
            "Keep the bus only if its transfer/retention benefit survives the direct coupling and runtime controls; otherwise redesign the language interface before scaling.",
            hardware=runtime_context(device),
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-root", default="/home/leo/tmp/wikitext-2-raw")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--base-steps", type=int, default=60)
    parser.add_argument("--stage-steps", type=int, default=40)
    parser.add_argument("--base-eval-every", type=int, default=60)
    parser.add_argument("--stage-eval-every", type=int, default=20)
    parser.add_argument("--stage-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--max-eval-tokens", type=int, default=1536)
    parser.add_argument("--task-eval-count", type=int, default=16)
    parser.add_argument("--curriculum-train-count", type=int, default=64)
    parser.add_argument("--curriculum-eval-count", type=int, default=32)
    parser.add_argument("--aged-checkpoint", default=str(ROOT / "checkpoints" / "lifetime-compounding-remora_local_rehearsal-seed7.pt"))
    parser.add_argument("--skip-ledger", action="store_true")
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "results" / "bus-ablation-v1.json"))
    args = parser.parse_args()
    result = run(
        args.wiki_root,
        seeds=args.seeds,
        device_name=args.device,
        base_steps=args.base_steps,
        stage_steps=args.stage_steps,
        base_eval_every=args.base_eval_every,
        stage_eval_every=args.stage_eval_every,
        stage_count=args.stage_count,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_eval_tokens=args.max_eval_tokens,
        task_eval_count=args.task_eval_count,
        curriculum_train_count=args.curriculum_train_count,
        curriculum_eval_count=args.curriculum_eval_count,
        aged_checkpoint=args.aged_checkpoint,
        output=args.output,
        record_ledger=not args.skip_ledger,
        save_checkpoints=not args.no_checkpoints,
    )
    print(json.dumps({"interpretation": result["interpretation"], "paired": result["paired"], "aged": result["aged_checkpoint_intervention"]}, indent=2))


if __name__ == "__main__":
    main()
