from __future__ import annotations

"""Aged Remora surgery against a near-equal-budget aged Transformer adapter.

Both models have already completed the five-stage lifetime run.  Remora
replaces a used routed expert with a new SwiGLU module and trains only that
module.  The conventional control keeps its aged attention LoRA state frozen,
adds a rank-64 LoRA repair to the corresponding feed-forward block, and trains
only those factors.  The budget mismatch is recorded explicitly.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.lifetime_curriculum import LifetimeTask, build_lifetime_curriculum
from environments.transfer_suite import build_transfer_suite
from experiments.lifetime_compounding import _base_evaluation, _task_width
from experiments.transfer_benchmark import _encode_examples, _evaluate_tasks, _response_loss
from remora.config import ModelConfig
from remora.data import sample_batch
from remora.ledger import record_experiment, record_failure
from remora.metrics import evaluate_stream
from remora.models import build_model
from remora.modules import SwiGLUExpert, matching_suffixes, replace_linear_with_lora
from remora.tokenizer import ByteTokenizer
from remora.utils import (
    changed_parameter_stats,
    choose_device,
    count_parameters,
    freeze_all,
    parameter_snapshot,
    runtime_context,
    set_seed,
    unfreeze_prefixes,
    write_json,
)


REMORA_TARGET_LAYER = 1
REMORA_TARGET_EXPERT = 0
BASELINE_LORA_RANK = 64
BASELINE_LORA_ALPHA = 64.0


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"values": [], "mean": None, "std": None}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
    return {"values": values, "mean": mean, "std": math.sqrt(variance)}


def _load_remora(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model("remora", ModelConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    return model.to(device), payload


def _load_baseline_lora(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**payload["config"])
    model = build_model("baseline", cfg)
    replace_linear_with_lora(
        model,
        matching_suffixes(("attention.qkv", "attention.out")),
        rank=8,
        alpha=8.0,
    )
    model.load_state_dict(payload["state_dict"])
    return model.to(device), payload


def _task_matrix(model: nn.Module, tasks: tuple[LifetimeTask, ...], tokenizer: ByteTokenizer, device: torch.device, count: int) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        task.stage_id: {
            split: _evaluate_tasks(model, list(getattr(task, split)[:count]), tokenizer, device)
            for split in ("valid", "shifted", "unseen")
        }
        for task in tasks
    }


def _base_stream_scores(
    model: nn.Module,
    suite,
    device: torch.device,
    *,
    batch_size: int,
    seq_len: int,
    max_tokens: int,
) -> dict[str, Any]:
    result = {}
    for domain in ("text", "code"):
        stream = suite.valid_streams[domain]
        limit = min(stream.numel(), max_tokens + 1)
        result[domain] = evaluate_stream(model, stream[:limit], batch_size, seq_len, device)
    return result


def _evaluate_bundle(
    model: nn.Module,
    tasks: tuple[LifetimeTask, ...],
    tokenizer: ByteTokenizer,
    suite,
    device: torch.device,
    *,
    task_eval_count: int,
    batch_size: int,
    seq_len: int,
    max_eval_tokens: int,
) -> dict[str, Any]:
    return {
        "tasks": _task_matrix(model, tasks, tokenizer, device, task_eval_count),
        "base_streams": _base_stream_scores(
            model,
            suite,
            device,
            batch_size=batch_size,
            seq_len=seq_len,
            max_tokens=max_eval_tokens,
        ),
    }


def _local_train(
    model: nn.Module,
    target_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    replay_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    base_batches: list[tuple[torch.Tensor, torch.Tensor]],
    selected_names: list[str],
    *,
    steps: int,
    seed: int,
    rehearsal_weight: float,
    device: torch.device,
) -> dict[str, Any]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("aged surgery requires a non-empty local trainable scope")
    optimizer = torch.optim.AdamW(trainable, lr=0.003, betas=(0.9, 0.95), weight_decay=0.0)
    losses = []
    target_tokens = int(target_batch[0].numel())
    replay_tokens = sum(int(batch[0].numel()) for batch in replay_batches)
    base_tokens = sum(int(batch[0].numel()) for batch in base_batches)
    started = time.perf_counter()
    for _ in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        target_loss = _response_loss(model, target_batch)
        replay_losses = [_response_loss(model, batch) for batch in replay_batches]
        replay_losses.extend(model(batch[0], batch[1])[1] for batch in base_batches)
        loss = target_loss + rehearsal_weight * torch.stack(replay_losses).mean()
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        optimizer.step()
        losses.append({"loss": float(loss.detach()), "target_loss": float(target_loss.detach()), "grad_norm": grad_norm})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_tokens = steps * (target_tokens + replay_tokens + base_tokens)
    return {
        "optimization_steps": steps,
        "target_tokens": steps * target_tokens,
        "rehearsal_task_tokens": steps * replay_tokens,
        "rehearsal_base_tokens": steps * base_tokens,
        "total_assimilation_tokens": total_tokens,
        "wall_seconds": time.perf_counter() - started,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "selected_parameter_names": selected_names,
        "trainable_parameters": sum(int(parameter.numel()) for parameter in trainable),
        "modeled_trainable_update_flops": 6 * sum(int(parameter.numel()) for parameter in trainable) * total_tokens,
    }


def _mean_task_accuracy(bundle: dict[str, Any], task_ids: list[str], split: str = "valid") -> float:
    values = [bundle["tasks"][task_id][split]["teacher_forced_accuracy"] for task_id in task_ids]
    return sum(values) / max(len(values), 1)


def _run_seed(
    *,
    remora_checkpoint: Path,
    baseline_checkpoint: Path,
    seed: int,
    device: torch.device,
    tasks: tuple[LifetimeTask, ...],
    tokenizer: ByteTokenizer,
    suite,
    task_width: int,
    steps: int,
    task_eval_count: int,
    batch_size: int,
    seq_len: int,
    max_eval_tokens: int,
    rehearsal_examples: int,
    rehearsal_weight: float,
    save_checkpoints: bool,
) -> dict[str, Any]:
    target_task = tasks[-1]
    target_batch = _encode_examples(list(target_task.train), tokenizer, device)
    replay_batches = [
        _encode_examples(list(task.train[:rehearsal_examples]), tokenizer, device)
        for task in tasks[:-1]
    ]
    # Equal fixed base replay batches for both arms.  They are intentionally
    # materialized once rather than sampled independently inside each arm.
    base_batches = []
    base_generator = torch.Generator(device="cpu").manual_seed(seed + 5000)
    for domain in ("text", "code"):
        stream = suite.train_streams[domain]
        base_batches.append(sample_batch(stream, batch_size, seq_len, device, base_generator))

    remora, remora_source = _load_remora(remora_checkpoint, device)
    baseline, baseline_source = _load_baseline_lora(baseline_checkpoint, device)
    remora_before = _evaluate_bundle(remora, tasks, tokenizer, suite, device, task_eval_count=task_eval_count, batch_size=batch_size, seq_len=seq_len, max_eval_tokens=max_eval_tokens)
    baseline_before = _evaluate_bundle(baseline, tasks, tokenizer, suite, device, task_eval_count=task_eval_count, batch_size=batch_size, seq_len=seq_len, max_eval_tokens=max_eval_tokens)

    # The selected path is a used module in the aged source, not a new
    # unconnected layer.  Its route load is recorded for intervention audit.
    with torch.no_grad():
        probe = target_batch[0][: min(target_batch[0].size(0), task_eval_count)]
        _, _, aux = remora(probe, return_aux=True)
        route_load = aux["route_weights"][REMORA_TARGET_LAYER].mean(dim=(0, 1)).detach().cpu().tolist()

    old_expert = remora.blocks[REMORA_TARGET_LAYER].experts.experts[REMORA_TARGET_EXPERT]
    old_expert_parameters = count_parameters(old_expert)
    replacement = SwiGLUExpert(remora.cfg.bus_dim, remora.cfg.d_ff, remora.cfg.bus_dim).to(device)
    remora.replace_expert(REMORA_TARGET_LAYER, REMORA_TARGET_EXPERT, replacement)
    remora_scope = [f"blocks.{REMORA_TARGET_LAYER}.experts.experts.{REMORA_TARGET_EXPERT}"]
    freeze_all(remora)
    remora_selected = unfreeze_prefixes(remora, remora_scope)
    remora_before_surgery = _evaluate_bundle(remora, tasks, tokenizer, suite, device, task_eval_count=task_eval_count, batch_size=batch_size, seq_len=seq_len, max_eval_tokens=max_eval_tokens)
    remora_snapshot = parameter_snapshot(remora)
    remora_training = _local_train(
        remora,
        target_batch,
        replay_batches,
        base_batches,
        remora_selected,
        steps=steps,
        seed=seed + 4000,
        rehearsal_weight=rehearsal_weight,
        device=device,
    )
    remora_after = _evaluate_bundle(remora, tasks, tokenizer, suite, device, task_eval_count=task_eval_count, batch_size=batch_size, seq_len=seq_len, max_eval_tokens=max_eval_tokens)
    remora_update = changed_parameter_stats(remora_snapshot, remora)

    # The baseline is modified only by two rank-64 LoRA factors at its matched
    # feed-forward block.  Existing aged attention LoRA factors remain frozen.
    baseline_ff_prefix = f"blocks.{REMORA_TARGET_LAYER}.ff"
    baseline_wrapped = replace_linear_with_lora(
        baseline,
        matching_suffixes((f"blocks.{REMORA_TARGET_LAYER}.ff.0", f"blocks.{REMORA_TARGET_LAYER}.ff.2")),
        rank=BASELINE_LORA_RANK,
        alpha=BASELINE_LORA_ALPHA,
    )
    if set(baseline_wrapped) != {f"blocks.{REMORA_TARGET_LAYER}.ff.0", f"blocks.{REMORA_TARGET_LAYER}.ff.2"}:
        raise AssertionError(f"unexpected aged baseline repair modules: {baseline_wrapped}")
    freeze_all(baseline)
    baseline_selected = [
        name
        for name, _parameter in baseline.named_parameters()
        if name.startswith(baseline_ff_prefix) and (".lora_A" in name or ".lora_B" in name)
    ]
    for name, parameter in baseline.named_parameters():
        if name in baseline_selected:
            parameter.requires_grad = True
    baseline_before_surgery = _evaluate_bundle(baseline, tasks, tokenizer, suite, device, task_eval_count=task_eval_count, batch_size=batch_size, seq_len=seq_len, max_eval_tokens=max_eval_tokens)
    baseline_snapshot = parameter_snapshot(baseline)
    baseline_training = _local_train(
        baseline,
        target_batch,
        replay_batches,
        base_batches,
        baseline_selected,
        steps=steps,
        seed=seed + 4000,
        rehearsal_weight=rehearsal_weight,
        device=device,
    )
    baseline_after = _evaluate_bundle(baseline, tasks, tokenizer, suite, device, task_eval_count=task_eval_count, batch_size=batch_size, seq_len=seq_len, max_eval_tokens=max_eval_tokens)
    baseline_update = changed_parameter_stats(baseline_snapshot, baseline)

    output_checkpoints = {}
    if save_checkpoints:
        remora_path = ROOT / "checkpoints" / f"aged-surgery-remora-seed{seed}.pt"
        baseline_path = ROOT / "checkpoints" / f"aged-surgery-baseline-lora-seed{seed}.pt"
        torch.save({"schema": "remora-v1-aged-surgery-checkpoint", "model_kind": "remora", "seed": seed, "config": remora.cfg.to_dict(), "state_dict": {name: value.detach().cpu() for name, value in remora.state_dict().items()}}, remora_path)
        torch.save({"schema": "remora-v1-aged-surgery-checkpoint", "model_kind": "baseline_lora", "seed": seed, "config": baseline.cfg.to_dict(), "state_dict": {name: value.detach().cpu() for name, value in baseline.state_dict().items()}}, baseline_path)
        output_checkpoints = {"remora": str(remora_path), "baseline_lora": str(baseline_path)}

    previous_ids = [task.stage_id for task in tasks[:-1]]
    remora_retention = _mean_task_accuracy(remora_after, previous_ids) / max(_mean_task_accuracy(remora_before_surgery, previous_ids), 1e-9)
    baseline_retention = _mean_task_accuracy(baseline_after, previous_ids) / max(_mean_task_accuracy(baseline_before_surgery, previous_ids), 1e-9)
    return {
        "seed": seed,
        "aged_checkpoints": {"remora": str(remora_checkpoint), "baseline_lora": str(baseline_checkpoint)},
        "target_task": target_task.stage_id,
        "used_path": {
            "layer": REMORA_TARGET_LAYER,
            "expert": REMORA_TARGET_EXPERT,
            "route_load_before_surgery": route_load,
            "old_expert_parameters": old_expert_parameters,
        },
        "remora": {
            "before_aging_surgery": remora_before,
            "after_replacement_before_training": remora_before_surgery,
            "after_local_rehearsal": remora_after,
            "training": remora_training,
            "update": remora_update,
            "new_module_parameters": count_parameters(replacement),
            "newly_trained_parameters": remora_training["trainable_parameters"],
            "old_task_retention_ratio": remora_retention,
        },
        "baseline_lora": {
            "before_aged_repair": baseline_before,
            "after_adapter_insertion_before_training": baseline_before_surgery,
            "after_local_rehearsal": baseline_after,
            "training": baseline_training,
            "update": baseline_update,
            "new_adapter_parameters": baseline_training["trainable_parameters"],
            "existing_aged_attention_lora_frozen": True,
            "old_task_retention_ratio": baseline_retention,
        },
        "budget_comparison": {
            "remora_newly_trained_parameters": remora_training["trainable_parameters"],
            "baseline_newly_trained_parameters": baseline_training["trainable_parameters"],
            "absolute_difference": remora_training["trainable_parameters"] - baseline_training["trainable_parameters"],
            "relative_difference_to_remora": (remora_training["trainable_parameters"] - baseline_training["trainable_parameters"]) / max(remora_training["trainable_parameters"], 1),
            "same_steps": steps,
            "same_target_and_rehearsal_batches": True,
        },
        "output_checkpoints": output_checkpoints,
    }


def run(
    *,
    checkpoint_dir: str | Path = ROOT / "checkpoints",
    seeds: list[int] | tuple[int, ...] = (7, 19, 31),
    device_name: str = "auto",
    steps: int = 60,
    task_eval_count: int = 16,
    batch_size: int = 16,
    seq_len: int = 96,
    max_eval_tokens: int = 1536,
    rehearsal_examples: int = 16,
    rehearsal_weight: float = 1.0,
    curriculum_train_count: int = 64,
    curriculum_eval_count: int = 32,
    output: str | Path = ROOT / "results" / "aged-surgery-v1.json",
    save_checkpoints: bool = True,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("at least one seed is required")
    device = choose_device(device_name)
    checkpoint_dir = Path(checkpoint_dir)
    cfg = ModelConfig.from_json(ROOT / "configs" / "v0_tiny.json")
    tokenizer = ByteTokenizer(cfg.vocab_size)
    tasks, curriculum = build_lifetime_curriculum(train_count=curriculum_train_count, eval_count=curriculum_eval_count)
    suite = build_transfer_suite(ROOT, "/home/leo/tmp/wikitext-2-raw", tokenizer=tokenizer)
    task_width = _task_width(tasks, tokenizer)
    runs = []
    for seed in seeds:
        remora_checkpoint = checkpoint_dir / f"lifetime-compounding-remora_local_rehearsal-seed{seed}.pt"
        baseline_checkpoint = checkpoint_dir / f"lifetime-compounding-baseline_lora_rehearsal-seed{seed}.pt"
        if not remora_checkpoint.is_file() or not baseline_checkpoint.is_file():
            raise FileNotFoundError(f"missing aged pair for seed {seed}: {remora_checkpoint}, {baseline_checkpoint}")
        set_seed(seed)
        runs.append(_run_seed(
            remora_checkpoint=remora_checkpoint,
            baseline_checkpoint=baseline_checkpoint,
            seed=seed,
            device=device,
            tasks=tasks,
            tokenizer=tokenizer,
            suite=suite,
            task_width=task_width,
            steps=steps,
            task_eval_count=task_eval_count,
            batch_size=batch_size,
            seq_len=seq_len,
            max_eval_tokens=max_eval_tokens,
            rehearsal_examples=rehearsal_examples,
            rehearsal_weight=rehearsal_weight,
            save_checkpoints=save_checkpoints,
        ))
    previous_ids = [task.stage_id for task in tasks[:-1]]
    paired = []
    for row in runs:
        paired.append({
            "seed": row["seed"],
            "target_valid_accuracy_remora_minus_baseline": row["remora"]["after_local_rehearsal"]["tasks"]["T5"]["valid"]["teacher_forced_accuracy"] - row["baseline_lora"]["after_local_rehearsal"]["tasks"]["T5"]["valid"]["teacher_forced_accuracy"],
            "target_shifted_accuracy_remora_minus_baseline": row["remora"]["after_local_rehearsal"]["tasks"]["T5"]["shifted"]["teacher_forced_accuracy"] - row["baseline_lora"]["after_local_rehearsal"]["tasks"]["T5"]["shifted"]["teacher_forced_accuracy"],
            "target_unseen_accuracy_remora_minus_baseline": row["remora"]["after_local_rehearsal"]["tasks"]["T5"]["unseen"]["teacher_forced_accuracy"] - row["baseline_lora"]["after_local_rehearsal"]["tasks"]["T5"]["unseen"]["teacher_forced_accuracy"],
            "retention_ratio_remora_minus_baseline": row["remora"]["old_task_retention_ratio"] - row["baseline_lora"]["old_task_retention_ratio"],
            "wall_seconds_remora_minus_baseline": row["remora"]["training"]["wall_seconds"] - row["baseline_lora"]["training"]["wall_seconds"],
            "tokens_remora_minus_baseline": row["remora"]["training"]["total_assimilation_tokens"] - row["baseline_lora"]["training"]["total_assimilation_tokens"],
        })
    result = {
        "schema": "remora-v1-aged-surgery-result",
        "experiment_family": "AGED-SURGERY-001+",
        "mode": "EXPERIENCED_REMORA_USED_MODULE_REPLACEMENT_VS_AGED_TRANSFORMER_LOCAL_LORA_REPAIR",
        "device": str(device),
        "training": {
            "seeds": list(seeds),
            "steps": steps,
            "task_eval_count": task_eval_count,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "rehearsal_examples": rehearsal_examples,
            "rehearsal_weight": rehearsal_weight,
            "target_stage": "T5",
            "aged_after_stages": 5,
        },
        "parameter_accounting": {
            "remora_replacement": {"module": "blocks.1.experts.experts.0", "old_type": "ExpertMLP", "new_type": "SwiGLUExpert", "new_parameters": 111456},
            "baseline_repair": {"module": "blocks.1.ff.{0,2}", "type": "rank-64 LoRA", "new_parameters": 110592, "old_attention_lora_frozen": True},
            "mismatch_statement": "The newly trained surgery scopes differ by 864 parameters (0.775% of the Remora scope); both use the same steps, target examples, and replay batches.",
        },
        "curriculum": curriculum,
        "suite": suite.metadata,
        "runs": runs,
        "paired": paired,
        "labels": {
            "MEASURED": ["task accuracy on valid/shifted/unseen splits", "base text/code loss", "steps", "tokens", "wall time", "changed parameters", "route load"],
            "DERIVED": ["retention ratios", "paired differences", "budget differences"],
            "MODELED": ["training FLOPs"],
            "HYPOTHESIS": ["aged modular replacement preserves accumulated capability at bounded local cost"],
        },
        "promotion": {"state": "CONTROLLED_EXPERIMENT_ONLY", "promoted": False},
        "interpretation": "MEASURED ADVERSARIAL RESULT: experienced Remora surgery is judged against an aged Transformer with a near-equal trainable LoRA repair on the full lifetime task/interface matrix; no surgery win is inferred from lineage alone.",
    }
    write_json(output, result)
    if any(row["remora"]["old_task_retention_ratio"] < 0.8 for row in runs):
        for row in runs:
            if row["remora"]["old_task_retention_ratio"] < 0.8:
                record_failure(
                    ROOT,
                    "AGED-SURGERY-001",
                    cfg.to_dict(),
                    {"curriculum": curriculum, "target": "T5", "rehearsal_weight": rehearsal_weight},
                    row["seed"],
                    f"Remora old-task retention ratio after aged surgery was {row['remora']['old_task_retention_ratio']:.4f}",
                    "A used module replacement can still damage accumulated lifetime behavior despite replay when its surrounding representation is co-adapted.",
                    "Retest after port-preserving initialization, a wider affected-neighborhood repair, or consolidation before another replacement.",
                    runtime={"device": str(device), "mode": "aged_surgery"},
                )
    record_experiment(
        ROOT,
        "AGED-SURGERY-001",
        "After five lifetime stages, replacing a used Remora module should preserve prior capability with a bounded local scope and compare favorably to a near-equal-budget aged Transformer adapter repair.",
        "Replace the used layer-1 expert, train only the new Remora SwiGLU module, and compare it with rank-64 LoRA repair of the corresponding aged Transformer feed-forward block using identical target/replay batches.",
        "Remora retains prior valid/shifted/unseen capability and recovers T5 with no broader update or materially higher assimilation cost than the matched LoRA control.",
        "The Transformer adapter matches or beats Remora retention/transfer at the matched budget, shifted interfaces collapse, or Remora surgery regresses prior tasks below the 0.8 retention ratio.",
        f"python -m experiments.aged_surgery --seeds {' '.join(str(seed) for seed in seeds)} --steps {steps}",
        0,
        {"paired": paired, "parameter_accounting": result["parameter_accounting"]},
        result["interpretation"],
        "Use the full task/interface results to decide whether to improve replacement initialization/ports or proceed to a second Ship-of-Theseus generation chain.",
        hardware=runtime_context(device),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--task-eval-count", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--max-eval-tokens", type=int, default=1536)
    parser.add_argument("--rehearsal-examples", type=int, default=16)
    parser.add_argument("--rehearsal-weight", type=float, default=1.0)
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "results" / "aged-surgery-v1.json"))
    args = parser.parse_args()
    result = run(
        checkpoint_dir=args.checkpoint_dir,
        seeds=args.seeds,
        device_name=args.device,
        steps=args.steps,
        task_eval_count=args.task_eval_count,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_eval_tokens=args.max_eval_tokens,
        rehearsal_examples=args.rehearsal_examples,
        rehearsal_weight=args.rehearsal_weight,
        output=args.output,
        save_checkpoints=not args.no_checkpoints,
    )
    print(json.dumps({"interpretation": result["interpretation"], "paired": result["paired"]}, indent=2))


if __name__ == "__main__":
    main()
