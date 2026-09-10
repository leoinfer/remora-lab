from __future__ import annotations

"""Adversarial multi-lifetime comparison at the existing Remora-v0 scale.

This experiment keeps the old transfer benchmark intact and adds the missing
adversary: a conventional Transformer with a real, parameter-matched LoRA
adaptation path.  Both models are trained from scratch on the same local
text/code streams, then receive the same sequential concept/interface
curriculum.  Every stage records transfer, retention, update scope, and
compute-normalized learning measurements.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.lifetime_curriculum import LifetimeExample, LifetimeTask, build_lifetime_curriculum
from environments.transfer_suite import TransferSuite, build_transfer_suite
from remora.config import ModelConfig
from remora.data import sample_batch
from remora.ledger import record_experiment
from remora.metrics import evaluate_stream, model_summary
from remora.models import build_model
from remora.modules import (
    lora_parameter_names,
    matching_suffixes,
    replace_linear_with_lora,
)
from remora.tokenizer import ByteTokenizer
from remora.utils import (
    changed_parameter_stats,
    choose_device,
    count_parameters,
    freeze_all,
    parameter_snapshot,
    runtime_context,
    set_seed,
    write_json,
)
from remora.plot import write_line_svg

# The prior transfer implementation contains the validated fixed-shape task
# encoding and exact external-verifier scorer.  Reuse those tested routines;
# the new experiment owns the sequential training and accounting below.
from experiments.transfer_benchmark import (  # noqa: E402
    _encode_examples,
    _evaluate_tasks,
    _evaluate_stream_limited,
    _reference_recurrence,
    _response_loss,
)


LORA_RANK = 8
LORA_ALPHA = 8.0
LORA_TARGET_SUFFIXES = ("attention.qkv", "attention.out")
ADAPTATION_MODES = (
    "remora_local_rehearsal",
    "remora_local_target_only",
    "baseline_lora_rehearsal",
    "baseline_lora_target_only",
    "remora_frozen",
    "baseline_frozen",
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _encode_fixed(
    examples: list[LifetimeExample] | tuple[LifetimeExample, ...],
    tokenizer: ByteTokenizer,
    device: torch.device,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = []
    for example in examples:
        prompt_ids = tokenizer.encode(example.prompt)
        sequence = prompt_ids + tokenizer.encode(example.response)
        if len(sequence) - 1 > width:
            raise ValueError(f"task example exceeds fixed width {width}: {example.task_id}")
        encoded.append((sequence[:-1], sequence[1:], len(prompt_ids)))
    if not encoded:
        raise ValueError("cannot encode an empty lifetime split")
    inputs = torch.zeros(len(encoded), width, dtype=torch.long, device=device)
    targets = torch.zeros_like(inputs)
    response_mask = torch.zeros(len(encoded), width, dtype=torch.float32, device=device)
    for row, (input_ids, target_ids, prompt_length) in enumerate(encoded):
        inputs[row, : len(input_ids)] = torch.tensor(input_ids, dtype=torch.long, device=device)
        targets[row, : len(target_ids)] = torch.tensor(target_ids, dtype=torch.long, device=device)
        response_start = max(prompt_length - 1, 0)
        response_mask[row, response_start : len(target_ids)] = 1.0
    return inputs, targets, response_mask


def _task_width(tasks: tuple[LifetimeTask, ...], tokenizer: ByteTokenizer) -> int:
    examples = [example for task in tasks for split in ("train", "valid", "shifted", "unseen") for example in getattr(task, split)]
    return max(len(tokenizer.encode(example.prompt + example.response)) - 1 for example in examples)


def _base_evaluation(
    model: nn.Module,
    suite: TransferSuite,
    *,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    max_eval_tokens: int,
) -> dict:
    _sync(device)
    started = time.perf_counter()
    streams = {
        domain: _evaluate_stream_limited(
            model,
            suite.valid_streams[domain],
            batch_size,
            seq_len,
            device,
            max_eval_tokens,
        )
        for domain in ("text", "code")
    }
    _sync(device)
    return {
        "streams": streams,
        "wall_seconds": time.perf_counter() - started,
        "recurrent_scan": "fixed_shape_stream_evaluation",
    }


def _all_task_evaluation(
    model: nn.Module,
    tasks: tuple[LifetimeTask, ...],
    tokenizer: ByteTokenizer,
    device: torch.device,
    *,
    task_eval_count: int,
    free_running_count: int = 0,
) -> dict:
    result = {}
    for task in tasks:
        result[task.stage_id] = {
            split: _evaluate_tasks(
                model,
                list(getattr(task, split)[:task_eval_count]),
                tokenizer,
                device,
                free_running_audit_count=(free_running_count if split == "valid" else 0),
            )
            for split in ("valid", "shifted", "unseen")
        }
    return result


def _current_task_evaluation(
    model: nn.Module,
    task: LifetimeTask,
    tokenizer: ByteTokenizer,
    device: torch.device,
    *,
    task_eval_count: int,
) -> dict:
    return {
        split: _evaluate_tasks(
            model,
            list(getattr(task, split)[:task_eval_count]),
            tokenizer,
            device,
            free_running_audit_count=(min(4, task_eval_count) if split == "valid" else 0),
        )
        for split in ("valid", "shifted", "unseen")
    }


def _train_base(
    model: nn.Module,
    suite: TransferSuite,
    *,
    seed: int,
    device: torch.device,
    steps: int,
    batch_size: int,
    seq_len: int,
    eval_every: int,
    max_eval_tokens: int,
) -> tuple[list[dict], list[dict]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    generator = torch.Generator(device="cpu").manual_seed(seed + 401)
    history = []
    evaluations = [{
        "step": 0,
        "tokens_seen": 0,
        "evaluation": _base_evaluation(
            model,
            suite,
            device=device,
            batch_size=batch_size,
            seq_len=seq_len,
            max_eval_tokens=max_eval_tokens,
        ),
    }]
    domains = ("text", "code")
    for step in range(1, steps + 1):
        domain = domains[(step - 1) % len(domains)]
        started = time.perf_counter()
        model.train()
        x, y = sample_batch(suite.train_streams[domain], batch_size, seq_len, device, generator)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        _sync(device)
        optimizer.step()
        _sync(device)
        history.append({
            "step": step,
            "domain": domain,
            "train_loss": float(loss.detach()),
            "grad_norm": grad_norm,
            "tokens": batch_size * seq_len,
            "wall_seconds": time.perf_counter() - started,
        })
        if step % eval_every == 0 or step == steps:
            evaluations.append({
                "step": step,
                "tokens_seen": step * batch_size * seq_len,
                "evaluation": _base_evaluation(
                    model,
                    suite,
                    device=device,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    max_eval_tokens=max_eval_tokens,
                ),
            })
    return history, evaluations


def _configure_adaptation(model: nn.Module, model_kind: str, mode: str) -> list[str]:
    freeze_all(model)
    if mode.endswith("frozen"):
        return []
    if model_kind == "remora":
        names = []
        for index in range(len(model.blocks)):
            names.extend(
                name
                for name, parameter in model.named_parameters()
                if name.startswith(f"blocks.{index}.plastic.")
            )
        if not names:
            raise ValueError("Remora plastic islands were not found")
        for name, parameter in model.named_parameters():
            if name in names:
                parameter.requires_grad = True
        return names
    if model_kind == "baseline_lora":
        # The wrapper freezes its base, but freeze_all above also freezes A/B;
        # explicitly reopen only the two factor matrices.
        for name, parameter in model.named_parameters():
            if ".lora_A" in name or ".lora_B" in name:
                parameter.requires_grad = True
        names = lora_parameter_names(model)
        if not names:
            raise ValueError("LoRA factors were not found")
        return names
    raise ValueError(f"unknown adaptation model kind: {model_kind}")


def _selected_update_stats(
    before: dict[str, torch.Tensor],
    model: nn.Module,
    selected_names: list[str],
    *,
    atol: float = 1e-12,
) -> dict:
    selected = set(selected_names)
    changed = 0
    total = 0
    l1 = 0.0
    for name, parameter in model.named_parameters():
        if name not in selected or name not in before:
            continue
        delta = (parameter.detach().cpu() - before[name]).abs()
        total += int(delta.numel())
        changed += int((delta > atol).sum())
        l1 += float(delta.sum())
    return {
        "changed_parameters": changed,
        "total_selected_parameters": total,
        "changed_fraction_of_selected": changed / max(total, 1),
        "l1": l1,
    }


def _full_loss(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    _, loss = model(batch[0], batch[1])
    return loss


def _stats(values: list[float]) -> dict:
    if not values:
        return {"values": [], "mean": None, "std": None}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
    return {"values": values, "mean": mean, "std": math.sqrt(variance)}


def _checkpoint(model: nn.Module, path: Path, *, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**metadata, "state_dict": _cpu_state(model)}, path)


def _run_arm(
    *,
    base_state: dict[str, torch.Tensor],
    model_kind: str,
    mode: str,
    cfg: ModelConfig,
    suite: TransferSuite,
    tasks: tuple[LifetimeTask, ...],
    tokenizer: ByteTokenizer,
    device: torch.device,
    seed: int,
    stage_steps: int,
    eval_every: int,
    batch_size: int,
    seq_len: int,
    max_eval_tokens: int,
    task_eval_count: int,
    rehearsal_weight: float,
    rehearsal_examples: int,
    task_width: int,
    stage_count: int,
    free_running_count: int,
    save_path: Path | None,
    remora_bus_mode: str = "shared",
) -> dict:
    active_tasks = tasks[:stage_count]
    if model_kind in {"baseline", "baseline_lora"}:
        model = build_model("baseline", cfg)
        model.load_state_dict(base_state)
    else:
        model = build_model("remora", cfg, bus_mode=remora_bus_mode)
        model.load_state_dict(base_state)
    if model_kind == "baseline_lora":
        replaced = replace_linear_with_lora(
            model,
            matching_suffixes(LORA_TARGET_SUFFIXES),
            rank=LORA_RANK,
            alpha=LORA_ALPHA,
        )
        if replaced != [
            f"blocks.{layer}.attention.{projection}"
            for layer in range(cfg.n_layers)
            for projection in ("qkv", "out")
            ]:
            raise AssertionError(f"unexpected LoRA targets: {replaced}")
    model = model.to(device)
    selected_names = _configure_adaptation(model, model_kind, mode)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = None
    if trainable_parameters:
        optimizer = torch.optim.AdamW(trainable_parameters, lr=0.003, betas=(0.9, 0.95), weight_decay=0.0)

    # The same fixed-width task batches are used by every arm.  This makes
    # supervised/rehearsal token counts and scan-shape costs directly
    # comparable instead of letting padding or shape compilation vary by arm.
    task_batches = {
        task.stage_id: _encode_fixed(list(task.train), tokenizer, device, task_width)
        for task in active_tasks
    }
    replay_task_batches = {
        task.stage_id: _encode_fixed(
            list(task.train[: min(rehearsal_examples, len(task.train))]), tokenizer, device, task_width
        )
        for task in active_tasks
    }
    base_generator = torch.Generator(device="cpu").manual_seed(seed + 17000)
    base_rehearsal_batches = [
        sample_batch(suite.train_streams[domain], batch_size, seq_len, device, base_generator)
        for domain in ("text", "code")
    ]
    target_tokens_per_step = int(task_batches[tasks[0].stage_id][0].numel())
    supervised_target_tokens_per_step = int(task_batches[tasks[0].stage_id][2].sum().item())
    stage_results = []
    arm_started = time.perf_counter()
    previous_task_ids: list[str] = []
    for stage_index, task in enumerate(active_tasks):
        target_batch = task_batches[task.stage_id]
        previous_task_ids = [item.stage_id for item in active_tasks[:stage_index]]
        old_task_batches = [
            replay_task_batches[task_id]
            for task_id in previous_task_ids
        ]
        before = parameter_snapshot(model)
        _sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        pre_started = time.perf_counter()
        pre_tasks = _all_task_evaluation(
            model,
            active_tasks,
            tokenizer,
            device,
            task_eval_count=task_eval_count,
            free_running_count=0,
        )
        pre_base = _base_evaluation(
            model,
            suite,
            device=device,
            batch_size=batch_size,
            seq_len=seq_len,
            max_eval_tokens=max_eval_tokens,
        )
        _sync(device)
        evaluations = [{
            "step": 0,
            "tokens_seen": 0,
            "evaluation_kind": "pre_stage_all_tasks",
            "tasks": pre_tasks,
            "base": pre_base,
            "wall_seconds": time.perf_counter() - pre_started,
        }]
        target_loss_history: list[float] = []
        stage_started = time.perf_counter()
        steps_completed = 0
        replay_task_tokens_per_step = sum(int(batch[0].numel()) for batch in old_task_batches)
        replay_base_tokens_per_step = sum(int(batch[0].numel()) for batch in base_rehearsal_batches)
        rehearsal_enabled = "rehearsal" in mode
        total_tokens_per_step = target_tokens_per_step
        if rehearsal_enabled:
            total_tokens_per_step += replay_task_tokens_per_step + replay_base_tokens_per_step
        checkpoint_steps = sorted(set([step for step in range(eval_every, stage_steps + 1, eval_every)] + [stage_steps]))
        for step in range(1, stage_steps + 1):
            step_started = time.perf_counter()
            if optimizer is not None:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                target_loss = _response_loss(model, target_batch)
                loss = target_loss
                if rehearsal_enabled:
                    replay_losses = [_response_loss(model, batch) for batch in old_task_batches]
                    replay_losses.extend(_full_loss(model, batch) for batch in base_rehearsal_batches)
                    if replay_losses:
                        loss = loss + rehearsal_weight * torch.stack(replay_losses).mean()
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0))
                _sync(device)
                optimizer.step()
                _sync(device)
                target_loss_history.append(float(target_loss.detach()))
            else:
                grad_norm = 0.0
            steps_completed = step
            if step in checkpoint_steps:
                current = _current_task_evaluation(
                    model,
                    task,
                    tokenizer,
                    device,
                    task_eval_count=task_eval_count,
                )
                evaluations.append({
                    "step": step,
                    "tokens_seen": step * total_tokens_per_step,
                    "new_task_tokens_seen": step * target_tokens_per_step,
                    "evaluation_kind": "current_task_checkpoint",
                    "current_task": current,
                    "target_train_loss": target_loss_history[-1] if target_loss_history else None,
                    "grad_norm": grad_norm,
                    "step_wall_seconds": time.perf_counter() - step_started,
                })
        after_tasks = _all_task_evaluation(
            model,
            active_tasks,
            tokenizer,
            device,
            task_eval_count=task_eval_count,
            free_running_count=free_running_count,
        )
        after_base = _base_evaluation(
            model,
            suite,
            device=device,
            batch_size=batch_size,
            seq_len=seq_len,
            max_eval_tokens=max_eval_tokens,
        )
        evaluations.append({
            "step": stage_steps,
            "tokens_seen": stage_steps * total_tokens_per_step,
            "new_task_tokens_seen": stage_steps * target_tokens_per_step,
            "evaluation_kind": "post_stage_all_tasks",
            "tasks": after_tasks,
            "base": after_base,
        })
        _sync(device)
        stage_wall_seconds = time.perf_counter() - stage_started
        changed = changed_parameter_stats(before, model)
        selected_changed = _selected_update_stats(before, model, selected_names)
        current_before = pre_tasks[task.stage_id]["valid"]["teacher_forced_accuracy"]
        current_after = after_tasks[task.stage_id]["valid"]["teacher_forced_accuracy"]
        threshold_step = None
        for evaluation in evaluations:
            if evaluation.get("evaluation_kind") == "current_task_checkpoint":
                accuracy = evaluation["current_task"]["valid"]["teacher_forced_accuracy"]
                if accuracy >= task.threshold:
                    threshold_step = evaluation["step"]
                    break
        old_task_deltas = {
            old_task.stage_id: after_tasks[old_task.stage_id]["valid"]["teacher_forced_accuracy"]
            - pre_tasks[old_task.stage_id]["valid"]["teacher_forced_accuracy"]
            for old_task in tasks[:stage_index]
        }
        peak_vram = None
        if device.type == "cuda":
            peak_vram = int(torch.cuda.max_memory_allocated(device))
        optimization_steps = steps_completed if optimizer is not None else 0
        measured_tokens = steps_completed * total_tokens_per_step if optimizer is not None else 0
        stage_result = {
            "stage_index": stage_index,
            "stage_id": task.stage_id,
            "concept_id": task.concept_id,
            "mode": mode,
            "rehearsal_enabled": rehearsal_enabled,
            "rehearsal_weight": rehearsal_weight,
            "rehearsal_task_ids": previous_task_ids,
            "scheduled_steps": steps_completed,
            "optimization_steps": optimization_steps,
            "new_task_tokens": optimization_steps * target_tokens_per_step,
            "supervised_new_response_tokens": optimization_steps * supervised_target_tokens_per_step,
            "rehearsal_tokens": optimization_steps * (replay_task_tokens_per_step + replay_base_tokens_per_step) if rehearsal_enabled else 0,
            "total_model_tokens": measured_tokens,
            "rehearsal_token_ratio": (
                (replay_task_tokens_per_step + replay_base_tokens_per_step) / max(total_tokens_per_step, 1)
                if rehearsal_enabled else 0.0
            ),
            "wall_seconds": stage_wall_seconds,
            "tokens_per_second": measured_tokens / max(stage_wall_seconds, 1e-9),
            "peak_vram_bytes": peak_vram,
            "trainable_parameters": count_parameters(model, trainable_only=True),
            "total_parameters": count_parameters(model),
            "changed_parameters": changed,
            "changed_trainable_island": selected_changed,
            "modeled_full_forward_training_flops": 6 * count_parameters(model) * measured_tokens,
            "modeled_trainable_update_flops": 6 * max(count_parameters(model, trainable_only=True), 1) * measured_tokens,
            "current_task_primary_accuracy_before": current_before,
            "current_task_primary_accuracy_after": current_after,
            "current_task_primary_gain": current_after - current_before,
            "shifted_accuracy_after": after_tasks[task.stage_id]["shifted"]["teacher_forced_accuracy"],
            "unseen_accuracy_after": after_tasks[task.stage_id]["unseen"]["teacher_forced_accuracy"],
            "primary_shifted_gap_after": after_tasks[task.stage_id]["valid"]["teacher_forced_accuracy"] - after_tasks[task.stage_id]["shifted"]["teacher_forced_accuracy"],
            "primary_unseen_gap_after": after_tasks[task.stage_id]["valid"]["teacher_forced_accuracy"] - after_tasks[task.stage_id]["unseen"]["teacher_forced_accuracy"],
            "threshold": task.threshold,
            "threshold_step": threshold_step,
            "threshold_new_task_tokens": (threshold_step * target_tokens_per_step if threshold_step is not None else None),
            "old_task_primary_deltas": old_task_deltas,
            "base_valid_loss_before": {domain: pre_base["streams"][domain]["loss"] for domain in ("text", "code")},
            "base_valid_loss_after": {domain: after_base["streams"][domain]["loss"] for domain in ("text", "code")},
            "evaluations": evaluations,
        }
        stage_results.append(stage_result)
    if save_path is not None:
        _checkpoint(
            model,
            save_path,
            metadata={
                "schema": "remora-v1-lifetime-compounding-checkpoint",
                "model_kind": model_kind,
                "adaptation_mode": mode,
                "seed": seed,
                "config": cfg.to_dict(),
                "trainable_parameters": count_parameters(model, trainable_only=True),
            },
        )
    result = {
        "model_kind": model_kind,
        "mode": mode,
        "seed": seed,
        "summary": model_summary(model),
        "lora": {
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "target_suffixes": list(LORA_TARGET_SUFFIXES),
            "targeted_modules": [
                f"blocks.{layer}.attention.{projection}"
                for layer in range(cfg.n_layers)
                for projection in ("qkv", "out")
            ] if model_kind == "baseline_lora" else [],
        },
        "selected_parameter_names": selected_names,
        "stage_results": stage_results,
        "wall_seconds": time.perf_counter() - arm_started,
        "checkpoint": str(save_path) if save_path is not None else None,
        "bus_mode": remora_bus_mode if model_kind == "remora" else None,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _base_result(model_kind: str, seed: int, model: nn.Module, history: list[dict], evaluations: list[dict], checkpoint: Path) -> dict:
    return {
        "model_kind": model_kind,
        "seed": seed,
        "summary": model_summary(model),
        "history": history,
        "evaluations": evaluations,
        "checkpoint": str(checkpoint),
        "scratch_trained": True,
    }


def _aggregate(runs: list[dict]) -> dict:
    aggregate = {"seeds": [run["seed"] for run in runs], "arms": {}, "paired": {}}
    for arm in ADAPTATION_MODES:
        arm_runs = [run["arms"][arm] for run in runs]
        stage_summaries = []
        for index, _task in enumerate(runs[0]["curriculum_stage_ids"]):
            current = [item["stage_results"][index]["current_task_primary_accuracy_after"] for item in arm_runs]
            shifted = [item["stage_results"][index]["shifted_accuracy_after"] for item in arm_runs]
            unseen = [item["stage_results"][index]["unseen_accuracy_after"] for item in arm_runs]
            retention = [
                sum(item["stage_results"][index]["old_task_primary_deltas"].values()) / max(len(item["stage_results"][index]["old_task_primary_deltas"]), 1)
                if item["stage_results"][index]["old_task_primary_deltas"] else 0.0
                for item in arm_runs
            ]
            threshold_steps = [item["stage_results"][index]["threshold_step"] for item in arm_runs]
            threshold_steps_present = [value for value in threshold_steps if value is not None]
            stage_summaries.append({
                "stage_id": runs[0]["curriculum_stage_ids"][index],
                "primary_accuracy_after": _stats(current),
                "shifted_accuracy_after": _stats(shifted),
                "unseen_accuracy_after": _stats(unseen),
                "mean_old_task_primary_delta": _stats(retention),
                "threshold_step": _stats(threshold_steps_present),
                "threshold_reached_count": len(threshold_steps_present),
                "wall_seconds": _stats([item["stage_results"][index]["wall_seconds"] for item in arm_runs]),
                "new_task_tokens": _stats([item["stage_results"][index]["new_task_tokens"] for item in arm_runs]),
                "modeled_full_forward_training_flops": _stats([item["stage_results"][index]["modeled_full_forward_training_flops"] for item in arm_runs]),
                "changed_fraction_total": _stats([item["stage_results"][index]["changed_parameters"]["changed_fraction"] for item in arm_runs]),
                "changed_fraction_selected": _stats([item["stage_results"][index]["changed_trainable_island"]["changed_fraction_of_selected"] for item in arm_runs]),
                "base_text_loss_after": _stats([item["stage_results"][index]["base_valid_loss_after"]["text"] for item in arm_runs]),
                "base_code_loss_after": _stats([item["stage_results"][index]["base_valid_loss_after"]["code"] for item in arm_runs]),
            })
        aggregate["arms"][arm] = {"stage_summaries": stage_summaries}
    for remora_arm, baseline_arm in (("remora_local_rehearsal", "baseline_lora_rehearsal"), ("remora_local_target_only", "baseline_lora_target_only")):
        paired_stages = []
        for index, stage_id in enumerate(runs[0]["curriculum_stage_ids"]):
            remora = [run["arms"][remora_arm]["stage_results"][index] for run in runs]
            baseline = [run["arms"][baseline_arm]["stage_results"][index] for run in runs]
            paired_stages.append({
                "stage_id": stage_id,
                "comparison": "DERIVED paired Remora minus Transformer-LoRA",
                "primary_accuracy_delta": _stats([
                    left["current_task_primary_accuracy_after"] - right["current_task_primary_accuracy_after"]
                    for left, right in zip(remora, baseline)
                ]),
                "shifted_accuracy_delta": _stats([
                    left["shifted_accuracy_after"] - right["shifted_accuracy_after"]
                    for left, right in zip(remora, baseline)
                ]),
                "wall_seconds_delta": _stats([
                    left["wall_seconds"] - right["wall_seconds"] for left, right in zip(remora, baseline)
                ]),
                "threshold_step_delta": _stats([
                    float(left["threshold_step"] - right["threshold_step"])
                    for left, right in zip(remora, baseline)
                    if left["threshold_step"] is not None and right["threshold_step"] is not None
                ]),
                "changed_fraction_delta": _stats([
                    left["changed_parameters"]["changed_fraction"] - right["changed_parameters"]["changed_fraction"]
                    for left, right in zip(remora, baseline)
                ]),
            })
        aggregate["paired"][f"{remora_arm}_vs_{baseline_arm}"] = paired_stages
    return aggregate


def run(
    wiki_root: str | Path,
    *,
    seeds: list[int] | tuple[int, ...] = (7,),
    device_name: str = "auto",
    base_steps: int = 120,
    stage_steps: int = 60,
    base_eval_every: int = 60,
    stage_eval_every: int = 20,
    stage_count: int = 5,
    free_running_count: int = 2,
    batch_size: int = 16,
    seq_len: int = 96,
    max_eval_tokens: int = 1536,
    task_eval_count: int = 16,
    curriculum_train_count: int = 64,
    curriculum_eval_count: int = 32,
    rehearsal_weight: float = 1.0,
    rehearsal_examples: int = 16,
    output: str | Path | None = None,
    record_ledger: bool = True,
    save_checkpoints: bool = True,
) -> dict:
    if not seeds:
        raise ValueError("at least one seed is required")
    if base_steps <= 0 or stage_steps <= 0 or base_eval_every <= 0 or stage_eval_every <= 0:
        raise ValueError("training steps and eval intervals must be positive")
    if rehearsal_examples <= 0:
        raise ValueError("rehearsal_examples must be positive")
    if not 1 <= stage_count <= 5:
        raise ValueError("stage_count must be between 1 and 5")
    if free_running_count < 0:
        raise ValueError("free_running_count cannot be negative")
    if max_eval_tokens < batch_size * seq_len:
        raise ValueError("max_eval_tokens must cover one full fixed stream evaluation batch")
    device = choose_device(device_name)
    cfg = ModelConfig.from_json(ROOT / "configs" / "v0_tiny.json")
    tokenizer = ByteTokenizer(cfg.vocab_size)
    suite = build_transfer_suite(ROOT, wiki_root, tokenizer=tokenizer)
    tasks, curriculum_metadata = build_lifetime_curriculum(
        train_count=curriculum_train_count,
        eval_count=curriculum_eval_count,
    )
    task_width = _task_width(tasks, tokenizer)
    runs = []
    for seed in seeds:
        seed_run = {"seed": seed, "bases": {}, "arms": {}, "curriculum_stage_ids": [task.stage_id for task in tasks[:stage_count]]}
        base_states = {}
        for model_kind in ("remora", "baseline"):
            set_seed(seed)
            model = build_model(model_kind, cfg).to(device)
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
            state = _cpu_state(model)
            base_states[model_kind] = state
            checkpoint = ROOT / "checkpoints" / f"lifetime-compounding-base-{model_kind}-seed{seed}.pt"
            if save_checkpoints:
                _checkpoint(
                    model,
                    checkpoint,
                    metadata={
                        "schema": "remora-v1-lifetime-compounding-base-checkpoint",
                        "model_kind": model_kind,
                        "seed": seed,
                        "config": cfg.to_dict(),
                        "scratch_trained": True,
                        "suite_metadata": suite.metadata,
                    },
                )
            seed_run["bases"][model_kind] = _base_result(model_kind, seed, model, history, evaluations, checkpoint)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        arm_specs = (
            ("remora_local_rehearsal", "remora", "remora_local_rehearsal"),
            ("remora_local_target_only", "remora", "remora_local_target_only"),
            ("baseline_lora_rehearsal", "baseline_lora", "baseline_lora_rehearsal"),
            ("baseline_lora_target_only", "baseline_lora", "baseline_lora_target_only"),
            ("remora_frozen", "remora", "remora_frozen"),
            ("baseline_frozen", "baseline", "baseline_frozen"),
        )
        for arm_index, (arm_name, model_kind, mode) in enumerate(arm_specs):
            set_seed(seed + 1000 + arm_index)
            checkpoint = ROOT / "checkpoints" / f"lifetime-compounding-{arm_name}-seed{seed}.pt"
            seed_run["arms"][arm_name] = _run_arm(
                base_state=base_states["remora" if model_kind == "remora" else "baseline"],
                model_kind=model_kind,
                mode=mode,
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
                rehearsal_weight=rehearsal_weight,
                rehearsal_examples=rehearsal_examples,
                task_width=task_width,
                stage_count=stage_count,
                free_running_count=free_running_count,
                save_path=checkpoint if save_checkpoints else None,
            )
        runs.append(seed_run)
    aggregate = _aggregate(runs)
    result = {
        "schema": "remora-v1-lifetime-compounding-result",
        "experiment_family": "LIFETIME-COMPOUNDING-001+",
        "mode": "FIXED_SCALE_SCRATCH_MULTI_LIFETIME_REMORA_VS_MATCHED_TRANSFORMER_LORA",
        "device": str(device),
        "config": cfg.to_dict(),
        "training": {
            "seeds": list(seeds),
            "base_steps": base_steps,
            "stage_steps": stage_steps,
            "base_eval_every": base_eval_every,
            "stage_eval_every": stage_eval_every,
            "stage_count": stage_count,
            "free_running_count": free_running_count,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "max_eval_tokens": max_eval_tokens,
            "task_eval_count": task_eval_count,
            "rehearsal_weight": rehearsal_weight,
            "rehearsal_examples": rehearsal_examples,
            "base_domains": ["text", "code"],
            "adaptation_optimizer": "AdamW(lr=0.003, betas=(0.9,0.95), weight_decay=0)",
            "base_optimizer": "AdamW(lr=0.0003, betas=(0.9,0.95), weight_decay=0.01)",
            "task_batch_width": task_width,
        },
        "parameter_accounting": {
            "remora_total_parameters": count_parameters(build_model("remora", cfg)),
            "remora_plastic_trainable_parameters": sum(
                parameter.numel()
                for name, parameter in build_model("remora", cfg).named_parameters()
                if ".plastic." in name
            ),
            "baseline_total_parameters_before_lora": count_parameters(build_model("baseline", cfg)),
            "baseline_lora_rank": LORA_RANK,
            "baseline_lora_target_suffixes": list(LORA_TARGET_SUFFIXES),
            "baseline_lora_trainable_parameters": cfg.n_layers * LORA_RANK * (
                (cfg.d_model + 3 * cfg.d_model) + (cfg.d_model + cfg.d_model)
            ),
            "mismatch_statement": "base architectures are the existing matched v0 pair; LoRA factors add adaptation storage but trainable adaptation parameters exactly match Remora plastic islands for the default config",
        },
        "curriculum": curriculum_metadata,
        "suite": suite.metadata,
        "runs": runs,
        "aggregate": aggregate,
        "labels": {
            "MEASURED": ["loss", "accuracy", "wall_seconds", "peak_vram_bytes", "changed_parameters", "tokens"],
            "DERIVED": ["retention", "forward_transfer", "backward_transfer", "paired_differences", "learning_velocity"],
            "MODELED": ["modeled_full_forward_training_flops", "modeled_trainable_update_flops"],
            "ESTIMATED": [],
            "HYPOTHESIS": ["age_compounding", "experience_to_intuition"],
            "EXTERNAL": ["deterministic curriculum oracle", "held-out stream source hashes"],
        },
        "promotion": {"state": "CONTROLLED_EXPERIMENT_ONLY", "promoted": False},
    }
    paired = aggregate["paired"]["remora_local_rehearsal_vs_baseline_lora_rehearsal"]
    later = [item["primary_accuracy_delta"]["mean"] for item in paired if item["primary_accuracy_delta"]["mean"] is not None]
    if later and any(value > 0.0 for value in later):
        interpretation = "MEASURED CONDITIONAL RESULT: at least one sequential stage favors Remora on primary exact accuracy; compounding, shifted-interface, wall-time, and retention gates remain the authority."
    else:
        interpretation = "MEASURED ADVERSARIAL RESULT: the sequential primary-accuracy comparison does not favor Remora at the recorded stages; no superiority claim is made."
    result["interpretation"] = interpretation
    if output:
        write_json(output, result)
        output_path = Path(output)
        write_line_svg(
            output_path.with_name(output_path.stem + "-primary-accuracy.svg"),
            {
                arm: [stage["primary_accuracy_after"]["mean"] or 0.0 for stage in aggregate["arms"][arm]["stage_summaries"]]
                for arm in ("remora_local_rehearsal", "baseline_lora_rehearsal", "remora_frozen", "baseline_frozen")
            },
            title="LIFETIME-COMPOUNDING-001 mean primary exact accuracy by age",
        )
        write_line_svg(
            output_path.with_name(output_path.stem + "-wall-seconds.svg"),
            {
                arm: [stage["wall_seconds"]["mean"] or 0.0 for stage in aggregate["arms"][arm]["stage_summaries"]]
                for arm in ("remora_local_rehearsal", "baseline_lora_rehearsal")
            },
            title="LIFETIME-COMPOUNDING-001 mean stage wall time",
        )
    if record_ledger:
        record_experiment(
            ROOT,
            "LIFETIME-COMPOUNDING-001",
            "Remora's local modular path should compound an age-dependent learning/retention advantage over a matched conventional Transformer with LoRA when both receive sequential related and new concepts.",
            "Preregister and run scratch-trained Remora, Transformer-LoRA, target-only, rehearsal, and frozen controls over five disjoint multi-interface lifetime stages at fixed v0 scale.",
            "Later related parity stages require fewer examples/steps or wall seconds at a fixed threshold, shifted/unseen transfer and retention remain useful, and the result survives matched trainable-parameter accounting.",
            "LoRA+rehearsal matches or beats Remora at equal trainable budget; any apparent gain disappears after wall/compute normalization; later related stages do not become easier; shifted interfaces collapse; frozen controls match; or staged surgery later requires broad retraining.",
            f"python -m experiments.lifetime_compounding --wiki-root {wiki_root} --seeds {' '.join(str(seed) for seed in seeds)} --base-steps {base_steps} --stage-steps {stage_steps} --stage-eval-every {stage_eval_every}",
            0,
            {"aggregate": aggregate, "parameter_accounting": result["parameter_accounting"], "curriculum": curriculum_metadata},
            interpretation,
            "Use the strongest counterevidence to choose aged surgery, bus ablation, consolidation, and one real resurrection test before any scaling decision.",
            hardware=runtime_context(device),
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-root", default="/home/leo/tmp/wikitext-2-raw")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--base-steps", type=int, default=120)
    parser.add_argument("--stage-steps", type=int, default=60)
    parser.add_argument("--base-eval-every", type=int, default=60)
    parser.add_argument("--stage-eval-every", type=int, default=20)
    parser.add_argument("--stage-count", type=int, default=5)
    parser.add_argument("--free-running-count", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--max-eval-tokens", type=int, default=1536)
    parser.add_argument("--task-eval-count", type=int, default=16)
    parser.add_argument("--curriculum-train-count", type=int, default=64)
    parser.add_argument("--curriculum-eval-count", type=int, default=32)
    parser.add_argument("--rehearsal-weight", type=float, default=1.0)
    parser.add_argument("--rehearsal-examples", type=int, default=16)
    parser.add_argument("--skip-ledger", action="store_true")
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "results" / "lifetime-compounding.json"))
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
        free_running_count=args.free_running_count,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_eval_tokens=args.max_eval_tokens,
        task_eval_count=args.task_eval_count,
        curriculum_train_count=args.curriculum_train_count,
        curriculum_eval_count=args.curriculum_eval_count,
        rehearsal_weight=args.rehearsal_weight,
        rehearsal_examples=args.rehearsal_examples,
        output=args.output,
        record_ledger=not args.skip_ledger,
        save_checkpoints=not args.no_checkpoints,
    )
    print(json.dumps({"interpretation": result["interpretation"], "aggregate": result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
