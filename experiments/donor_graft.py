from __future__ import annotations

"""Aged Remora Qwen-organ graft with foreign-weight ablations.

This is intentionally a local utility experiment, not a Qwen teacher run.  A
post-lifetime Remora checkpoint is modified at one used expert path.  The
Qwen shared expert is frozen behind low-rank ports, and the actual donor core
is compared with shuffled, random, and zero controls while the evaluator and
adaptation budget stay fixed.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.synthetic_corpus import build_heldout_corpus, build_target_corpus
from remora.config import ModelConfig
from remora.data import encode_stream, sample_batch
from remora.donors.graft import QwenSharedExpertGraft, make_donor_variant
from remora.donors.payload import QwenSharedExpertOrgan, load_payload
from remora.metrics import model_summary
from remora.ledger import record_experiment
from remora.models import build_model
from remora.modules import ExpertMLP, SwiGLUExpert
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

from experiments.transfer_benchmark import _evaluate_stream_limited


TRAINING_ARMS = (
    "native_frozen",
    "native_expert_local",
    "native_random_expert_local",
    "native_swiglu_local",
    "donor_actual_local",
    "donor_shuffled_local",
    "donor_random_local",
)


def _payload_storage_accounting(tensors: dict[str, torch.Tensor]) -> dict[str, int]:
    """Count source payload storage separately from the runtime FP32 organ."""

    source_bytes = sum(int(tensor.numel() * tensor.element_size()) for tensor in tensors.values())
    parameter_count = sum(int(tensor.numel()) for tensor in tensors.values())
    # Qwen's selected payload is BF16; QwenSharedExpertOrgan intentionally
    # promotes it to FP32 for the bounded standalone harness.  This is a
    # storage/precision conversion, not learned re-training.
    runtime_bytes = parameter_count * 4
    return {
        "source_payload_bytes": source_bytes,
        "runtime_donor_core_bytes": runtime_bytes,
        "donor_parameter_count": parameter_count,
    }


class _ZeroExpert(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.width = int(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def _load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**checkpoint["config"])
    model = build_model("remora", cfg)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device), checkpoint


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def _evaluate(model: nn.Module, old_stream: torch.Tensor, target_stream: torch.Tensor, *, device: torch.device, batch_size: int, seq_len: int, max_tokens: int) -> dict[str, Any]:
    _sync(device)
    old = _evaluate_stream_limited(model, old_stream, batch_size, seq_len, device, max_tokens)
    target = _evaluate_stream_limited(model, target_stream, batch_size, seq_len, device, max_tokens)
    route_load = getattr(model.blocks[1].experts, "last_load", None)
    return {
        "old": old,
        "target": target,
        "route_load_block1": route_load.detach().cpu().tolist() if route_load is not None else None,
    }


def _replace_port_state(graft: QwenSharedExpertGraft, port_state: dict[str, torch.Tensor]) -> None:
    current = graft.state_dict()
    for name, value in port_state.items():
        if name not in current:
            raise KeyError(name)
        current[name].copy_(value)
    graft.load_state_dict(current, strict=True)


def _port_state(
    tensors: dict[str, torch.Tensor], *, bus_dim: int, donor_dim: int, rank: int, seed: int
) -> dict[str, torch.Tensor]:
    set_seed(seed)
    reference = QwenSharedExpertGraft(tensors, bus_dim=bus_dim, donor_dim=donor_dim, rank=rank)
    return {
        name: value.detach().cpu().clone()
        for name, value in reference.state_dict().items()
        if name.startswith("input_port.") or name.startswith("output_port.")
    }


def _install_donor(
    model: nn.Module,
    tensors: dict[str, torch.Tensor],
    *,
    cfg: ModelConfig,
    variant: str,
    variant_seed: int,
    port_state: dict[str, torch.Tensor],
) -> QwenSharedExpertGraft:
    graft = QwenSharedExpertGraft(
        tensors,
        bus_dim=cfg.bus_dim,
        donor_dim=2560,
        rank=8,
        donor_variant=variant,
        variant_seed=variant_seed,
    )
    _replace_port_state(graft, port_state)
    graft = graft.to(next(model.parameters()).device)
    model.replace_expert(1, 0, graft)
    return graft


def _install_native(model: nn.Module, cfg: ModelConfig, arm: str, *, seed: int) -> nn.Module | None:
    if arm in {"native_frozen", "native_expert_local"}:
        return model.blocks[1].experts.experts[0]
    if arm == "native_random_expert_local":
        set_seed(seed)
        replacement = ExpertMLP(cfg.bus_dim, cfg.d_ff, cfg.bus_dim)
        model.replace_expert(1, 0, replacement.to(next(model.parameters()).device))
        return replacement
    if arm == "native_swiglu_local":
        set_seed(seed)
        replacement = SwiGLUExpert(cfg.bus_dim, cfg.d_ff, cfg.bus_dim)
        model.replace_expert(1, 0, replacement.to(next(model.parameters()).device))
        return replacement
    raise ValueError(arm)


def _prepare_scope(model: nn.Module, arm: str, module: nn.Module | None) -> list[str]:
    freeze_all(model)
    if arm == "native_frozen":
        return []
    if module is None:
        raise ValueError(f"missing replacement module for {arm}")
    if arm.startswith("donor_"):
        return unfreeze_prefixes(model, ["blocks.1.experts.experts.0.input_port", "blocks.1.experts.experts.0.output_port"])
    return unfreeze_prefixes(model, ["blocks.1.experts.experts.0"])


def _train_local(
    model: nn.Module,
    old_stream: torch.Tensor,
    target_stream: torch.Tensor,
    *,
    selected_names: list[str],
    steps: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    seed: int,
    rehearsal_weight: float,
) -> dict[str, Any]:
    if not selected_names:
        return {"optimization_steps": 0, "new_task_tokens": 0, "rehearsal_tokens": 0, "loss_first": None, "loss_last": None, "wall_seconds": 0.0}
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=0.003, betas=(0.9, 0.95), weight_decay=0.0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    started = time.perf_counter()
    for _ in range(steps):
        x, y = sample_batch(target_stream, batch_size, seq_len, device, generator)
        old_x, old_y = sample_batch(old_stream, batch_size, seq_len, device, generator)
        optimizer.zero_grad(set_to_none=True)
        _, target_loss = model(x, y)
        _, old_loss = model(old_x, old_y)
        loss = target_loss + rehearsal_weight * old_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    _sync(device)
    return {
        "optimization_steps": steps,
        "new_task_tokens": steps * batch_size * seq_len,
        "rehearsal_tokens": steps * batch_size * seq_len,
        "total_assimilation_tokens": steps * batch_size * seq_len * 2,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "wall_seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def _intervention_loss(
    model: nn.Module,
    old_stream: torch.Tensor,
    target_stream: torch.Tensor,
    *,
    cfg: ModelConfig,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    max_tokens: int,
) -> dict[str, Any]:
    original = model.blocks[1].experts.experts[0]
    baseline = _evaluate(model, old_stream, target_stream, device=device, batch_size=batch_size, seq_len=seq_len, max_tokens=max_tokens)
    model.blocks[1].experts.experts[0] = _ZeroExpert(cfg.bus_dim).to(device)
    ablated = _evaluate(model, old_stream, target_stream, device=device, batch_size=batch_size, seq_len=seq_len, max_tokens=max_tokens)
    model.blocks[1].experts.experts[0] = original
    return {
        "baseline": baseline,
        "ablated": ablated,
        "target_loss_delta_after_zero": ablated["target"]["loss"] - baseline["target"]["loss"],
        "old_loss_delta_after_zero": ablated["old"]["loss"] - baseline["old"]["loss"],
    }


@torch.no_grad()
def _donor_ablations(
    model: nn.Module,
    graft: QwenSharedExpertGraft,
    source_tensors: dict[str, torch.Tensor],
    old_stream: torch.Tensor,
    target_stream: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    seq_len: int,
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    original_buffers = {name: buffer.detach().clone() for name, buffer in graft.organ.named_buffers()}
    actual = _evaluate(model, old_stream, target_stream, device=device, batch_size=batch_size, seq_len=seq_len, max_tokens=max_tokens)
    result: dict[str, Any] = {"actual": actual, "controls": {}}
    for variant in ("shuffled", "random", "zero"):
        organ = QwenSharedExpertOrgan(make_donor_variant(source_tensors, variant, seed=seed))
        for name, buffer in graft.organ.named_buffers():
            buffer.copy_(dict(organ.named_buffers())[name].to(buffer.device))
        result["controls"][variant] = _evaluate(model, old_stream, target_stream, device=device, batch_size=batch_size, seq_len=seq_len, max_tokens=max_tokens)
    for name, buffer in graft.organ.named_buffers():
        buffer.copy_(original_buffers[name])
    for variant, evaluation in result["controls"].items():
        evaluation["target_loss_delta_vs_actual"] = evaluation["target"]["loss"] - actual["target"]["loss"]
        evaluation["old_loss_delta_vs_actual"] = evaluation["old"]["loss"] - actual["old"]["loss"]
    return result


def _run_seed(
    *,
    checkpoint_path: str | Path,
    payload_path: str | Path,
    seed: int,
    device: torch.device,
    steps: int,
    batch_size: int,
    seq_len: int,
    max_eval_tokens: int,
    rehearsal_weight: float,
    save_dir: Path | None,
) -> dict[str, Any]:
    tensors = load_payload(payload_path)
    payload_accounting = _payload_storage_accounting(tensors)
    probe_model, source = _load_checkpoint(checkpoint_path, device)
    cfg = probe_model.cfg
    tokenizer = ByteTokenizer(cfg.vocab_size)
    old_stream = encode_stream(build_heldout_corpus(500, seed=source["seed"] + 1000), tokenizer)
    target_stream = encode_stream(build_target_corpus(600, seed=seed + 200), tokenizer)
    intervention = _intervention_loss(
        probe_model,
        old_stream,
        target_stream,
        cfg=cfg,
        device=device,
        batch_size=batch_size,
        seq_len=seq_len,
        max_tokens=max_eval_tokens,
    )
    del probe_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    port_state = _port_state(tensors, bus_dim=cfg.bus_dim, donor_dim=2560, rank=8, seed=seed + 7000)
    arms: dict[str, Any] = {}
    for arm_index, arm in enumerate(TRAINING_ARMS):
        set_seed(seed + 100 + arm_index)
        model, source = _load_checkpoint(checkpoint_path, device)
        original_params = model_summary(model)["parameters"]
        graft = None
        if arm.startswith("donor_"):
            variant = arm.removeprefix("donor_").removesuffix("_local")
            graft = _install_donor(model, tensors, cfg=cfg, variant=variant, variant_seed=seed + 9000, port_state=port_state)
            module = graft
        else:
            module = _install_native(model, cfg, arm, seed=seed + 9000)
        selected_names = _prepare_scope(model, arm, module)
        before = parameter_snapshot(model)
        pre = _evaluate(model, old_stream, target_stream, device=device, batch_size=batch_size, seq_len=seq_len, max_tokens=max_eval_tokens)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        training = _train_local(
            model,
            old_stream,
            target_stream,
            selected_names=selected_names,
            steps=steps,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
            seed=seed + 2000 + arm_index,
            rehearsal_weight=rehearsal_weight,
        )
        post = _evaluate(model, old_stream, target_stream, device=device, batch_size=batch_size, seq_len=seq_len, max_tokens=max_eval_tokens)
        changed = changed_parameter_stats(before, model)
        peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        selected_parameter_count = sum(
            int(parameter.numel())
            for name, parameter in model.named_parameters()
            if name in selected_names
        )
        record: dict[str, Any] = {
            "arm": arm,
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "pre": pre,
            "post": post,
            "training": training,
            "selected_parameter_names": selected_names,
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "parameter_count": model_summary(model)["parameters"],
            "base_parameter_count_before_surgery": original_params,
            "changed_parameter_stats": changed,
            "peak_vram_bytes": peak_vram,
            "target_loss_delta": post["target"]["loss"] - pre["target"]["loss"],
            "old_loss_delta": post["old"]["loss"] - pre["old"]["loss"],
            "promotion_state": "CANDIDATE_ATTACHED" if arm.startswith("donor_") else "CONTROL",
        }
        if graft is not None:
            record["donor"] = {
                "variant": graft.donor_variant,
                "donor_payload_parameter_count": graft.donor_payload_parameter_count,
                "port_parameter_count": graft.port_parameter_count,
                "parameters_preserved_unchanged": graft.donor_payload_parameter_count,
                "parameters_analytically_transformed": 0,
                "parameters_discarded": 0,
                "newly_trained_parameters": selected_parameter_count,
                "donor_core_frozen": True,
                "payload_storage": payload_accounting,
                "assimilation_cost": {
                    "gradient_steps": training["optimization_steps"],
                    "new_task_tokens": training["new_task_tokens"],
                    "rehearsal_tokens": training["rehearsal_tokens"],
                    "total_tokens": training["total_assimilation_tokens"],
                    "wall_seconds": training["wall_seconds"],
                    "modeled_trainable_update_flops": 6 * selected_parameter_count * training["total_assimilation_tokens"],
                    "modeled_forward_flops_including_runtime_donor": 6 * (
                        model_summary(model)["parameters"] + graft.donor_payload_parameter_count
                    ) * training["total_assimilation_tokens"],
                    "original_donor_training_compute": {
                        "status": "UNMEASURED",
                        "value": None,
                        "reason": "The downloaded checkpoint does not contain the donor training run's optimizer/data/compute ledger.",
                    },
                },
                "interface_signature": graft.interface_signature(),
                "ablation": _donor_ablations(
                    model,
                    graft,
                    tensors,
                    old_stream,
                    target_stream,
                    device=device,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    max_tokens=max_eval_tokens,
                    seed=seed + 12000,
                ),
            }
        if save_dir is not None and arm == "donor_actual_local":
            save_path = save_dir / f"qwen-neural-graft-{arm}-seed{seed}.pt"
            torch.save({
                "schema": "remora-qwen-neural-graft-checkpoint-v1",
                "source_checkpoint": str(checkpoint_path),
                "payload_path": str(payload_path),
                "seed": seed,
                "config": cfg.to_dict(),
                "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "graft_result": record,
            }, save_path)
            record["saved_checkpoint"] = str(save_path)
        arms[arm] = record
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "seed": seed,
        "aged_checkpoint": str(checkpoint_path),
        "intervention": intervention,
        "arms": arms,
    }


def _aggregate(seed_runs: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"seeds": [run["seed"] for run in seed_runs], "arms": {}, "paired_donor_actual_vs_controls": {}}
    for arm in TRAINING_ARMS:
        values = [run["arms"][arm] for run in seed_runs]
        aggregate["arms"][arm] = {
            "target_loss": [item["post"]["target"]["loss"] for item in values],
            "old_loss": [item["post"]["old"]["loss"] for item in values],
            "target_loss_delta": [item["target_loss_delta"] for item in values],
            "old_loss_delta": [item["old_loss_delta"] for item in values],
            "wall_seconds": [item["training"]["wall_seconds"] for item in values],
            "trainable_parameters": [item["trainable_parameters"] for item in values],
            "changed_fraction": [item["changed_parameter_stats"]["changed_fraction"] for item in values],
        }
    for control in ("shuffled", "random", "zero"):
        actual = []
        ablated = []
        for run in seed_runs:
            for item in (run["arms"]["donor_actual_local"],):
                actual.append(item["donor"]["ablation"]["actual"]["target"]["loss"])
                ablated.append(item["donor"]["ablation"]["controls"][control]["target"]["loss"])
        aggregate["paired_donor_actual_vs_controls"][control] = {
            "actual_target_loss": actual,
            "control_target_loss": ablated,
            "control_minus_actual": [right - left for left, right in zip(actual, ablated)],
        }
    return aggregate


def run(
    checkpoint_dir: str | Path,
    payload_path: str | Path,
    *,
    seeds: list[int] | tuple[int, ...] = (7, 19, 31),
    device_name: str = "auto",
    steps: int = 60,
    batch_size: int = 16,
    seq_len: int = 96,
    max_eval_tokens: int = 1536,
    rehearsal_weight: float = 1.0,
    output: str | Path = ROOT / "results" / "qwen-neural-graft-v1.json",
    save_checkpoints: bool = True,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("seeds must not be empty")
    device = choose_device(device_name)
    checkpoint_dir = Path(checkpoint_dir)
    save_dir = ROOT / "checkpoints" if save_checkpoints else None
    seed_runs = []
    for seed in seeds:
        checkpoint = checkpoint_dir / f"lifetime-compounding-remora_local_rehearsal-seed{seed}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        seed_runs.append(_run_seed(
            checkpoint_path=checkpoint,
            payload_path=payload_path,
            seed=seed,
            device=device,
            steps=steps,
            batch_size=batch_size,
            seq_len=seq_len,
            max_eval_tokens=max_eval_tokens,
            rehearsal_weight=rehearsal_weight,
            save_dir=save_dir,
        ))
    result = {
        "schema": "remora-qwen-neural-graft-result-v1",
        "experiment_family": "QWEN-NEURAL-ORGAN-002+",
        "mode": "AGED_REMORA_USED_PATH_WRAPPED_FOREIGN_WEIGHT_SURGERY",
        "device": str(device),
        "payload_path": str(payload_path),
        "seeds": list(seeds),
        "training": {
            "steps": steps,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "max_eval_tokens": max_eval_tokens,
            "rehearsal_weight": rehearsal_weight,
            "optimizer": "AdamW(lr=0.003, betas=(0.9,0.95), weight_decay=0)",
            "graft_port_rank": 8,
            "used_path": "blocks.1.experts.experts.0",
            "assimilation_token_definition": "new_task_tokens + rehearsal_tokens for the frozen-core port-repair run",
        },
        "controls": {
            "native_frozen": "aged Remora checkpoint without local update",
            "native_expert_local": "existing ExpertMLP at the used path, local update",
            "native_random_expert_local": "same-shape fresh ExpertMLP replacement, local update",
            "native_swiglu_local": "Remora-native SwiGLU replacement, local update",
            "donor_actual_local": "actual Qwen trained shared-expert core, frozen behind rank-8 ports, local port update",
            "donor_shuffled_local": "same graft with per-tensor value permutation, same port state and budget",
            "donor_random_local": "same graft with shape/std-matched random donor core, same port state and budget",
            "donor_zero_ablation": "post-training zero donor core with ports held fixed",
        },
        "predeclared_gates": {
            "donor_core_must_be_frozen": True,
            "foreign_weights_must_change_behavior_under_ablation": "actual vs shuffled/random/zero donor core should differ on held-out target; if not, donor weights are not shown responsible",
            "native_and_donor_controls_use_same_checkpoint_and_streams": True,
            "no_promotion_without_external_decision": True,
        },
        "runs": seed_runs,
        "aggregate": _aggregate(seed_runs),
        "donor_lineage": {
            "source_model": "Qwen/Qwen3.8-Flash-Next",
            "source_revision": "f5d08274bafd880402bd16f5e3e6c514136ec06c",
            "source_component": "model.language_model.layers.0.mlp.shared_expert plus shared_expert_gate.weight",
            "extraction": str(payload_path),
            "conversion": "minimal Neural IR -> frozen native-width organ -> rank-8 low-rank Remora ports",
            "attachment": "blocks.1.experts.experts.0",
            "payload_storage": _payload_storage_accounting(load_payload(payload_path)),
            "original_donor_training_compute": {
                "status": "UNMEASURED",
                "value": None,
                "reason": "No original Qwen training ledger is present in the local checkpoint source.",
            },
        },
        "labels": {
            "MEASURED": ["held-out losses", "ablation deltas", "wall time", "changed parameters", "VRAM", "trainable parameters"],
            "DERIVED": ["control-minus-actual loss differences", "parameter fractions"],
            "MODELED": ["modeled_trainable_update_flops", "modeled_forward_flops_including_runtime_donor"],
            "ESTIMATED": [],
            "HYPOTHESIS": ["foreign trained weights improve the used Remora path after local repair"],
            "EXTERNAL": ["deterministic synthetic corpus construction"],
        },
        "promotion_state": "CONTROLLED_EXPERIMENT_ONLY",
    }
    write_json(output, result)
    runtime = runtime_context(device)
    runtime["mode"] = "aged_donor_graft"
    runtime["command"] = "python -m experiments.donor_graft"
    record_experiment(
        ROOT,
        "QWEN-NEURAL-GRAFT-001",
        "Actual trained Qwen neural machinery can contribute to an experienced Remora used path after only local interface repair, and its contribution should disappear under donor-core destruction controls.",
        "Replace blocks.1.experts.experts.0 in post-lifetime Remora checkpoints with a frozen Qwen shared expert behind rank-8 ports; compare native, fresh, shuffled, random, and zero donor controls on fixed held-out streams.",
        "The actual donor arm improves or preserves target/old capability at bounded local cost, actual-vs-destroyed donor cores produce a reproducible held-out difference, and no broad model parameters are trained.",
        "Actual graft is no better than same-port random/shuffled/zero controls; donor ablation has no effect; retention collapses; port weights are the only effective computation; or the result depends on unequal data/steps/checkpoints.",
        f"python -m experiments.donor_graft --checkpoint-dir {checkpoint_dir} --payload {payload_path} --seeds {' '.join(str(seed) for seed in seeds)} --steps {steps}",
        0,
        {"aggregate": result["aggregate"], "promotion_state": result["promotion_state"]},
        "MEASURED: donor graft remains controlled until actual-vs-destroyed-weight utility and independent reproduction are established.",
        "If donor core matters, test lower-rank/subspace conversion and a stateful Gated-DeltaNet organ; if it does not, record the boundary failure and do not promote.",
        hardware=runtime,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the aged Remora Qwen donor-organ graft experiment.")
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints"))
    parser.add_argument("--payload", default=str(ROOT / "results" / "qwen-neural-organ-layer0.safetensors"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--max-eval-tokens", type=int, default=1536)
    parser.add_argument("--rehearsal-weight", type=float, default=1.0)
    parser.add_argument("--output", default=str(ROOT / "results" / "qwen-neural-graft-v1.json"))
    parser.add_argument("--no-checkpoints", action="store_true")
    args = parser.parse_args()
    result = run(
        args.checkpoint_dir,
        args.payload,
        seeds=args.seeds,
        device_name=args.device,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_eval_tokens=args.max_eval_tokens,
        rehearsal_weight=args.rehearsal_weight,
        output=args.output,
        save_checkpoints=not args.no_checkpoints,
    )
    print(json.dumps({"output": args.output, "aggregate": result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
