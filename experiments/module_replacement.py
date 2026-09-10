from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.synthetic_corpus import build_heldout_corpus, build_target_corpus
from remora.config import ModelConfig
from remora.data import encode_stream, sample_batch
from remora.ledger import record_experiment, record_failure
from remora.metrics import evaluate_stream, model_summary
from remora.models import build_model
from remora.modules import SwiGLUExpert
from remora.utils import (
    changed_parameter_stats,
    choose_device,
    freeze_all,
    parameter_snapshot,
    set_seed,
    unfreeze_prefixes,
    write_json,
)


def _load(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**checkpoint["config"])
    model_kind = "remora" if checkpoint["model_type"].startswith("remora") else "baseline"
    model = build_model(model_kind, cfg)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device), checkpoint


def _adapt(
    model,
    stream,
    seq_len,
    batch_size,
    device,
    prefixes,
    steps,
    seed,
    lr,
    retention_stream=None,
    retention_weight: float = 0.0,
):
    freeze_all(model)
    selected = unfreeze_prefixes(model, prefixes)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history = []
    for step in range(steps):
        x, y = sample_batch(stream, batch_size, seq_len, device, generator)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if retention_stream is not None and retention_weight:
            old_x, old_y = sample_batch(retention_stream, batch_size, seq_len, device, generator)
            _, retention_loss = model(old_x, old_y)
            loss = loss + retention_weight * retention_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        history.append(float(loss.detach()))
    return selected, history


def _evaluate(model, old_stream, target_stream, seq_len, batch_size, device) -> dict:
    return {
        "old": evaluate_stream(model, old_stream, batch_size, seq_len, device),
        "target": evaluate_stream(model, target_stream, batch_size, seq_len, device),
    }


def run(
    checkpoint: str | Path,
    baseline_checkpoint: str | Path | None = None,
    seed: int = 53,
    steps: int = 120,
    device_name: str = "auto",
    output: str | Path | None = None,
    retention_weight: float = 0.0,
) -> dict:
    set_seed(seed)
    device = choose_device(device_name)
    model, source = _load(checkpoint, device)
    cfg = model.cfg
    old_stream = encode_stream(build_heldout_corpus(500, seed=source["seed"] + 1000))
    target_stream = encode_stream(build_target_corpus(600, seed=seed + 200))
    seq_len = min(cfg.max_seq_len, 96)
    batch_size = 32
    original = _evaluate(model, old_stream, target_stream, seq_len, batch_size, device)

    old_expert = model.blocks[1].experts.experts[0]
    old_expert_params = sum(p.numel() for p in old_expert.parameters())
    replacement = SwiGLUExpert(cfg.bus_dim, cfg.d_ff, cfg.bus_dim).to(device)
    replacement_params = sum(p.numel() for p in replacement.parameters())
    model.replace_expert(1, 0, replacement)
    replaced_before = _evaluate(model, old_stream, target_stream, seq_len, batch_size, device)
    before_local = parameter_snapshot(model)
    scope = ["blocks.1.experts.experts.0"]
    selected, adaptation_history = _adapt(
        model,
        target_stream,
        seq_len,
        batch_size,
        device,
        scope,
        steps,
        seed + 1,
        2e-3,
        retention_stream=old_stream,
        retention_weight=retention_weight,
    )
    replaced_after = _evaluate(model, old_stream, target_stream, seq_len, batch_size, device)
    local_stats = changed_parameter_stats(before_local, model)
    total_params = model_summary(model)["parameters"]
    retention_gate = replaced_after["old"]["loss"] <= original["old"]["loss"] * 2.0
    target_improved = replaced_after["target"]["loss"] < replaced_before["target"]["loss"]
    conclusion = (
        "MEASURED: replacement recovered target capability under the declared local update scope and passed the old-task retention gate."
        if retention_gate and target_improved
        else "MEASURED FAILURE: replacement improved the target but did not pass the old-task retention gate under this update policy."
    )

    baseline_result = None
    if baseline_checkpoint:
        baseline, baseline_source = _load(baseline_checkpoint, device)
        before_baseline = parameter_snapshot(baseline)
        baseline_before = _evaluate(baseline, old_stream, target_stream, seq_len, batch_size, device)
        freeze_all(baseline)
        for p in baseline.parameters():
            p.requires_grad = True
        optimizer = torch.optim.AdamW(baseline.parameters(), lr=2e-3)
        generator = torch.Generator(device="cpu").manual_seed(seed + 2)
        baseline_history = []
        for _ in range(steps):
            x, y = sample_batch(target_stream, batch_size, seq_len, device, generator)
            optimizer.zero_grad(set_to_none=True)
            _, loss = baseline(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1.0)
            optimizer.step()
            baseline_history.append(float(loss.detach()))
        baseline_result = {
            "before": baseline_before,
            "after": _evaluate(baseline, old_stream, target_stream, seq_len, batch_size, device),
            "update": changed_parameter_stats(before_baseline, baseline),
            "history_first_last": [baseline_history[0], baseline_history[-1]],
            "parameter_count": model_summary(baseline)["parameters"],
        }

    result = {
        "schema": "remora-v0-module-replacement-result",
        "seed": seed,
        "device": str(device),
        "source_checkpoint": str(checkpoint),
        "source_seed": source["seed"],
        "scope": {"replaced": "blocks.1.experts.experts.0", "retrained_prefixes": scope, "selected_parameters": selected},
        "original": original,
        "replaced_before_local_training": replaced_before,
        "replaced_after_local_training": replaced_after,
        "replacement_parameter_accounting": {
            "old_expert_parameters": old_expert_params,
            "new_expert_parameters": replacement_params,
            "added_parameters": replacement_params - old_expert_params,
            "total_model_parameters": total_params,
            "new_module_fraction_of_model": replacement_params / total_params,
        },
        "local_update": local_stats,
        "local_adaptation_loss_first_last": [adaptation_history[0], adaptation_history[-1]],
        "retention_weight": retention_weight,
        "baseline_full_model_control": baseline_result,
        "interpretation": conclusion,
    }
    if output:
        out_checkpoint = Path(output).with_suffix(".pt")
        torch.save({**source, "state_dict": model.state_dict(), "replacement_result": result}, out_checkpoint)
        result["checkpoint"] = str(out_checkpoint)
        write_json(output, result)
    experiment_id = "MODULE-REPLACEMENT-002" if retention_weight else "MODULE-REPLACEMENT-001"
    record_experiment(
        ROOT,
        experiment_id,
        "A used specialist can be structurally replaced and locally adapted while preserving more old capability per updated parameter than full-model adaptation.",
        "Replace layer-1 expert-0 with a SwiGLU implementation sharing expert-v1 ports; train only that module and compare a full-model baseline control.",
        "Target loss improves, old loss remains bounded, and the local update fraction is much smaller than the full-model control.",
        "Replacement has no causal effect, target adaptation requires broad retraining, or old capability collapses beyond the predeclared retention gate.",
        f"python -m experiments.module_replacement --checkpoint {checkpoint} --retention-weight {retention_weight}",
        seed,
        result,
        conclusion,
        "Repeat with a different layer/expert, a frozen-replacement ablation, and a retention-weight sweep before promotion.",
        hardware={"device": str(device), "retention_weight": retention_weight},
    )
    if not retention_gate:
        record_failure(
            ROOT,
            experiment_id,
            cfg.to_dict(),
            {"old_stream": "heldout synthetic corpus", "target_stream": "target synthetic corpus", "seq_len": seq_len, "batch_size": batch_size},
            seed,
            f"old loss moved from {original['old']['loss']:.6f} to {replaced_after['old']['loss']:.6f} while target loss moved to {replaced_after['target']['loss']:.6f}",
            "A newly initialized replacement perturbs a shared routed path enough that target-only local training forgets old behavior; rehearsal may trade target speed for retention.",
            "Retest after changing the replacement initialization, adding a distillation/rehearsal term, or changing the surrounding router/bus while retaining the original failure context.",
            runtime={"device": str(device), "retention_weight": retention_weight},
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default=str(ROOT / "results" / "module-replacement.json"))
    parser.add_argument("--retention-weight", type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.baseline_checkpoint, args.seed, args.steps, args.device, args.output, args.retention_weight), indent=2))


if __name__ == "__main__":
    main()
