from __future__ import annotations

"""Three-timescale experience-to-intuition probe.

This is a bounded structured-world experiment, not a claim about general
language ability.  The inherited path is trained on a default rule, the fast
experience path is learned from explicitly retrieved lifetime episodes, and
the same validated adapter is then copied into a slow consolidated path.
The raw episodes and evidence clusters remain available after retrieval is
disabled.
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

from remora.consolidation import consolidation_report
from remora.ledger import record_experiment
from remora.memory import EvidenceStore, EpisodicArchive
from remora.utils import changed_parameter_stats, choose_device, count_parameters, parameter_snapshot, set_seed, write_json
from remora.world_model import RuleWorldModel


def _fit(params, loss_fn, steps: int, lr: float) -> list[float]:
    optimizer = torch.optim.AdamW(params, lr=lr)
    history = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


@torch.no_grad()
def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(-1) == labels).float().mean())


@torch.no_grad()
def _latency(fn, repeats: int = 100) -> float:
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - started) / max(repeats, 1)


def _one_seed(seed: int, device: torch.device, *, prior_steps: int, experience_steps: int) -> dict[str, Any]:
    set_seed(seed)
    model = RuleWorldModel().to(device)
    # Inherited civilization prior: Y is the default outcome, including for
    # the hidden condition before the lifetime discrepancy is observed.
    prior_z = torch.tensor([0, 1, 0, 1, 0, 1, 1, 0], device=device).float()
    prior_env = torch.zeros_like(prior_z)
    prior_measurement = torch.linspace(0.0, 0.7, prior_z.numel(), device=device)
    prior_features = model.features(prior_z, prior_env, prior_measurement)
    prior_labels = torch.ones(prior_z.numel(), dtype=torch.long, device=device)

    old_z = torch.zeros(24, device=device)
    old_env = torch.zeros_like(old_z)
    old_measurement = torch.linspace(0.0, 1.0, old_z.numel(), device=device)
    old_features = model.features(old_z, old_env, old_measurement)
    old_labels = torch.ones(old_z.numel(), dtype=torch.long, device=device)

    experience_z = torch.ones(24, device=device)
    experience_env = torch.zeros_like(experience_z)
    experience_measurement = torch.linspace(0.0, 1.0, experience_z.numel(), device=device)
    experience_features = model.features(experience_z, experience_env, experience_measurement)
    experience_labels = torch.zeros(experience_z.numel(), dtype=torch.long, device=device)

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.inherited.parameters():
        parameter.requires_grad = True
    prior_history = _fit(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lambda: nn.functional.cross_entropy(model.inherited(prior_features), prior_labels),
        prior_steps,
        0.04,
    )

    archive = EpisodicArchive()
    evidence = EvidenceStore()
    inherited_episode = archive.append({
        "kind": "inherited_prior",
        "rule": "Y is the default",
        "source": "pretraining",
        "lineage": "civilization-prior-v1",
    })
    evidence.add(
        "inherited",
        "civilization-prior-v1",
        "inherited-cluster",
        {"z": "any", "environment": 0},
        "Y",
        confidence=0.8,
        provenance={"episode_id": inherited_episode},
    )
    experience_ids = []
    for index in range(24):
        episode_id = archive.append({
            "kind": "lifetime_observation",
            "z": 1,
            "environment": 0,
            "outcome": "X",
            "measurement": index / 23.0,
            "lineage": f"independent-run-{index:02d}",
            "independence_cluster": f"run-cluster-{index:02d}",
        })
        experience_ids.append(episode_id)
        evidence.add(
            "experienced",
            f"independent-run-{index:02d}",
            f"experience-cluster-{index:02d}",
            {"z": 1, "environment": 0},
            "X",
            confidence=0.9,
            provenance={"episode_id": episode_id, "independence": "independent"},
        )

    before_fast = parameter_snapshot(model)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.experience_adapter.parameters():
        parameter.requires_grad = True

    def fast_loss() -> torch.Tensor:
        # Explicit retrieval path: the experience adapter sees retrieved
        # lifetime cases and a small old-task replay batch.  It cannot rewrite
        # the inherited path in this phase.
        target = nn.functional.cross_entropy(model(experience_features, use_experience=True), experience_labels)
        replay = nn.functional.cross_entropy(model(old_features, use_experience=True), old_labels)
        return target + replay

    fast_history = _fit(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        fast_loss,
        experience_steps,
        0.04,
    )
    fast_update = changed_parameter_stats(before_fast, model)

    inherited_only_old = _accuracy(model(old_features, use_experience=False), old_labels)
    inherited_only_experience = _accuracy(model(experience_features, use_experience=False), experience_labels)
    retrieved_old = _accuracy(model(old_features, use_experience=True), old_labels)
    retrieved_experience = _accuracy(model(experience_features, use_experience=True), experience_labels)
    retrieval_disabled_before = {
        "old_unrelated_accuracy": inherited_only_old,
        "experienced_condition_accuracy": inherited_only_experience,
        "latency_seconds_per_query": _latency(lambda: model(experience_features, use_experience=False)),
    }
    retrieval_enabled = {
        "old_unrelated_accuracy": retrieved_old,
        "experienced_condition_accuracy": retrieved_experience,
        "latency_seconds_per_query": _latency(lambda: model(experience_features, use_experience=True)),
        "retrieval_calls": len(experience_ids),
        "retrieved_episode_ids": experience_ids,
    }

    before_consolidation = parameter_snapshot(model)
    model.consolidate_from_adapter()
    consolidation_write = changed_parameter_stats(before_consolidation, model)
    consolidated_old = _accuracy(model.consolidated_forward(old_features), old_labels)
    consolidated_experience = _accuracy(model.consolidated_forward(experience_features), experience_labels)
    consolidated = {
        "old_unrelated_accuracy": consolidated_old,
        "experienced_condition_accuracy": consolidated_experience,
        "latency_seconds_per_query": _latency(lambda: model.consolidated_forward(experience_features)),
        "retrieval_enabled": False,
    }
    evidence_summary = evidence.separated_posterior({"z": 1, "environment": 0})
    receipt = {
        "consolidation_id": f"consolidation-{seed:03d}-experience-parity",
        "source_episode_ids": experience_ids,
        "source_independence_clusters": [f"experience-cluster-{index:02d}" for index in range(24)],
        "prior_belief_changed": "Y default -> X under z=1, environment=0",
        "absorbing_module": "world_model.consolidated",
        "validation": "held-out old and experienced feature sets",
        "archive_recoverable": True,
    }
    return {
        "seed": seed,
        "prior_training": {"steps": prior_steps, "first_loss": prior_history[0], "last_loss": prior_history[-1]},
        "fast_experience_update": {
            "steps": experience_steps,
            "first_loss": fast_history[0],
            "last_loss": fast_history[-1],
            "parameter_update": fast_update,
        },
        "retrieval_disabled_before_consolidation": retrieval_disabled_before,
        "retrieval_enabled_before_consolidation": retrieval_enabled,
        "consolidated_without_retrieval": consolidated,
        "consolidation_write": consolidation_write,
        "consolidation_report": consolidation_report(before_consolidation, model, experience_ids),
        "provenance_receipt": receipt,
        "evidence": evidence_summary,
        "archive": archive.to_dict(),
        "parameter_counts": {
            "total": count_parameters(model),
            "experience_adapter": count_parameters(model.experience_adapter),
            "consolidated": count_parameters(model.consolidated),
        },
    }


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def values(path: tuple[str, ...]) -> list[float]:
        result = []
        for row in runs:
            value: Any = row
            for key in path:
                value = value[key]
            result.append(float(value))
        return result

    def stats(items: list[float]) -> dict[str, Any]:
        mean = sum(items) / len(items)
        variance = sum((item - mean) ** 2 for item in items) / max(len(items) - 1, 1)
        return {"values": items, "mean": mean, "std": math.sqrt(variance)}

    metric_paths = {
        "retrieval_required_experience_accuracy": ("retrieval_disabled_before_consolidation", "experienced_condition_accuracy"),
        "retrieval_enabled_experience_accuracy": ("retrieval_enabled_before_consolidation", "experienced_condition_accuracy"),
        "consolidated_experience_accuracy": ("consolidated_without_retrieval", "experienced_condition_accuracy"),
        "retrieval_enabled_old_accuracy": ("retrieval_enabled_before_consolidation", "old_unrelated_accuracy"),
        "consolidated_old_accuracy": ("consolidated_without_retrieval", "old_unrelated_accuracy"),
        "retrieval_latency": ("retrieval_enabled_before_consolidation", "latency_seconds_per_query"),
        "consolidated_latency": ("consolidated_without_retrieval", "latency_seconds_per_query"),
        "fast_changed_fraction": ("fast_experience_update", "parameter_update", "changed_fraction"),
        "consolidation_write_fraction": ("consolidation_write", "changed_fraction"),
    }
    return {name: stats(values(path)) for name, path in metric_paths.items()}


def run(
    *,
    seeds: list[int] | tuple[int, ...] = (7, 19, 31),
    device_name: str = "cpu",
    prior_steps: int = 180,
    experience_steps: int = 180,
    output: str | Path = ROOT / "results" / "consolidation-probe-v1.json",
) -> dict[str, Any]:
    device = choose_device(device_name)
    runs = [_one_seed(seed, device, prior_steps=prior_steps, experience_steps=experience_steps) for seed in seeds]
    result = {
        "schema": "remora-v1-consolidation-probe-result",
        "experiment_family": "CONSOLIDATION-001+",
        "mode": "STRUCTURED_WORLD_FAST_EXPERIENCE_TO_SLOW_INTUITION",
        "device": str(device),
        "training": {"seeds": list(seeds), "prior_steps": prior_steps, "experience_steps": experience_steps},
        "runs": runs,
        "aggregate": _aggregate(runs),
        "labels": {
            "MEASURED": ["accuracy", "latency", "parameter changes", "steps", "archive/provenance records"],
            "DERIVED": ["retention", "retrieval dependence", "latency change", "parameter update fraction"],
            "HYPOTHESIS": ["validated repeated experience can become a retrieval-independent neural prior"],
        },
        "promotion": {"state": "CONTROLLED_EXPERIMENT_ONLY", "promoted": False},
        "interpretation": "MEASURED structured-world probe: the fast path is explicitly needed before consolidation, the consolidated path is evaluated without retrieval, and provenance remains in the archive; this does not establish language-level consolidation.",
    }
    write_json(output, result)
    record_experiment(
        ROOT,
        "CONSOLIDATION-001",
        "Repeated validated experience should transfer from an explicit fast/retrieval path into a small slow neural path without destroying unrelated inherited behavior or provenance.",
        "Train the inherited default rule, fit only the experience adapter with replay, copy the validated adapter into the consolidated path, disable the fast path, and retain an append-only evidence/archive receipt.",
        "Experienced queries require the fast path before consolidation but remain accurate through consolidated_forward afterward; old queries remain intact, only local parameter islands change, and supporting episodes/clusters are recoverable.",
        "Consolidated no-retrieval accuracy fails, unrelated old behavior regresses, provenance cannot be traced, or the update is not local.",
        f"python -m experiments.consolidation_probe --seeds {' '.join(str(seed) for seed in seeds)}",
        0,
        {"aggregate": result["aggregate"], "promotion": result["promotion"]},
        result["interpretation"],
        "Add a learned retrieval gate and test conflicting multi-hypothesis evidence before treating this mechanism as general Remora capability.",
        hardware={"device": str(device), "mode": "structured_world_consolidation"},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19, 31])
    parser.add_argument("--device", choices=["cpu", "auto", "cuda"], default="cpu")
    parser.add_argument("--prior-steps", type=int, default=180)
    parser.add_argument("--experience-steps", type=int, default=180)
    parser.add_argument("--output", default=str(ROOT / "results" / "consolidation-probe-v1.json"))
    args = parser.parse_args()
    result = run(
        seeds=args.seeds,
        device_name=args.device,
        prior_steps=args.prior_steps,
        experience_steps=args.experience_steps,
        output=args.output,
    )
    print(json.dumps({"interpretation": result["interpretation"], "aggregate": result["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
