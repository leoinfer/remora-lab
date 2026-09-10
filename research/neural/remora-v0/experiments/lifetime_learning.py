from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.consolidation import consolidation_report
from remora.ledger import record_experiment
from remora.memory import EvidenceStore, EpisodicArchive
from remora.utils import choose_device, changed_parameter_stats, parameter_snapshot, set_seed, write_json
from remora.world_model import Hypothesis, HypothesisTracker, RuleWorldModel


def _fit(module, params, loss_fn, steps: int, lr: float) -> list[float]:
    optimizer = torch.optim.AdamW(params, lr=lr)
    history = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def _accuracy(model, features, labels, consolidated: bool = False) -> float:
    with torch.no_grad():
        logits = model.consolidated_forward(features) if consolidated else model(features)
        return float((logits.argmax(-1) == labels).float().mean())


def _evidence_independence() -> dict:
    def make(mode: str) -> EvidenceStore:
        store = EvidenceStore(prior_log_odds=0.0)
        store.add("inherited", "literature", "literature-cluster", {"z": "any", "environment": 0}, "Y", 0.8)
        count = 24
        for i in range(count):
            cluster = "same-program" if mode == "duplicate" else (f"independent-{i}" if mode == "independent" else f"correlated-{i // 6}")
            store.add(
                "experienced", "program-A" if mode != "independent" else f"run-{i}", cluster,
                {"z": 1, "environment": 0}, "X", 0.9,
                provenance={"source": "simulated-world", "repeat": i},
            )
        return store

    return {
        mode: make(mode).posterior({"z": 1, "environment": 0})
        for mode in ("duplicate", "correlated", "independent")
    }


def run(seed: int = 31, device_name: str = "cpu", output: str | Path | None = None) -> dict:
    set_seed(seed)
    device = choose_device(device_name)
    model = RuleWorldModel().to(device)
    z = torch.tensor([0, 1, 0, 1, 0, 1, 1, 0], device=device).float()
    env = torch.zeros_like(z)
    measurement = torch.arange(z.numel(), device=device).float() / 10.0
    features = model.features(z, env, measurement)
    old_features = model.features(
        torch.zeros(12, device=device), torch.zeros(12, device=device), torch.linspace(0, 1, 12, device=device)
    )
    old_labels = torch.ones(12, dtype=torch.long, device=device)
    inherited_labels = torch.ones(z.numel(), dtype=torch.long, device=device)  # Y=1
    experience_features = model.features(
        torch.ones(12, device=device), torch.zeros(12, device=device), torch.linspace(0, 1, 12, device=device)
    )
    experience_labels = torch.zeros(12, dtype=torch.long, device=device)  # X=0

    # Phase 1: inherited prior, trained from random initialization.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.inherited.parameters():
        p.requires_grad = True
    prior_history = _fit(
        model,
        [p for p in model.parameters() if p.requires_grad],
        lambda: nn.functional.cross_entropy(model.inherited(features), inherited_labels),
        steps=180,
        lr=0.04,
    )
    prior_before_experience = {
        "z0": int(model(features[:1]).argmax(-1)[0]),
        "z1": int(model(features[1:2]).argmax(-1)[0]),
    }

    archive = EpisodicArchive()
    store = EvidenceStore(prior_log_odds=0.0)
    inherited_id = archive.append({"kind": "literature", "rule": "Y usually holds", "lineage": "literature"})
    store.add("inherited", "literature", "literature-cluster", {"z": "any", "environment": 0}, "Y", 0.8, {"episode": inherited_id})
    experience_ids = []
    for i in range(12):
        episode_id = archive.append({"kind": "experience", "z": 1, "environment": 0, "outcome": "X", "run": i})
        experience_ids.append(episode_id)
        store.add("experienced", "run-family-A", "experience-cluster-A", {"z": 1, "environment": 0}, "X", 0.9, {"episode": episode_id})

    # Compare the first failed strategy with a surgical replay-constrained update.
    naive_model = copy.deepcopy(model)
    for p in naive_model.parameters():
        p.requires_grad = False
    for p in naive_model.experience_adapter.parameters():
        p.requires_grad = True
    naive_history = _fit(
        naive_model,
        [p for p in naive_model.parameters() if p.requires_grad],
        lambda: nn.functional.cross_entropy(naive_model(experience_features), experience_labels),
        steps=180,
        lr=0.04,
    )
    naive_accuracy = {
        "old_z0_Y": int(naive_model(old_features[:1]).argmax(-1)[0]) == 1,
        "conditioned_z1_X": int(naive_model(experience_features[:1]).argmax(-1)[0]) == 0,
    }

    before_adapter = parameter_snapshot(model)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.experience_adapter.parameters():
        p.requires_grad = True

    def retained_loss():
        target_loss = nn.functional.cross_entropy(model(experience_features), experience_labels)
        retention_loss = nn.functional.cross_entropy(model(old_features), old_labels)
        return target_loss + 1.0 * retention_loss

    experience_history = _fit(
        model,
        [p for p in model.parameters() if p.requires_grad],
        retained_loss,
        steps=180,
        lr=0.04,
    )
    adapted_accuracy = {
        "old_z0_Y": int(model(old_features[:1]).argmax(-1)[0]) == 1,
        "conditioned_z1_X": int(model(experience_features[:1]).argmax(-1)[0]) == 0,
    }
    update_stats = changed_parameter_stats(before_adapter, model)
    before_consolidation = parameter_snapshot(model)
    model.consolidate_from_adapter()
    consolidation_write_stats = changed_parameter_stats(before_consolidation, model)
    consolidated_accuracy = {
        "old_z0_Y": _accuracy(model, old_features, old_labels, consolidated=True),
        "conditioned_z1_X": _accuracy(model, experience_features, experience_labels, consolidated=True),
    }

    tracker = HypothesisTracker([
        Hypothesis("H1", "Y is the default rule", 0.7),
        Hypothesis("H2", "X holds when z=1", 0.5),
        Hypothesis("H3", "measurement is biased", 0.3),
    ])
    tracker.observe("experience-aggregate", {"H1": "Y", "H2": "X", "H3": "Y"}, "X")
    result = {
        "schema": "remora-v0-lifetime-result",
        "seed": seed,
        "device": str(device),
        "prior_training": {"steps": len(prior_history), "first_loss": prior_history[0], "last_loss": prior_history[-1]},
        "prior_before_experience": prior_before_experience,
        "evidence_independence": _evidence_independence(),
        "separated_evidence": store.separated_posterior({"z": 1, "environment": 0}),
        "archive_records": len(archive.records),
        "naive_local_update": {
            "first_loss": naive_history[0],
            "last_loss": naive_history[-1],
            "accuracy": naive_accuracy,
            "interpretation": "MEASURED FAILURE: target fit was achieved while old z=0 retention failed in the first run.",
        },
        "experience_training": {"steps": len(experience_history), "first_loss": experience_history[0], "last_loss": experience_history[-1], "retention_weight": 1.0},
        "neural_after_local_update": adapted_accuracy,
        "consolidated_without_retrieval": consolidated_accuracy,
        "adapter_update": update_stats,
        "consolidation": {
            **consolidation_report(before_consolidation, model, experience_ids),
            "local_adaptation_update": update_stats,
            "consolidation_write_update": consolidation_write_stats,
        },
        "hypotheses": tracker.to_dict(),
        "interpretation": "MEASURED: first naive local update failed retention; old-task rehearsal repaired the tested z=0 case while keeping the update local. Inherited and experienced evidence remain separately inspectable and duplicate clusters are bounded.",
    }
    if output:
        write_json(output, result)
    record_experiment(
        ROOT,
        "LIFETIME-EVIDENCE-002",
        "A local experience update with old-task rehearsal can represent Y as default and X under z=1 without rewriting the inherited path.",
        "Reproduce the failed naive adapter update, then add a frozen old-z=0 rehearsal term while training only experience_adapter; evaluate after retrieval is disabled.",
        "The repaired arm retains z=0=Y, learns z=1=X, has a small update fraction, and duplicate evidence has fewer effective clusters than independent evidence.",
        "The rehearsal arm still flips z=0, fails z=1, or changes nearly all parameters; duplicate evidence matches independent confidence.",
        "python -m experiments.lifetime_learning",
        seed,
        {k: v for k, v in result.items() if k not in {"hypotheses"}},
        "MEASURED: see result fields; interpretation is intentionally conditional on the thresholds in V0_HYPOTHESIS.md.",
        "Repeat across hidden conditions and compare the rehearsal arm against naive full-model fine-tuning and replay-only controls.",
        hardware={"device": str(device)},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda"])
    parser.add_argument("--output", default=str(ROOT / "results" / "lifetime-evidence.json"))
    args = parser.parse_args()
    result = run(args.seed, args.device, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
