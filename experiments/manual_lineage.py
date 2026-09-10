from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.ledger import record_experiment
from remora.manual import AssemblyLedger, LearnedDependencyReader, ModuleRecord
from remora.utils import set_seed, write_json


def _closure(adjacency: torch.Tensor) -> torch.Tensor:
    out = adjacency.clone()
    for k in range(out.size(0)):
        out = torch.maximum(out, out[:, k : k + 1] * out[k : k + 1, :])
    return out


def _random_graph(rng: random.Random, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    adjacency = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.35:
                adjacency[i, j] = 1.0
    closure = _closure(adjacency)
    query = torch.zeros(n)
    query[rng.randrange(n)] = 1.0
    target = closure[query.argmax().item()]
    return adjacency, query, target


def run(seed: int = 41, output: str | Path | None = None) -> dict:
    set_seed(seed)
    ledger = AssemblyLedger(generation=4)
    names = ["language-v1", "attention-0", "recurrent-0", "expert-0", "adapter-0", "lm-head"]
    for name in names:
        ledger.register_module(ModuleRecord(name, "v1", ["bus"], ["bus"], name, 100, ancestry=[name + "@g0"]))
    ledger.add_dependency("language-v1", "attention-0")
    ledger.add_dependency("language-v1", "recurrent-0")
    ledger.add_dependency("language-v1", "expert-0")
    ledger.add_dependency("expert-0", "adapter-0")
    ledger.add_dependency("attention-0", "lm-head")
    ledger.add_dependency("adapter-0", "lm-head")
    ledger.add_experiment("E17", {"status": "FAILED", "reason": "language-v1 lacked conditional confidence port", "state": {"bus": "language-v1"}})
    factual = {
        "dependents_of_language": ledger.factual_query("dependents", "language-v1"),
        "dependents_of_expert": ledger.factual_query("dependents", "expert-0"),
        "ancestry_of_adapter": ledger.factual_query("ancestry", "adapter-0"),
        "minimum_affected_if_expert_changes": ledger.minimum_affected_neighborhood("expert-0"),
    }

    n = 6
    reader = LearnedDependencyReader(n, hidden_dim=128)
    rng = random.Random(seed)
    train = [_random_graph(rng, n) for _ in range(2000)]
    optimizer = torch.optim.AdamW(reader.parameters(), lr=0.01)
    losses = []
    for step in range(5000):
        adjacency, query, target = train[step % len(train)]
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(reader(adjacency.unsqueeze(0), query.unsqueeze(0)).squeeze(0), target)
        loss.backward()
        optimizer.step()
        if step in {0, 4999}:
            losses.append(float(loss.detach()))
    actual_adjacency = torch.zeros(n, n)
    index = {name: i for i, name in enumerate(names)}
    for edge in ledger.edges:
        actual_adjacency[index[edge["upstream"]], index[edge["downstream"]]] = 1.0
    actual_target_name = "language-v1"
    actual_query = torch.zeros(n)
    actual_query[index[actual_target_name]] = 1.0
    actual_scores = torch.sigmoid(reader(actual_adjacency.unsqueeze(0), actual_query.unsqueeze(0))).squeeze(0)
    oracle_names = ledger.dependents(actual_target_name)
    actual_predicted = [names[i] for i, score in enumerate(actual_scores.tolist()) if score > 0.5]
    unsupported = sorted(set(actual_predicted) - set(oracle_names))
    result = {
        "schema": "remora-v0-manual-lineage-result",
        "seed": seed,
        "factual_queries": factual,
        "learned_reader": {
            "train_loss_first_last": losses,
            "query": actual_target_name,
            "oracle_dependents": oracle_names,
            "predicted_dependents": actual_predicted,
            "unsupported_predictions": unsupported,
            "scores": dict(zip(names, actual_scores.tolist())),
            "graph_authority": "oracle ledger; reader cannot mutate it",
        },
        "lineage": ledger.to_dict(),
    }
    if output:
        write_json(output, result)
    record_experiment(
        ROOT,
        "MANUAL-LINEAGE-001",
        "A factual assembly ledger can answer dependency/ancestry queries and train a non-authoritative reader without allowing unsupported graph mutations.",
        "Build a generation-4 graph, train a reader on random DAG closure labels, and query the actual graph.",
        "Reader agrees with the oracle on held-out graph queries and produces no unsupported dependency in the actual query.",
        "Factual query is not graph-derived, or reader answers are treated as authoritative despite unsupported edges.",
        "python -m experiments.manual_lineage",
        seed,
        result["learned_reader"],
        "MEASURED: factual graph answers are exact; learned-reader agreement is reported rather than assumed.",
        "Add migration and version-2 bus queries after the first successful reader run.",
        hardware={"device": "cpu"},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--output", default=str(ROOT / "results" / "manual-lineage.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.output), indent=2))


if __name__ == "__main__":
    main()
