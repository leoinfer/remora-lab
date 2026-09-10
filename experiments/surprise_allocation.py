from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.ledger import record_experiment
from remora.utils import write_json
from remora.world_model import SurpriseTracker


def _high_prediction(tracker: SurpriseTracker, cluster: str, repeat: int = 1) -> None:
    for i in range(repeat):
        tracker.observe(torch.tensor([[0.95, 0.05]]), 1, cluster, {"repeat": i})


def run(output: str | Path | None = None) -> dict:
    low = SurpriseTracker()
    for i in range(4):
        low.observe(torch.tensor([[0.95, 0.05]]), 0, f"low-{i}")

    isolated = SurpriseTracker()
    _high_prediction(isolated, "one-run", repeat=1)

    duplicate = SurpriseTracker()
    _high_prediction(duplicate, "same-broken-program", repeat=24)

    independent = SurpriseTracker()
    for i in range(24):
        _high_prediction(independent, f"run-{i}")

    result = {
        "schema": "remora-v0-surprise-allocation-result",
        "hypothesis": "Persistent high surprise across independent clusters should allocate more compute than a duplicate or isolated anomaly.",
        "scenarios": {
            "low": low.summary(),
            "isolated": isolated.summary(),
            "duplicate": duplicate.summary(),
            "independent": independent.summary(),
        },
        "interpretation": "MEASURED MECHANISM TEST: cluster-aware surprise distinguishes repeated correlated failure from persistent independent contradiction; this allocator is a toy policy, not a calibrated world model.",
    }
    if output:
        write_json(output, result)
    record_experiment(
        ROOT,
        "SURPRISE-ALLOCATION-001",
        result["hypothesis"],
        "Aggregate categorical NLL by cluster and map low/moderate/high persistent surprise to bounded compute budgets.",
        "Independent persistent surprise receives a higher allocation than duplicate evidence, while low surprise stays cheap.",
        "Duplicate observations receive the same high allocation as independent clusters, or low-surprise observations trigger expensive inspection.",
        "python -m experiments.surprise_allocation",
        0,
        result["scenarios"],
        result["interpretation"],
        "Replace the toy thresholds with calibration data and connect allocation to an actual retrieval/reasoning budget.",
        hardware={"device": "cpu"},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "results" / "surprise-allocation.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
