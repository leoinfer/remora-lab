from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.evolution import Candidate, ResurrectionQueue
from remora.ledger import record_experiment
from remora.utils import write_json


def run(output: str | Path | None = None) -> dict:
    queue = ResurrectionQueue()
    queue.add(Candidate(
        "expert-small-bus", "REJECTED_UNDER_CONDITIONS_X", 0.8, 0.8, 0.9, 1.0,
        {"bus_dim": 32, "router": "dense", "task": "conditional"},
        "capacity bottleneck in language-v1",
    ))
    queue.add(Candidate(
        "safe-baseline", "ACTIVE", 0.35, 0.2, 0.1, 1.0,
        {"bus_dim": 96, "router": "dense", "task": "conditional"},
        "none",
    ))
    old_state = {"bus_dim": 32, "router": "dense", "task": "conditional"}
    new_state = {"bus_dim": 96, "router": "sparse", "task": "conditional"}
    old_priority = queue.priorities(old_state)
    new_priority = queue.priorities(new_state)
    old_quality = {"expert-small-bus": 0.41, "safe-baseline": 0.58}
    new_quality = {"expert-small-bus": 0.76, "safe-baseline": 0.61}
    result = {
        "schema": "remora-v0-resurrection-result",
        "queue_before_context_change": old_priority,
        "queue_after_context_change": new_priority,
        "synthetic_candidate_trial": {"old_state_quality": old_quality, "new_state_quality": new_quality},
        "success": new_quality["expert-small-bus"] > new_quality["safe-baseline"],
        "interpretation": "MODELED/SYNTHETIC: mechanism test; candidate quality is a controlled environment signal, not a neural benchmark.",
        "queue": queue.to_dict(),
    }
    if output:
        write_json(output, result)
    record_experiment(
        ROOT,
        "RESURRECTION-001",
        "A candidate failure is conditional on surrounding state and should receive higher retest priority after that state changes.",
        "Record a small-bus failure, change bus/router context, and run a controlled candidate-quality trial.",
        "Priority rises and the formerly bad candidate becomes competitive under the changed context.",
        "Failure is promoted to permanent rejection or priority does not react to changed failure conditions.",
        "python -m experiments.resurrection",
        0,
        result,
        "MEASURED mechanism output; synthetic quality is not evidence of a neural architecture win.",
        "Replace modeled quality with an actual paired expert trial once checkpoints exist.",
        hardware={"device": "cpu"},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "results" / "resurrection.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
