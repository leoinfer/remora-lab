from __future__ import annotations

"""Re-run one real failed mechanism after a changed Remora system state.

The original module-replacement failure is intentionally read as an
immutable artifact.  This experiment uses the queue to rank that same
candidate after the update policy and model age have changed, then invokes
the real local replacement harness.  The synthetic queue demo remains a
separate mechanism control in ``experiments.resurrection``.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.module_replacement import run as run_module_replacement
from remora.evolution import Candidate, ResurrectionQueue
from remora.ledger import record_experiment
from remora.utils import write_json


def _candidate_from_failure(failure: dict) -> Candidate:
    scope = failure.get("scope", {})
    return Candidate(
        candidate_id="swiGLU-expert-local-adaptation",
        status="REJECTED_UNDER_CONDITIONS_X",
        expected_upside=0.85,
        uncertainty=0.75,
        novelty=0.60,
        test_cost=1.0,
        failure_state={
            "checkpoint_age_stages": 0,
            "retention_weight": float(failure.get("retention_weight") or 0.0),
            "replacement": scope.get("replaced", "blocks.1.experts.experts.0"),
            "training_policy": "target_only",
        },
        failure_reason=(
            "target-only replacement reached target loss "
            f"{failure['replaced_after_local_training']['target']['loss']:.6f} "
            "but exceeded the old-task retention gate"
        ),
    )


def run(
    failure_path: str | Path = ROOT / "results" / "module-replacement-v1-failure.json",
    aged_checkpoint: str | Path = ROOT / "checkpoints" / "lifetime-compounding-remora_local_rehearsal-seed7.pt",
    seed: int = 7,
    steps: int = 120,
    device_name: str = "auto",
    output: str | Path | None = ROOT / "results" / "resurrection-real-v1.json",
) -> dict:
    failure_path = Path(failure_path)
    aged_checkpoint = Path(aged_checkpoint)
    with failure_path.open() as handle:
        failure = json.load(handle)

    candidate = _candidate_from_failure(failure)
    queue = ResurrectionQueue()
    queue.add(candidate)
    old_state = dict(candidate.failure_state)
    new_state = {
        "checkpoint_age_stages": 5,
        "retention_weight": 1.0,
        "replacement": old_state["replacement"],
        "training_policy": "target_plus_old_rehearsal",
    }
    before = queue.priorities(old_state)
    after = queue.priorities(new_state)

    if not aged_checkpoint.exists():
        raise FileNotFoundError(f"aged checkpoint is required for real resurrection: {aged_checkpoint}")
    trial_output = ROOT / "results" / "resurrection-real-module-replacement.json"
    trial = run_module_replacement(
        checkpoint=aged_checkpoint,
        seed=seed,
        steps=steps,
        device_name=device_name,
        output=trial_output,
        retention_weight=1.0,
    )
    retention_gate = trial["replaced_after_local_training"]["old"]["loss"] <= trial["original"]["old"]["loss"] * 2.0
    target_improved = trial["replaced_after_local_training"]["target"]["loss"] < trial["replaced_before_local_training"]["target"]["loss"]
    actual_success = bool(retention_gate and target_improved)
    candidate.history.append({
        "state": new_state,
        "priority_before": before[0]["priority"],
        "priority_after": after[0]["priority"],
        "trial_result": str(trial_output),
        "success": actual_success,
    })

    result = {
        "schema": "remora-v1-real-resurrection-result",
        "experiment_family": "RESURRECTION-REAL-001+",
        "source_failure": {
            "path": str(failure_path),
            "immutable": True,
            "failure_state": old_state,
            "failure_reason": candidate.failure_reason,
        },
        "changed_context": new_state,
        "queue_before_context_change": before,
        "queue_after_context_change": after,
        "actual_trial": {
            "module_replacement_result": str(trial_output),
            "checkpoint": str(aged_checkpoint),
            "retention_gate_passed": retention_gate,
            "target_improved": target_improved,
            "success": actual_success,
            "measured": trial,
        },
        "queue": queue.to_dict(),
        "interpretation": (
            "MEASURED: a real previously failed replacement was re-run after "
            "the retention policy and model age changed; success is conditional "
            "on both target recovery and the fixed old-stream gate."
            if actual_success
            else "MEASURED FAILURE: the queue correctly reconsidered a real failed "
            "candidate, but the changed context did not make the candidate pass "
            "both target and retention gates."
        ),
    }
    if output:
        write_json(output, result)
    record_experiment(
        ROOT,
        "RESURRECTION-REAL-001",
        "A real failed local replacement should be reconsidered when the surrounding system state removes or weakens its recorded failure condition.",
        "Take the immutable target-only SwiGLU replacement failure, increase model age to five lifetime stages, switch to old-task rehearsal, and actually rerun the same replacement candidate.",
        "The queue priority rises under the changed context and the real rerun either passes both target recovery and old-task retention or produces a second explicit failure.",
        "The queue does not react to the changed failure condition, the candidate is silently changed, or the rerun is reported as a win without both gates.",
        f"python -m experiments.resurrection_real --failure {failure_path} --aged-checkpoint {aged_checkpoint} --seed {seed}",
        seed,
        {
            "priority_before": before,
            "priority_after": after,
            "retention_gate_passed": retention_gate,
            "target_improved": target_improved,
            "actual_success": actual_success,
            "trial_output": str(trial_output),
        },
        result["interpretation"],
        "Keep the candidate active only if the measured gates pass; otherwise preserve the second failure and change the surrounding interface or replay policy before another retest.",
        hardware={"device": device_name, "mode": "real_failure_resurrection"},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure", default=str(ROOT / "results" / "module-replacement-v1-failure.json"))
    parser.add_argument("--aged-checkpoint", default=str(ROOT / "checkpoints" / "lifetime-compounding-remora_local_rehearsal-seed7.pt"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default=str(ROOT / "results" / "resurrection-real-v1.json"))
    args = parser.parse_args()
    result = run(args.failure, args.aged_checkpoint, args.seed, args.steps, args.device, args.output)
    print(json.dumps({
        "interpretation": result["interpretation"],
        "queue_before": result["queue_before_context_change"],
        "queue_after": result["queue_after_context_change"],
        "actual_trial": result["actual_trial"],
    }, indent=2))


if __name__ == "__main__":
    main()
