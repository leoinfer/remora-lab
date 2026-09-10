from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .utils import append_jsonl, git_hash, runtime_context


def record_experiment(
    repo_root: str | Path,
    experiment_id: str,
    hypothesis: str,
    change: str,
    expected_result: str,
    falsification_condition: str,
    command: str,
    seed: int,
    metrics: dict,
    conclusion: str,
    next_action: str,
    status: str = "COMPLETE",
    hardware: dict | None = None,
) -> dict:
    repo_root = Path(repo_root)
    record = {
        "experiment_id": experiment_id,
        "hypothesis": hypothesis,
        "change": change,
        "expected_result": expected_result,
        "falsification_condition": falsification_condition,
        "command": command,
        "seed": seed,
        "git_hash": git_hash(repo_root),
        "runtime": hardware or runtime_context(__import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")),
        "metrics": metrics,
        "conclusion": conclusion,
        "next_action": next_action,
        "status": status,
    }
    append_jsonl(repo_root / "ledger" / "experiments.jsonl", record)
    return record


def record_failure(
    repo_root: str | Path,
    experiment_id: str,
    model_config: dict,
    data_config: dict,
    seed: int,
    what_failed: str,
    current_hypothesis: str,
    resurrection_conditions: str,
    runtime: dict | None = None,
) -> dict:
    """Append a structured negative result without removing the main record."""

    repo_root = Path(repo_root)
    record = {
        "schema": "remora-v0-failure-record-v1",
        "experiment_id": experiment_id,
        "git_hash": git_hash(repo_root),
        "model_config": model_config,
        "data_config": data_config,
        "seed": seed,
        "runtime": runtime or runtime_context(__import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")),
        "what_failed": what_failed,
        "current_hypothesis": current_hypothesis,
        "resurrection_conditions": resurrection_conditions,
    }
    append_jsonl(repo_root / "ledger" / "failures.jsonl", record)
    return record
