from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.synthetic_corpus import build_language_corpus, build_target_corpus
from remora.config import ModelConfig
from remora.data import encode_stream
from remora.models import build_model
from remora.utils import choose_device, set_seed, write_json
from remora.ledger import record_experiment


def _load(checkpoint: str | Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model("remora", ModelConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    return model.to(device), payload


@torch.no_grad()
def _route_mean(model, stream: torch.Tensor, device: torch.device, seq_len: int, max_batches: int = 12) -> torch.Tensor:
    model.eval()
    rows: list[torch.Tensor] = []
    width = seq_len
    for index in range(max_batches):
        start = index * width
        if start + width > stream.numel():
            break
        probe = stream[start : start + width].unsqueeze(0).to(device)
        _, _, aux = model(probe, return_aux=True)
        rows.append(torch.stack([weights.mean(dim=(0, 1)).cpu() for weights in aux["route_weights"]]))
    if not rows:
        raise ValueError("stream is too short for route probes")
    return torch.stack(rows).mean(0)


def _score(language: torch.Tensor, target: torch.Tensor) -> dict:
    delta = (language - target).abs()
    return {
        "language_mean": language.tolist(),
        "target_mean": target.tolist(),
        "mean_absolute_task_shift": float(delta.mean()),
        "max_task_shift": float(delta.max()),
    }


def run(
    checkpoint: str | Path,
    seed: int = 7,
    device_name: str = "auto",
    output: str | Path | None = None,
) -> dict:
    set_seed(seed)
    device = choose_device(device_name)
    model, source = _load(checkpoint, device)
    cfg = model.cfg
    language = encode_stream(build_language_corpus(500, seed=source["seed"]))
    target = encode_stream(build_target_corpus(600, seed=seed + 200))
    seq_len = min(96, cfg.max_seq_len)
    trained_language = _route_mean(model, language, device, seq_len)
    trained_target = _route_mean(model, target, device, seq_len)

    # Same architecture and seed policy, but random weights: controls for
    # token-distribution geometry that exists before training.
    set_seed(source["seed"])
    fresh = build_model("remora", cfg).to(device)
    fresh_language = _route_mean(fresh, language, device, seq_len)
    fresh_target = _route_mean(fresh, target, device, seq_len)
    trained_score = _score(trained_language, trained_target)
    fresh_score = _score(fresh_language, fresh_target)
    result = {
        "schema": "remora-v0-specialization-result",
        "checkpoint": str(checkpoint),
        "device": str(device),
        "trained": trained_score,
        "fresh_random_control": fresh_score,
        "derived": {
            "task_shift_gain_over_fresh": trained_score["mean_absolute_task_shift"] / max(fresh_score["mean_absolute_task_shift"], 1e-9),
            "trained_router_is_task_selective": trained_score["mean_absolute_task_shift"] > fresh_score["mean_absolute_task_shift"],
        },
        "interpretation": "MEASURED/DERIVED: compare trained-vs-fresh task route shifts; a router existing in the graph is not treated as specialization evidence.",
    }
    if output:
        write_json(output, result)
    record_experiment(
        ROOT,
        "SPECIALIZATION-001",
        "Joint training will make routed experts selectively active for different synthetic task streams beyond the shift induced by a random router.",
        "Measure per-layer expert loads on language and target streams for the trained checkpoint and an identically configured fresh random model.",
        "The trained model shows a larger task-conditioned route shift than the fresh control and the result is reported per layer.",
        "The trained route shift is no larger than fresh control or all claims rely only on average balance/entropy.",
        f"python -m experiments.specialization --checkpoint {checkpoint}",
        seed,
        result,
        result["interpretation"],
        "If selectivity is weak, add an explicit load/task auxiliary objective or change expert granularity and rerun against a shuffled-router control.",
        hardware={"device": str(device)},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default=str(ROOT / "results" / "specialization.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.seed, args.device, args.output), indent=2))


if __name__ == "__main__":
    main()
