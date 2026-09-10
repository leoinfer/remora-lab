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

from environments.synthetic_corpus import build_heldout_corpus, build_target_corpus
from remora.config import ModelConfig
from remora.data import encode_stream
from remora.ledger import record_experiment
from remora.metrics import evaluate_stream
from remora.models import build_model
from remora.utils import choose_device, set_seed, write_json


class ZeroExpert(nn.Module):
    """A causal ablation with the same expert-v1 tensor port."""

    interface_version = "expert-v1"

    def __init__(self, width: int):
        super().__init__()
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def interface_signature(self) -> dict:
        return {"version": self.interface_version, "input_dim": self.width, "output_dim": self.width}


def _load(checkpoint: str | Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model("remora", ModelConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    return model.to(device), payload


def _evaluate_short(model, old_stream, target_stream, device, seq_len: int) -> dict:
    # A bounded intervention screen; the full held-out runs remain the primary
    # capability measurements in the training/replacement artifacts.
    old = old_stream[: seq_len * 8]
    target = target_stream[: seq_len * 8]
    return {
        "old": evaluate_stream(model, old, 8, seq_len, device),
        "target": evaluate_stream(model, target, 8, seq_len, device),
    }


def run(
    checkpoint: str | Path,
    seed: int = 79,
    device_name: str = "auto",
    output: str | Path | None = None,
) -> dict:
    set_seed(seed)
    device = choose_device(device_name)
    model, source = _load(checkpoint, device)
    cfg = model.cfg
    old_stream = encode_stream(build_heldout_corpus(500, seed=source["seed"] + 1000))
    target_stream = encode_stream(build_target_corpus(600, seed=seed + 200))
    seq_len = min(96, cfg.max_seq_len)
    original = _evaluate_short(model, old_stream, target_stream, device, seq_len)
    ablations: list[dict] = []
    for layer_index in range(cfg.n_layers):
        for expert_index in range(cfg.n_experts):
            intervened = copy.deepcopy(model)
            intervened.blocks[layer_index].experts.replace_expert(
                expert_index, ZeroExpert(cfg.bus_dim).to(device)
            )
            scores = _evaluate_short(intervened, old_stream, target_stream, device, seq_len)
            ablations.append({
                "layer": layer_index,
                "expert": expert_index,
                "old_loss_delta": scores["old"]["loss"] - original["old"]["loss"],
                "target_loss_delta": scores["target"]["loss"] - original["target"]["loss"],
                "scores": scores,
            })
            del intervened

    ranked = sorted(ablations, key=lambda row: row["target_loss_delta"], reverse=True)
    pair = None
    if len(ranked) >= 2:
        first, second = ranked[:2]
        intervened = copy.deepcopy(model)
        intervened.blocks[first["layer"]].experts.replace_expert(first["expert"], ZeroExpert(cfg.bus_dim).to(device))
        intervened.blocks[second["layer"]].experts.replace_expert(second["expert"], ZeroExpert(cfg.bus_dim).to(device))
        scores = _evaluate_short(intervened, old_stream, target_stream, device, seq_len)
        pair_delta = scores["target"]["loss"] - original["target"]["loss"]
        pair = {
            "members": [[first["layer"], first["expert"]], [second["layer"], second["expert"]]],
            "scores": scores,
            "target_loss_delta": pair_delta,
            "interaction_delta": pair_delta - first["target_loss_delta"] - second["target_loss_delta"],
        }
        del intervened

    result = {
        "schema": "remora-v0-credit-assignment-result",
        "checkpoint": str(checkpoint),
        "device": str(device),
        "original": original,
        "single_module_ablations": ranked,
        "top_pair_intervention": pair,
        "interpretation": "MEASURED INTERVENTION TEST: responsibility is ranked by replace/ablate changes in held-out loss; learned self-attribution is not used.",
    }
    if output:
        write_json(output, result)
    record_experiment(
        ROOT,
        "CREDIT-ASSIGNMENT-001",
        "Ablation interventions can identify which used experts causally affect a target failure or capability.",
        "Replace each routed expert with a zero-output expert, measure old/target loss deltas, and test interaction for the two strongest target effects.",
        "At least one intervention changes target loss and the pair interaction is reported rather than attributing responsibility from router weights alone.",
        "All ablations are behaviorally identical, or responsibility is inferred without an intervention control.",
        f"python -m experiments.credit_assignment --checkpoint {checkpoint}",
        seed,
        result,
        result["interpretation"],
        "Use the ranking to target a replacement/replay experiment and compare it with a random-module intervention.",
        hardware={"device": str(device)},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=79)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default=str(ROOT / "results" / "credit-assignment.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.seed, args.device, args.output), indent=2))


if __name__ == "__main__":
    main()
