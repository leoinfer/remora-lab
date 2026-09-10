from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.config import ModelConfig
from remora.models import build_model
from remora.utils import choose_device, count_parameters, set_seed, write_json
from remora.ledger import record_experiment


def run(seed: int = 17, steps: int = 120, device_name: str = "cpu", output: str | Path | None = None) -> dict:
    set_seed(seed)
    device = choose_device(device_name)
    cfg = ModelConfig(vocab_size=128, max_seq_len=32, d_model=64, n_layers=2, n_heads=4, d_ff=64, bus_dim=32, n_experts=2, adapter_dim=8, baseline_d_ff=160)
    # A fixed, repeated batch tests gradient plumbing and memorization, not generalization.
    base = torch.arange(32, device=device).long().remainder(96) + 32
    x = base.unsqueeze(0).repeat(4, 1)
    y = torch.roll(x, shifts=-1, dims=1)
    results = {}
    for kind in ("remora", "baseline"):
        model = build_model(kind, cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
        losses = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        results[kind] = {
            "parameters": count_parameters(model),
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "decreased": losses[-1] < losses[0],
            "losses": losses,
        }
    result = {"schema": "remora-v0-overfit-result", "seed": seed, "device": str(device), "steps": steps, "results": results, "caveat": "fixed-batch memorization gate; not evidence of reasoning or generalization"}
    if output:
        write_json(output, result)
    record_experiment(
        ROOT,
        "OVERFIT-001",
        "Both the modular model and matched baseline have working gradients and can memorize a fixed tiny batch.",
        f"Train each scratch-initialized model on one fixed batch for {steps} steps.",
        "Both losses decrease and remain finite.",
        "Either model has no finite loss decrease, indicating broken forward/backward plumbing.",
        "python -m experiments.overfit",
        seed,
        results,
        "MEASURED: fixed-batch overfit only.",
        "Proceed to held-out scratch pretraining if both arms pass.",
        hardware={"device": str(device)},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda"])
    parser.add_argument("--output", default=str(ROOT / "results" / "overfit.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.steps, args.device, args.output), indent=2))


if __name__ == "__main__":
    main()
