from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.synthetic_corpus import build_heldout_corpus, build_language_corpus
from remora.config import ModelConfig
from remora.data import encode_stream, sample_batch
from remora.ledger import record_experiment
from remora.metrics import evaluate_stream, model_summary
from remora.models import build_model
from remora.plot import write_line_svg
from remora.routing import route_summary
from remora.utils import choose_device, git_hash, runtime_context, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["remora", "baseline"], required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "v0_tiny.json"))
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-root", default=str(ROOT))
    parser.add_argument("--train-lines", type=int, default=4000)
    parser.add_argument("--heldout-lines", type=int, default=700)
    return parser.parse_args()


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    cfg = ModelConfig.from_json(args.config)
    device = choose_device(args.device)
    seq_len = min(args.seq_len, cfg.max_seq_len)
    tokenizer_stream = encode_stream(build_language_corpus(args.train_lines, seed=args.seed))
    heldout_stream = encode_stream(build_heldout_corpus(args.heldout_lines, seed=args.seed + 1000))
    model = build_model(args.model, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    batch_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    run_id = args.run_id or f"{args.model}-seed{args.seed}-{int(time.time())}"
    output_root = Path(args.output_root)
    (output_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_root / "results").mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    eval_history: list[dict] = []
    started = time.perf_counter()
    print(json.dumps({"run_id": run_id, "model": args.model, "device": str(device), "summary": model_summary(model)}))
    for step in range(1, args.steps + 1):
        step_started = time.perf_counter()
        model.train()
        x, y = sample_batch(tokenizer_stream, args.batch_size, seq_len, device, batch_generator)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        row = {
            "step": step,
            "train_loss": float(loss.detach()),
            "grad_norm": grad_norm,
            "step_seconds": time.perf_counter() - step_started,
        }
        history.append(row)
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            evaluation = evaluate_stream(model, heldout_stream, args.batch_size, seq_len, device)
            evaluation["step"] = step
            eval_history.append(evaluation)
            print(json.dumps({**row, "heldout": evaluation}), flush=True)
    final_eval = eval_history[-1]
    summary = model_summary(model)
    route = {}
    if args.model == "remora":
        model.eval()
        with torch.no_grad():
            probe = heldout_stream[:seq_len].unsqueeze(0).to(device)
            _, _, aux = model(probe, return_aux=True)
        route = route_summary(aux["route_weights"])
    checkpoint = {
        "schema": "remora-v0-checkpoint",
        "model_type": args.model,
        "config": cfg.to_dict(),
        "seed": args.seed,
        "git_hash": git_hash(ROOT),
        "runtime": runtime_context(device),
        "summary": summary,
        "route_summary": route,
        "train_corpus": {"lines": args.train_lines, "seed": args.seed, "tokens": int(tokenizer_stream.numel())},
        "heldout_corpus": {"lines": args.heldout_lines, "seed": args.seed + 1000, "tokens": int(heldout_stream.numel())},
        "history": history,
        "eval_history": eval_history,
        "elapsed_seconds": time.perf_counter() - started,
        "state_dict": model.state_dict(),
    }
    checkpoint_path = output_root / "checkpoints" / f"{run_id}.pt"
    torch.save(checkpoint, checkpoint_path)
    result = {
        "schema": "remora-v0-training-result",
        "run_id": run_id,
        "model_type": args.model,
        "config": cfg.to_dict(),
        "seed": args.seed,
        "git_hash": git_hash(ROOT),
        "runtime": runtime_context(device),
        "summary": summary,
        "route_summary": route,
        "final_eval": final_eval,
        "history": history,
        "eval_history": eval_history,
        "checkpoint": str(checkpoint_path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    result_path = output_root / "results" / f"{run_id}.json"
    write_json(result_path, result)
    write_line_svg(
        output_root / "results" / f"{run_id}-loss.svg",
        {
            "train loss": [x["train_loss"] for x in history],
            "heldout loss": [x["loss"] for x in eval_history],
        },
        title=f"{args.model} loss ({run_id})",
    )
    record_experiment(
        ROOT,
        experiment_id=f"PRETRAIN-{args.model.upper()}-{run_id}",
        hypothesis="A scratch-initialized small model learns the fixed held-out stream.",
        change=f"Train {args.model} for {args.steps} steps with seed {args.seed}.",
        expected_result="Held-out loss decreases from initialization.",
        falsification_condition="Held-out loss is flat or diverges while the matched control trains.",
        command=" ".join(sys.argv),
        seed=args.seed,
        metrics={"summary": summary, "final_eval": final_eval, "route_summary": route, "checkpoint": str(checkpoint_path)},
        conclusion="MEASURED: held-out loss decreased." if eval_history[-1]["loss"] < eval_history[0]["loss"] else "MEASURED: no held-out improvement.",
        next_action="Use the checkpoint for controlled adaptation and replacement tests.",
        hardware=runtime_context(device),
    )
    return result


if __name__ == "__main__":
    train(parse_args())
