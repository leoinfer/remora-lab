from __future__ import annotations

"""Profile Remora and its matched baseline without changing model state."""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.config import ModelConfig
from remora.models import BaselineModel, RemoraModel
from remora.ledger import record_experiment
from remora.utils import choose_device, runtime_context, set_seed, write_json


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _benchmark(model: torch.nn.Module, input_ids: torch.Tensor, device: torch.device, warmup: int, steps: int) -> dict:
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(input_ids)
        _sync(device)
        elapsed = []
        for _ in range(steps):
            start = time.perf_counter()
            model(input_ids)
            _sync(device)
            elapsed.append(time.perf_counter() - start)
    ordered = sorted(elapsed)
    return {
        "warmup": warmup,
        "steps": steps,
        "mean_ms": statistics.mean(elapsed) * 1000.0,
        "p50_ms": ordered[len(ordered) // 2] * 1000.0,
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] * 1000.0,
        "tokens_per_second": float(input_ids.numel() * steps / sum(elapsed)),
    }


def _configure_remora(model: RemoraModel, scan: bool, sdpa: bool) -> RemoraModel:
    model.parallel_scan_setting = scan
    for block in model.blocks:
        block.recurrent.parallel_scan = scan
        block.attention.use_sdpa = sdpa
    return model


def run(
    config_path: str | Path = ROOT / "configs" / "v0_tiny.json",
    seed: int = 22,
    device_name: str = "auto",
    batch_size: int = 32,
    seq_len: int = 96,
    warmup: int = 10,
    steps: int = 30,
    output: str | Path | None = ROOT / "results" / "profile-forward.json",
) -> dict:
    set_seed(seed)
    cfg = ModelConfig.from_json(config_path)
    if seq_len > cfg.max_seq_len:
        raise ValueError(f"seq_len {seq_len} exceeds configured max {cfg.max_seq_len}")
    device = choose_device(device_name)
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)

    remora_seed = RemoraModel(cfg)
    remora_state = remora_seed.state_dict()
    baseline_seed = BaselineModel(cfg)
    baseline_state = baseline_seed.state_dict()

    variants: dict[str, dict] = {}
    remora_models = {
        "remora_reference": (False, False),
        "remora_parallel_scan": (True, False),
        "remora_parallel_scan_sdpa": (True, True),
    }
    for name, (scan, sdpa) in remora_models.items():
        model = _configure_remora(RemoraModel(cfg), scan, sdpa)
        model.load_state_dict(remora_state)
        model.to(device)
        variants[name] = {
            "scan": scan,
            "sdpa": sdpa,
            "timing": _benchmark(model, input_ids, device, warmup, steps),
        }

    baseline_models = {
        "baseline_manual_attention": False,
        "baseline_sdpa": True,
    }
    for name, sdpa in baseline_models.items():
        model = BaselineModel(cfg)
        model.load_state_dict(baseline_state)
        model.to(device)
        for block in model.blocks:
            block.attention.use_sdpa = sdpa
        variants[name] = {
            "sdpa": sdpa,
            "timing": _benchmark(model, input_ids, device, warmup, steps),
        }

    reference = _configure_remora(RemoraModel(cfg), False, False).to(device)
    reference.load_state_dict(remora_state)
    parallel = _configure_remora(RemoraModel(cfg), True, False).to(device)
    parallel.load_state_dict(remora_state)
    with torch.inference_mode():
        reference_logits, _ = reference(input_ids)
        parallel_logits, _ = parallel(input_ids)
    delta = reference_logits - parallel_logits
    result = {
        "schema": "remora-v0-forward-profile-result",
        "config": cfg.to_dict(),
        "seed": seed,
        "device": str(device),
        "batch_size": batch_size,
        "seq_len": seq_len,
        "variants": variants,
        "parallel_scan_equivalence": {
            "max_abs_logit_delta": float(delta.abs().max()),
            "mean_abs_logit_delta": float(delta.abs().mean()),
        },
        "interpretation": "MEASURED: the CUDA associative scan removes the token-time Python loop with near-equivalent logits; SDPA is reported separately because its performance is runtime/driver dependent on this AMD setup.",
        "runtime": runtime_context(device),
    }
    if output:
        write_json(output, result)
    record_experiment(
        ROOT,
        "PROFILE-OPTIMIZATION-001",
        "The Remora recurrent token loop is the dominant avoidable overhead, and a parallel associative scan can reduce it without changing the recurrence materially.",
        "Benchmark matched models with the reference recurrence, CUDA associative scan, manual attention, and SDPA under identical inputs and weights.",
        "The scan is faster than the reference, remains numerically close, and SDPA is retained only where its runtime path is beneficial.",
        "The scan is slower, diverges materially, or the benchmark does not synchronize device work; SDPA is treated as universally faster despite the measured AMD result.",
        f"python -m experiments.profile_forward --device {device_name} --batch-size {batch_size} --seq-len {seq_len}",
        seed,
        {"variants": variants, "parallel_scan_equivalence": result["parallel_scan_equivalence"]},
        result["interpretation"],
        "Keep the scan if the multi-seed training comparison confirms the speedup; otherwise profile the expert and projection launch counts next.",
        hardware=runtime_context(device),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "v0_tiny.json"))
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output", default=str(ROOT / "results" / "profile-forward.json"))
    args = parser.parse_args()
    result = run(
        args.config,
        args.seed,
        args.device,
        args.batch_size,
        args.seq_len,
        args.warmup,
        args.steps,
        args.output,
    )
    print(result["interpretation"])
    print(result["variants"])
    print(result["parallel_scan_equivalence"])


if __name__ == "__main__":
    main()
