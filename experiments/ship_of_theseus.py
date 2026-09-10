from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.synthetic_corpus import build_heldout_corpus, build_target_corpus
from experiments.module_replacement import _adapt, _evaluate
from remora.config import ModelConfig
from remora.data import encode_stream
from remora.ledger import record_experiment, record_failure
from remora.manual.ledger import AssemblyLedger, ModuleRecord
from remora.models import build_model
from remora.modules import SwiGLUExpert
from remora.utils import choose_device, count_parameters, parameter_snapshot, set_seed, changed_parameter_stats, write_json


def _load(checkpoint: str | Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model("remora", ModelConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    return model.to(device), payload


def _fingerprint(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _record(ledger: AssemblyLedger, module_id: str, version: str, module: torch.nn.Module, ancestry: list[str], status: str = "ACTIVE") -> None:
    ledger.register_module(ModuleRecord(
        module_id=module_id,
        version=version,
        inputs=["language-v1"],
        outputs=["language-v1"],
        state_owner=module_id,
        parameter_count=count_parameters(module),
        ancestry=ancestry,
        status=status,
        content_hash=_fingerprint(module),
    ))


def run(
    checkpoint: str | Path,
    seed: int = 83,
    steps_per_generation: int = 40,
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
    batch_size = 32
    ledger = AssemblyLedger(generation=0)
    bus = model.bus
    _record(ledger, "language-v1", "v1", bus, ["language-v1@g0"])
    _record(ledger, "lm-head", "v1", model.lm_head, ["lm-head@g0"])
    current_ids: dict[int, str] = {}
    for layer in range(cfg.n_layers):
        module_id = f"expert-layer{layer}-g0"
        current_ids[layer] = module_id
        _record(ledger, module_id, "expert-v1", model.blocks[layer].experts.experts[0], [f"{module_id}@g0"])
        ledger.add_dependency("language-v1", module_id)
        ledger.add_dependency(module_id, "lm-head")

    generations: list[dict] = []
    initial_scores = _evaluate(model, old_stream, target_stream, seq_len, batch_size, device)
    for generation, layer in enumerate(range(cfg.n_layers), start=1):
        old_id = current_ids[layer]
        old_record = ledger.modules[old_id]
        old_record.status = "DORMANT"
        new_id = f"expert-layer{layer}-g{generation}"
        replacement = SwiGLUExpert(cfg.bus_dim, cfg.d_ff, cfg.bus_dim).to(device)
        old_expert = model.blocks[layer].experts.replace_expert(0, replacement)
        before_replace = _evaluate(model, old_stream, target_stream, seq_len, batch_size, device)
        before_local = parameter_snapshot(model)
        scope = [f"blocks.{layer}.experts.experts.0"]
        _, history = _adapt(
            model,
            target_stream,
            seq_len,
            batch_size,
            device,
            scope,
            steps_per_generation,
            seed + generation,
            2e-3,
            retention_stream=old_stream,
            retention_weight=1.0,
        )
        after = _evaluate(model, old_stream, target_stream, seq_len, batch_size, device)
        local_stats = changed_parameter_stats(before_local, model)
        _record(ledger, new_id, "expert-v1", model.blocks[layer].experts.experts[0], [old_id, f"{new_id}@g{generation}"])
        ledger.add_dependency("language-v1", new_id)
        ledger.add_dependency(new_id, "lm-head")
        old_record.replacements.append(new_id)
        current_ids[layer] = new_id
        generations.append({
            "generation": generation,
            "layer": layer,
            "replaced": old_id,
            "installed": new_id,
            "old_module_parameters": count_parameters(old_expert),
            "new_module_parameters": count_parameters(replacement),
            "before_replacement": before_replace,
            "after_local_rehearsal": after,
            "local_update": local_stats,
            "adaptation_loss_first_last": [history[0], history[-1]],
        })
        ledger.generation = generation

    final_scores = _evaluate(model, old_stream, target_stream, seq_len, batch_size, device)
    result = {
        "schema": "remora-v0-ship-of-theseus-result",
        "checkpoint": str(checkpoint),
        "seed": seed,
        "device": str(device),
        "steps_per_generation": steps_per_generation,
        "initial_scores": initial_scores,
        "generations": generations,
        "final_scores": final_scores,
        "lineage": {
            "current_modules": current_ids,
            "generation": ledger.generation,
            "assembly": ledger.to_dict(),
        },
        "retention_gate": {
            "threshold": initial_scores["old"]["loss"] * 2.0,
            "passed": final_scores["old"]["loss"] <= initial_scores["old"]["loss"] * 2.0,
        },
        "interpretation": "MEASURED: sequential local replacements retained a coherent factual ancestry graph; capability retention is judged against the fixed old stream and is not inferred from lineage alone.",
    }
    if output:
        out = Path(output)
        out_checkpoint = out.with_suffix(".pt")
        torch.save({**source, "state_dict": model.state_dict(), "ship_of_theseus_result": result}, out_checkpoint)
        result["output_checkpoint"] = str(out_checkpoint)
        write_json(out, result)
        ledger.save(out.with_name(out.stem + "-assembly.json"))
    if not result["retention_gate"]["passed"]:
        record_failure(
            ROOT,
            "SHIP-THESEUS-001",
            cfg.to_dict(),
            {"old_stream": "heldout synthetic corpus", "target_stream": "target synthetic corpus", "steps_per_generation": steps_per_generation},
            seed,
            f"final old loss {final_scores['old']['loss']:.6f} exceeded retention threshold {initial_scores['old']['loss'] * 2.0:.6f}",
            "Repeated replacement perturbations accumulate even when each individual replacement uses old-task rehearsal.",
            "Reduce replacement rate, add inter-generation replay/consolidation, or replace router/bus ports before claiming whole-system replacement.",
            runtime={"device": str(device)},
        )
    record_experiment(
        ROOT,
        "SHIP-THESEUS-001",
        "A sequence of locally adapted expert replacements can preserve useful behavior while retaining coherent ancestry and dependency records.",
        f"Replace expert-0 in each of {cfg.n_layers} layers sequentially; train only the new module for {steps_per_generation} steps with old-stream rehearsal.",
        "Final old-task loss remains within 2x the initial loss, target loss remains finite, and each generation has a graph-linked ancestor.",
        "Final old-task loss exceeds the retention gate, a generation requires broad updates, or lineage cannot connect every replacement to its predecessor.",
        f"python -m experiments.ship_of_theseus --checkpoint {checkpoint} --steps-per-generation {steps_per_generation}",
        seed,
        result,
        result["interpretation"],
        "If retention passes, run a longer generation chain with an untouched module control; if it fails, preserve the chain as a cumulative replacement limit.",
        hardware={"device": str(device)},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--steps-per-generation", type=int, default=40)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default=str(ROOT / "results" / "ship-of-theseus.json"))
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.seed, args.steps_per_generation, args.device, args.output), indent=2))


if __name__ == "__main__":
    main()
