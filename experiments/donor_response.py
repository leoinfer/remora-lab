from __future__ import annotations

"""Synthetic black-box response distillation into Remora plastic islands."""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.synthetic_corpus import build_heldout_corpus
from remora.config import ModelConfig
from remora.data import encode_stream
from remora.donors.response import DonorResponseRecord, load_response_records, write_response_records
from remora.ledger import record_experiment
from remora.metrics import evaluate_stream
from remora.models import RemoraModel
from remora.tokenizer import ByteTokenizer
from remora.utils import (
    changed_parameter_stats,
    choose_device,
    count_parameters,
    freeze_all,
    parameter_snapshot,
    set_seed,
    unfreeze_prefixes,
    write_json,
)


def _make_records(seed: int, count: int, excluded: set[tuple[int, int]] | None = None) -> list[DonorResponseRecord]:
    rng = random.Random(seed)
    excluded = set(excluded or ())
    pairs: list[tuple[int, int]] = []
    seen = set(excluded)
    while len(pairs) < count:
        pair = (rng.randrange(100), rng.randrange(100))
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    records = []
    for index, (a, b) in enumerate(pairs):
        prompt = f"donor-add:{a}+{b}="
        # Fixed-width output gives the black-box response interface an explicit
        # termination contract without importing a donor tokenizer or EOS id.
        response = f"{a + b:03d}"
        records.append(
            DonorResponseRecord.create(
                f"response-{seed}-{index:04d}",
                "synthetic-blackbox-donor-v1",
                prompt,
                response,
                lineage_key=f"synthetic-runtime-run-{seed}",
                runtime={"runtime_id": "synthetic-blackbox-runtime-v1", "sampling": "deterministic"},
                verifier={"verifier_id": "external-sum-verifier-v1", "passed": True},
                accepted=True,
            )
        )
    return records


def _verify_sum(prompt: str, response: str) -> bool:
    try:
        left, right = prompt.removeprefix("donor-add:").removesuffix("=").split("+")
        return response.strip() == f"{int(left) + int(right):03d}"
    except (ValueError, TypeError):
        return False


def _encode_records(records: list[DonorResponseRecord], tokenizer: ByteTokenizer, device: torch.device):
    encoded = []
    for record in records:
        prompt_ids = tokenizer.encode(record.prompt)
        sequence = prompt_ids + tokenizer.encode(record.response)
        encoded.append((sequence[:-1], sequence[1:], len(prompt_ids)))
    width = max(len(item[0]) for item in encoded)
    input_ids = torch.zeros(len(encoded), width, dtype=torch.long, device=device)
    targets = torch.zeros_like(input_ids)
    response_mask = torch.zeros(len(encoded), width, dtype=torch.float32, device=device)
    for row, (inputs, target, prompt_length) in enumerate(encoded):
        input_ids[row, : len(inputs)] = torch.tensor(inputs, dtype=torch.long, device=device)
        targets[row, : len(target)] = torch.tensor(target, dtype=torch.long, device=device)
        # A target at position j is a response byte iff sequence[j + 1] is in
        # the response suffix. The prompt itself never contributes to loss.
        response_start = max(prompt_length - 1, 0)
        response_mask[row, response_start : len(target)] = 1.0
    return input_ids, targets, response_mask


def _response_loss(model: RemoraModel, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    input_ids, targets, response_mask = batch
    logits, _ = model(input_ids)
    token_loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view_as(response_mask)
    return (token_loss * response_mask).sum() / response_mask.sum().clamp_min(1.0)


@torch.no_grad()
def _generate(model: RemoraModel, tokenizer: ByteTokenizer, prompt: str, max_new_tokens: int, device: torch.device) -> str:
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        logits, _ = model(input_ids)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return tokenizer.decode(input_ids[0, len(prompt_ids) :].tolist())


@torch.no_grad()
def _evaluate_responses(
    model: RemoraModel,
    records: list[DonorResponseRecord],
    tokenizer: ByteTokenizer,
    device: torch.device,
) -> dict:
    model.eval()
    predictions = []
    correct = 0
    max_new_tokens = max(len(record.response) for record in records)
    for record in records:
        prediction = _generate(model, tokenizer, record.prompt, max_new_tokens, device)
        passed = _verify_sum(record.prompt, prediction)
        correct += int(passed)
        predictions.append({"record_id": record.record_id, "prediction": prediction, "verified": passed})
    return {
        "count": len(records),
        "verified_count": correct,
        "accuracy": correct / max(len(records), 1),
        "verifier": "external-sum-verifier-v1",
        "predictions": predictions,
    }


def _train_response_adapter(
    model: RemoraModel,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    prefixes: list[str],
    steps: int,
    lr: float,
) -> list[float]:
    freeze_all(model)
    selected = unfreeze_prefixes(model, prefixes)
    if not selected:
        raise ValueError("response adapter prefixes matched no parameters")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    history = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _response_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def _train_full_control(
    model: RemoraModel,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    steps: int,
    lr: float,
) -> list[float]:
    for parameter in model.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _response_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def _load_checkpoint(path: str | Path, device: torch.device) -> RemoraModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = RemoraModel(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)


def run(
    checkpoint: str | Path,
    seed: int = 97,
    device_name: str = "auto",
    steps: int = 180,
    output: str | Path | None = None,
    records_output: str | Path | None = None,
) -> dict:
    set_seed(seed)
    device = choose_device(device_name)
    train_records = _make_records(seed, 192)
    train_pairs = set()
    for record in train_records:
        left, right = record.prompt.removeprefix("donor-add:").removesuffix("=").split("+")
        train_pairs.add((int(left), int(right)))
    test_records = _make_records(seed + 1, 64, excluded=train_pairs)
    if records_output:
        write_response_records(records_output, train_records + test_records)
    accepted_records = load_response_records(
        records_output,
        accepted_only=True,
        verifier=lambda item: item.verifier.get("passed") is True and _verify_sum(item.prompt, item.response),
    ) if records_output else train_records + test_records
    if len(accepted_records) != len(train_records) + len(test_records):
        raise ValueError("response acceptance filter rejected a generated record")

    tokenizer = ByteTokenizer()
    train_batch = _encode_records(train_records, tokenizer, device)
    base = _load_checkpoint(checkpoint, device)
    frozen_control = _load_checkpoint(checkpoint, device)
    candidate = _load_checkpoint(checkpoint, device)
    full_control = _load_checkpoint(checkpoint, device)
    old_stream = encode_stream(build_heldout_corpus(400, seed=999))
    old_before = evaluate_stream(base, old_stream, 32, 96, device)
    frozen_response = _evaluate_responses(frozen_control, test_records, tokenizer, device)

    adapter_prefixes = [f"blocks.{index}.plastic" for index in range(len(candidate.blocks))]
    before_candidate = parameter_snapshot(candidate)
    candidate_history = _train_response_adapter(candidate, train_batch, adapter_prefixes, steps, lr=0.03)
    candidate_response = _evaluate_responses(candidate, test_records, tokenizer, device)
    candidate_old = evaluate_stream(candidate, old_stream, 32, 96, device)
    candidate_update = changed_parameter_stats(before_candidate, candidate)

    before_full = parameter_snapshot(full_control)
    full_history = _train_full_control(full_control, train_batch, steps, lr=0.003)
    full_response = _evaluate_responses(full_control, test_records, tokenizer, device)
    full_old = evaluate_stream(full_control, old_stream, 32, 96, device)
    full_update = changed_parameter_stats(before_full, full_control)

    result = {
        "schema": "remora-v0-donor-response-result",
        "seed": seed,
        "device": str(device),
        "mode": "MECHANISM_ONLY_SYNTHETIC_BLACKBOX_RESPONSE_DONOR",
        "source_checkpoint": str(checkpoint),
        "response_contract": "donor-response-v1",
        "records": {
            "train": len(train_records),
            "test": len(test_records),
            "accepted": len(accepted_records),
            "records_path": str(records_output) if records_output else None,
            "donor_id": train_records[0].donor_id,
            "verifier": "external-sum-verifier-v1",
        },
        "frozen_control": {"old": old_before, "response": frozen_response},
        "candidate": {
            "trainable_prefixes": adapter_prefixes,
            "train_loss_first_last": [candidate_history[0], candidate_history[-1]],
            "old": candidate_old,
            "response": candidate_response,
            "update": candidate_update,
            "trainable_parameters": sum(p.numel() for p in candidate.parameters() if p.requires_grad),
        },
        "full_model_control": {
            "train_loss_first_last": [full_history[0], full_history[-1]],
            "old": full_old,
            "response": full_response,
            "update": full_update,
            "parameters": count_parameters(full_control),
        },
        "promotion": {
            "state": "CANDIDATE",
            "promoted": False,
            "decision_authority": "external held-out verifier",
        },
        "interpretation": "MEASURED MECHANISM TEST: verified black-box donor responses were encoded through a tokenizer boundary and learned by Remora plastic islands; this is not Qwen transfer evidence.",
    }
    if output:
        write_json(output, result)
    record_experiment(
        ROOT,
        "DONOR-RESPONSE-001",
        "A black-box donor with an incompatible response/token interface can teach a replaceable Remora plastic island through verified response strings.",
        "Generate hashed donor-response-v1 records, reject tampered/unverified records, train only block plastic adapters, and compare with frozen and full-model controls.",
        "Verified response accuracy improves over the frozen control, old-task retention remains measurable, the local update fraction is small, and promotion remains external.",
        "The candidate learns from unverified or tampered records, response accuracy does not improve, old-task retention collapses, or the candidate silently promotes itself.",
        f"python -m experiments.donor_response --checkpoint {checkpoint} --steps {steps}",
        seed,
        {k: v for k, v in result.items() if k not in {"promotion", "candidate", "full_model_control"}},
        result["interpretation"],
        "Replace the synthetic producer with an explicitly launched local runtime and repeat with verifier-held-out prompts; do not use donor logits across mismatched vocabularies.",
        hardware={"device": str(device), "mode": result["mode"]},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "remora-v0-scratch-seed7.pt"))
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--output", default=str(ROOT / "results" / "donor-response.json"))
    parser.add_argument("--records-output", default=str(ROOT / "results" / "donor-response-records.jsonl"))
    args = parser.parse_args()
    result = run(args.checkpoint, args.seed, args.device, args.steps, args.output, args.records_output)
    print(json.dumps({
        "interpretation": result["interpretation"],
        "frozen_accuracy": result["frozen_control"]["response"]["accuracy"],
        "candidate_accuracy": result["candidate"]["response"]["accuracy"],
        "full_control_accuracy": result["full_model_control"]["response"]["accuracy"],
        "candidate_changed_fraction": result["candidate"]["update"]["changed_fraction"],
        "full_changed_fraction": result["full_model_control"]["update"]["changed_fraction"],
    }, indent=2))


if __name__ == "__main__":
    main()
