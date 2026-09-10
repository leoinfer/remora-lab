from __future__ import annotations

"""Controlled response distillation with a small, externally verified skill.

The arithmetic response experiment is intentionally retained as a hard
negative result.  This companion experiment asks a narrower question first:
can a frozen donor's useful, discrete lookup behavior enter a replaceable
Remora island, survive old-stream rehearsal, and transfer to a modest prompt
template change?  The donor is still synthetic; no resident large model is
loaded.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.synthetic_corpus import build_heldout_corpus
from experiments.donor_response import _encode_records, _load_checkpoint
from remora.data import contiguous_batches, encode_stream
from remora.donors.response import DonorResponseRecord, load_response_records, write_response_records
from remora.ledger import record_experiment, record_failure
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


_KEYS = (
    "anchor",
    "basin",
    "copper",
    "delta",
    "ember",
    "fathom",
    "granite",
    "harbor",
)
_LABELS = "ABCDEFGH"


def _make_records(
    seed: int,
    count: int,
    *,
    template: str,
    prefix: str,
) -> list[DonorResponseRecord]:
    generator = torch.Generator().manual_seed(seed)
    records = []
    for index in range(count):
        key_index = int(torch.randint(len(_KEYS), (), generator=generator))
        key = _KEYS[key_index]
        prompt = template.format(key=key)
        response = _LABELS[key_index]
        records.append(
            DonorResponseRecord.create(
                f"{prefix}-{seed}-{index:04d}",
                "synthetic-lookup-donor-v1",
                prompt,
                response,
                lineage_key=f"synthetic-lookup-lineage-{seed}",
                runtime={"runtime_id": "synthetic-lookup-runtime-v1", "sampling": "deterministic"},
                verifier={
                    "verifier_id": "external-lookup-verifier-v1",
                    "passed": True,
                    "key": key,
                    "label": response,
                },
                accepted=True,
            )
        )
    return records


def _verify_lookup(prompt: str, response: str) -> bool:
    for index, key in enumerate(_KEYS):
        if key in prompt:
            return response.strip() == _LABELS[index]
    return False


@torch.no_grad()
def _evaluate_lookup(
    model: RemoraModel,
    records: list[DonorResponseRecord],
    tokenizer: ByteTokenizer,
    device: torch.device,
) -> dict:
    model.eval()
    predictions = []
    correct = 0
    for record in records:
        input_ids = torch.tensor(tokenizer.encode(record.prompt), dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(input_ids)
        prediction = tokenizer.decode([int(logits[0, -1].argmax())])
        passed = _verify_lookup(record.prompt, prediction)
        correct += int(passed)
        predictions.append({"record_id": record.record_id, "prediction": prediction, "verified": passed})
    return {
        "count": len(records),
        "verified_count": correct,
        "accuracy": correct / max(len(records), 1),
        "verifier": "external-lookup-verifier-v1",
        "predictions": predictions,
    }


def _language_batch(stream: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return next(contiguous_batches(stream, batch_size=32, seq_len=96, device=device))


def _response_loss(model: RemoraModel, batch) -> torch.Tensor:
    input_ids, targets, response_mask = batch
    logits, _ = model(input_ids)
    token_loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view_as(response_mask)
    return (token_loss * response_mask).sum() / response_mask.sum().clamp_min(1.0)


def _language_loss(model: RemoraModel, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    _, loss = model(batch[0], batch[1])
    return loss


def _train_adapter(
    model: RemoraModel,
    response_batch,
    old_batch: tuple[torch.Tensor, torch.Tensor] | None,
    prefixes: list[str],
    steps: int,
    lr: float,
    retention_weight: float,
) -> list[float]:
    freeze_all(model)
    selected = unfreeze_prefixes(model, prefixes)
    if not selected:
        raise ValueError("lookup adapter prefixes matched no parameters")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    history = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _response_loss(model, response_batch)
        if old_batch is not None and retention_weight:
            loss = loss + retention_weight * _language_loss(model, old_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def _train_full(
    model: RemoraModel,
    response_batch,
    old_batch: tuple[torch.Tensor, torch.Tensor],
    steps: int,
    lr: float,
    retention_weight: float,
) -> list[float]:
    for parameter in model.parameters():
        parameter.requires_grad = True
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    history = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _response_loss(model, response_batch) + retention_weight * _language_loss(model, old_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        history.append(float(loss.detach()))
    return history


def _arm(
    checkpoint: str | Path,
    device: torch.device,
    train_batch,
    old_batch,
    old_stream,
    test_records,
    shifted_records,
    tokenizer,
    *,
    mode: str,
    steps: int,
    adapter_lr: float,
    full_lr: float,
    retention_weight: float,
) -> dict:
    model = _load_checkpoint(checkpoint, device)
    before = parameter_snapshot(model)
    if mode == "frozen":
        history = []
    elif mode == "adapter_target_only":
        history = _train_adapter(
            model,
            train_batch,
            None,
            [f"blocks.{index}.plastic" for index in range(len(model.blocks))],
            steps,
            adapter_lr,
            0.0,
        )
    elif mode == "adapter_rehearsal":
        history = _train_adapter(
            model,
            train_batch,
            old_batch,
            [f"blocks.{index}.plastic" for index in range(len(model.blocks))],
            steps,
            adapter_lr,
            retention_weight,
        )
    elif mode == "full_rehearsal":
        history = _train_full(model, train_batch, old_batch, steps, full_lr, retention_weight)
    else:
        raise ValueError(mode)
    return {
        "mode": mode,
        "train_loss_first_last": [history[0], history[-1]] if history else None,
        "old": evaluate_stream(model, old_stream, 32, 96, device),
        "same_interface": _evaluate_lookup(model, test_records, tokenizer, device),
        "shifted_interface": _evaluate_lookup(model, shifted_records, tokenizer, device),
        "update": changed_parameter_stats(before, model),
        "trainable_parameters": count_parameters(model, trainable_only=True),
    }


def run(
    checkpoint: str | Path,
    seed: int = 207,
    device_name: str = "auto",
    steps: int = 500,
    adapter_lr: float = 0.003,
    full_lr: float = 0.003,
    retention_weight: float = 1.0,
    output: str | Path | None = None,
    records_output: str | Path | None = None,
) -> dict:
    set_seed(seed)
    device = choose_device(device_name)
    train_records = _make_records(seed, 192, template="donor-route:key={key}=", prefix="lookup-train")
    test_records = _make_records(seed + 1, 64, template="donor-route:key={key}=", prefix="lookup-test")
    shifted_records = _make_records(
        seed + 2,
        64,
        template="donor-route-v2:context=stable;key={key}=>",
        prefix="lookup-shifted",
    )
    all_records = train_records + test_records + shifted_records
    if records_output:
        write_response_records(records_output, all_records)
    accepted_records = load_response_records(
        records_output,
        accepted_only=True,
        verifier=lambda item: item.verifier.get("passed") is True and _verify_lookup(item.prompt, item.response),
    ) if records_output else all_records
    if len(accepted_records) != len(all_records):
        raise ValueError("lookup response acceptance filter rejected a generated record")

    tokenizer = ByteTokenizer()
    train_batch = _encode_records(train_records, tokenizer, device)
    old_stream = encode_stream(build_heldout_corpus(400, seed=999))
    old_batch = _language_batch(old_stream, device)
    arms = {}
    for mode in ("frozen", "adapter_target_only", "adapter_rehearsal", "full_rehearsal"):
        arms[mode] = _arm(
            checkpoint,
            device,
            train_batch,
            old_batch,
            old_stream,
            test_records,
            shifted_records,
            tokenizer,
            mode=mode,
            steps=steps,
            adapter_lr=adapter_lr,
            full_lr=full_lr,
            retention_weight=retention_weight,
        )

    frozen_loss = arms["frozen"]["old"]["loss"]
    candidate = arms["adapter_rehearsal"]
    target_improved = candidate["same_interface"]["accuracy"] > arms["frozen"]["same_interface"]["accuracy"]
    retention_ok = candidate["old"]["loss"] <= max(frozen_loss * 2.0, frozen_loss + 0.5)
    if target_improved and retention_ok:
        interpretation = "MEASURED CONDITIONAL PASS: verified lookup responses entered plastic islands with old-stream rehearsal and a bounded update; shifted-template transfer is reported separately."
    else:
        interpretation = "MEASURED FAILURE: the adapter-rehearsal arm did not clear the predeclared response-improvement and retention gates; this is retained as a negative donor-interface result."

    result = {
        "schema": "remora-v0-donor-response-lookup-result",
        "seed": seed,
        "device": str(device),
        "mode": "MECHANISM_ONLY_SYNTHETIC_BLACKBOX_LOOKUP_DONOR",
        "source_checkpoint": str(checkpoint),
        "response_contract": "donor-response-v1",
        "records": {
            "train": len(train_records),
            "same_interface_test": len(test_records),
            "shifted_interface_test": len(shifted_records),
            "accepted": len(accepted_records),
            "records_path": str(records_output) if records_output else None,
            "donor_id": train_records[0].donor_id,
            "verifier": "external-lookup-verifier-v1",
            "train_lineage_key": train_records[0].lineage_key,
        },
        "controls": arms,
        "training": {
            "steps": steps,
            "adapter_lr": adapter_lr,
            "full_lr": full_lr,
            "retention_weight": retention_weight,
        },
        "promotion": {
            "state": "CANDIDATE",
            "promoted": False,
            "decision_authority": "external held-out verifier",
        },
        "interpretation": interpretation,
    }
    if output:
        write_json(output, result)
    record_experiment(
        ROOT,
        "DONOR-RESPONSE-002",
        "A verified black-box lookup response should enter a replaceable Remora plastic island without broad retraining, while old capability is protected by rehearsal.",
        "Train adapter-only, adapter-only-with-old-stream-rehearsal, and full-model-with-rehearsal controls on hashed donor-response-v1 records; evaluate same and shifted prompt interfaces with an external verifier.",
        "The rehearsal adapter beats the frozen response control, retains old loss within 2x or +0.5, and remains an unpromoted candidate; shifted-template accuracy is measured as transfer, not hidden.",
        "The rehearsal adapter does not improve verified response accuracy, exceeds the retention gate, updates nearly all parameters, accepts invalid records, or self-promotes.",
        f"python -m experiments.donor_response_lookup --checkpoint {checkpoint} --steps {steps}",
        seed,
        {
            "device": str(device),
            "mode": result["mode"],
            "controls": arms,
            "records": result["records"],
        },
        interpretation,
        "If the lookup arm passes, repeat with a separately launched local donor runtime and a verifier-held-out task; if it fails, redesign the port or plastic-island interface before querying Qwen.",
        hardware={"device": str(device), "mode": result["mode"]},
    )
    if not (target_improved and retention_ok):
        record_failure(
            ROOT,
            "DONOR-RESPONSE-002",
            _load_checkpoint(checkpoint, torch.device("cpu")).cfg.to_dict(),
            {"task": "synthetic_lookup", "train": len(train_records), "test": len(test_records), "shifted_test": len(shifted_records)},
            seed,
            f"adapter rehearsal response accuracy {candidate['same_interface']['accuracy']:.6f} versus frozen {arms['frozen']['same_interface']['accuracy']:.6f}; old loss {candidate['old']['loss']:.6f} versus frozen {frozen_loss:.6f}",
            "The current plastic output path may require a learned query/response head or a better-conditioned common-bus port; the first arithmetic trial also failed under sparse OOD supervision.",
            "Retest after adding a response-specific port or after pretraining a small compatible donor on the same protocol; keep the candidate unpromoted.",
            runtime={"device": str(device), "mode": result["mode"]},
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "remora-v0-scratch-seed7.pt"))
    parser.add_argument("--seed", type=int, default=207)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--adapter-lr", type=float, default=0.003)
    parser.add_argument("--full-lr", type=float, default=0.003)
    parser.add_argument("--retention-weight", type=float, default=1.0)
    parser.add_argument("--output", default=str(ROOT / "results" / "donor-response-lookup.json"))
    parser.add_argument("--records-output", default=str(ROOT / "results" / "donor-response-lookup-records.jsonl"))
    args = parser.parse_args()
    result = run(
        args.checkpoint,
        args.seed,
        args.device,
        args.steps,
        args.adapter_lr,
        args.full_lr,
        args.retention_weight,
        args.output,
        args.records_output,
    )
    print(json.dumps({
        "interpretation": result["interpretation"],
        "frozen": result["controls"]["frozen"]["same_interface"]["accuracy"],
        "adapter_target_only": result["controls"]["adapter_target_only"]["same_interface"]["accuracy"],
        "adapter_rehearsal": result["controls"]["adapter_rehearsal"]["same_interface"]["accuracy"],
        "adapter_rehearsal_shifted": result["controls"]["adapter_rehearsal"]["shifted_interface"]["accuracy"],
        "full_rehearsal": result["controls"]["full_rehearsal"]["same_interface"]["accuracy"],
        "adapter_changed_fraction": result["controls"]["adapter_rehearsal"]["update"]["changed_fraction"],
        "full_changed_fraction": result["controls"]["full_rehearsal"]["update"]["changed_fraction"],
        "adapter_old_loss": result["controls"]["adapter_rehearsal"]["old"]["loss"],
        "frozen_old_loss": result["controls"]["frozen"]["old"]["loss"],
    }, indent=2))


if __name__ == "__main__":
    main()
