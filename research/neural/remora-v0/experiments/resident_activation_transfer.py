from __future__ import annotations

"""Test whether a resident donor hidden slice carries useful transferable signal.

The donor is loaded only after the explicit CLI acknowledgment. Labels come
from the external parity oracle, never from donor text. The donor is frozen;
only a ``donor-port-v1`` adapter and a small candidate head are trained. A
prompt-byte control and a shuffled-feature control make the result falsifiable
and prevent a finite activation capture from being called useful by default.
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

from environments.transfer_suite import TaskExample, build_parity_examples
from remora.donors.activation import load_activation_bundle, write_activation_bundle, write_activation_records
from remora.donors.port import TeacherPortAdapter
from remora.donors.runtime import DonorRuntimeSpec, LocalTransformersDonor
from remora.ledger import record_experiment, record_failure
from remora.utils import changed_parameter_stats, choose_device, count_parameters, parameter_snapshot, set_seed, write_json


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean())


def _prompt_histogram(examples: list[TaskExample], tokenizer) -> torch.Tensor:
    rows = []
    for example in examples:
        ids = torch.tensor(tokenizer.encode(example.prompt), dtype=torch.long)
        rows.append(torch.bincount(ids, minlength=tokenizer.vocab_size).float() / max(ids.numel(), 1))
    return torch.stack(rows)


def _fit_candidate(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_count: int,
    valid_count: int,
    shifted_count: int,
    *,
    device: torch.device,
    steps: int,
    seed: int,
) -> dict:
    set_seed(seed)
    port = TeacherPortAdapter(
        teacher_dim=features.size(-1),
        bus_dim=96,
        bottleneck=96,
    ).to(device)
    head = nn.Linear(96, 2).to(device)
    candidate = nn.ModuleDict({"port": port, "head": head})
    before = parameter_snapshot(candidate)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=0.03, weight_decay=0.0)
    history = []
    train_features = features[:train_count].to(device)
    train_labels = labels[:train_count].to(device)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        packet = port(train_features.unsqueeze(1), producer="resident-donor")
        logits = head(packet.latent[:, 0])
        loss = nn.functional.cross_entropy(logits, train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
        optimizer.step()
        history.append(float(loss.detach()))
    with torch.no_grad():
        all_features = features.to(device)
        logits = head(port(all_features.unsqueeze(1), producer="resident-donor").latent[:, 0])
    offsets = (train_count, train_count + valid_count, train_count + valid_count + shifted_count)
    split_logits = {
        "train": logits[: offsets[0]],
        "valid": logits[offsets[0] : offsets[1]],
        "shifted": logits[offsets[1] : offsets[2]],
    }
    split_labels = {
        "train": labels[: offsets[0]].to(device),
        "valid": labels[offsets[0] : offsets[1]].to(device),
        "shifted": labels[offsets[1] : offsets[2]].to(device),
    }
    return {
        "loss_first_last": [history[0], history[-1]],
        "accuracy": {split: _accuracy(split_logits[split], split_labels[split]) for split in split_logits},
        "parameters": count_parameters(candidate),
        "changed_parameters": changed_parameter_stats(before, candidate),
    }


def _fit_classifier(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    train_count: int,
    valid_count: int,
    shifted_count: int,
    *,
    device: torch.device,
    steps: int,
    seed: int,
) -> dict:
    set_seed(seed)
    classifier = nn.Sequential(
        nn.Linear(inputs.size(-1), 48),
        nn.Tanh(),
        nn.Linear(48, 2),
    ).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.03, weight_decay=0.0)
    train_inputs = inputs[:train_count].to(device)
    train_labels = labels[:train_count].to(device)
    history = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(train_inputs)
        loss = nn.functional.cross_entropy(logits, train_labels)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    with torch.no_grad():
        logits = classifier(inputs.to(device))
    first = train_count
    second = train_count + valid_count
    split_logits = {
        "train": logits[:first],
        "valid": logits[first:second],
        "shifted": logits[second : second + shifted_count],
    }
    split_labels = {
        "train": labels[:first],
        "valid": labels[first:second],
        "shifted": labels[second : second + shifted_count],
    }
    return {
        "loss_first_last": [history[0], history[-1]],
        "accuracy": {split: _accuracy(split_logits[split], split_labels[split].to(device)) for split in split_logits},
        "parameters": count_parameters(classifier),
    }


def _split_labels(examples: list[TaskExample]) -> torch.Tensor:
    return torch.tensor([int(example.response) for example in examples], dtype=torch.long)


def run(
    model_path: str,
    *,
    runtime_id: str,
    layer_name: str = "model.layers.0",
    device_name: str = "auto",
    trust_remote_code: bool = False,
    prompt_format: str = "chat_template",
    train_count: int = 32,
    valid_count: int = 32,
    shifted_count: int = 32,
    train_steps: int = 240,
    seed: int = 931,
    output: str | Path | None = None,
    records_output: str | Path | None = None,
    bundle_output: str | Path | None = None,
) -> dict:
    if min(train_count, valid_count, shifted_count, train_steps) <= 0:
        raise ValueError("activation-transfer counts and train_steps must be positive")
    device = choose_device(device_name)
    examples = (
        build_parity_examples(train_count, 931, low=0, high=16)
        + build_parity_examples(valid_count, 932, low=16, high=24)
        + build_parity_examples(shifted_count, 933, low=24, high=32, shifted=True)
    )
    donor = LocalTransformersDonor(
        DonorRuntimeSpec(
            model_path=model_path,
            runtime_id=runtime_id,
            device=str(device),
            max_new_tokens=1,
            trust_remote_code=trust_remote_code,
            prompt_format=prompt_format,
            allow_model_load=True,
        )
    )
    records, activations = donor.activation_records(
        [example.prompt for example in examples],
        donor_id=f"resident-{Path(model_path).name}",
        layer_name=layer_name,
        lineage_key=f"{runtime_id}:activation-transfer-parity-v1",
    )
    if records_output:
        write_activation_records(records_output, records)
    if bundle_output:
        payload_bytes = write_activation_bundle(bundle_output, records, activations)
        loaded = load_activation_bundle(
            bundle_output,
            records_output,
            max_payload_bytes=64 * 1024 * 1024,
            accepted_only=True,
        )
    else:
        payload_bytes = sum(value.numel() * value.element_size() for value in activations.values())
        loaded = {"records": records, "activations": activations, "payload_bytes": payload_bytes}
    features = torch.stack([loaded["activations"][record.record_id] for record in records])
    labels = _split_labels(examples)
    del donor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    candidate = _fit_candidate(
        features,
        labels,
        train_count,
        valid_count,
        shifted_count,
        device=device,
        steps=train_steps,
        seed=seed,
    )
    from remora.tokenizer import ByteTokenizer

    tokenizer = ByteTokenizer()
    prompt_inputs = _prompt_histogram(examples, tokenizer)
    prompt_control = _fit_classifier(
        prompt_inputs,
        labels,
        train_count,
        valid_count,
        shifted_count,
        device=device,
        steps=train_steps,
        seed=seed + 1,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 2)
    shuffled = features[torch.randperm(features.size(0), generator=generator)]
    shuffled_control = _fit_candidate(
        shuffled,
        labels,
        train_count,
        valid_count,
        shifted_count,
        device=device,
        steps=train_steps,
        seed=seed + 2,
    )
    utility = candidate["accuracy"]["valid"] - prompt_control["accuracy"]["valid"]
    shifted_utility = candidate["accuracy"]["shifted"] - prompt_control["accuracy"]["shifted"]
    useful = utility >= 0.05 and shifted_utility >= 0.0
    result = {
        "schema": "remora-v0-resident-activation-transfer-result",
        "mode": "EXPLICIT_LOCAL_DONOR_ACTIVATION_TRANSFER_WITH_EXTERNAL_PARITY_ORACLE",
        "model_path": str(Path(model_path).expanduser().resolve()),
        "runtime_id": runtime_id,
        "layer_name": layer_name,
        "prompt_format": prompt_format,
        "chat_template_sha256": records[0].runtime.get("chat_template_sha256"),
        "config_repairs": records[0].runtime.get("config_repairs", []),
        "device": str(device),
        "seed": seed,
        "split_policy": {
            "train": "math-parity-v1 pairs [0,16)",
            "valid": "math-parity-v1 pairs [16,24)",
            "shifted": "calculate[parity] v2 pairs [24,32)",
            "label_source": "external parity oracle; donor responses never used as labels",
        },
        "records": len(records),
        "loaded_record_count": len(loaded["records"]),
        "activation_shape": records[0].shape,
        "activation_dtype": records[0].dtype,
        "payload_bytes_loaded": loaded["payload_bytes"],
        "payload_bytes_written": payload_bytes,
        "records_path": str(records_output) if records_output else None,
        "bundle_path": str(bundle_output) if bundle_output else None,
        "candidate": candidate,
        "prompt_byte_control": prompt_control,
        "shuffled_activation_control": shuffled_control,
        "derived": {
            "valid_utility_over_prompt_control": utility,
            "shifted_utility_over_prompt_control": shifted_utility,
            "useful_gate": "valid utility >= 0.05 and shifted utility >= 0.0",
        },
        "promotion": {
            "state": "CANDIDATE_USEFUL" if useful else "CANDIDATE_REJECTED_UNDER_THIS_TASK",
            "promoted": False,
            "decision_authority": "external held-out evaluator",
        },
        "interpretation": (
            "MEASURED CONDITIONAL PASS: the selected resident donor layer beat the prompt-only control on the held-out parity split and did not lose on the shifted interface; the port remains an unpromoted candidate."
            if useful
            else
            "MEASURED FAILURE OR INCONCLUSIVE: the selected resident donor layer did not clear the predeclared held-out utility gate; no donor-derived module was promoted."
        ),
    }
    if output:
        write_json(output, result)
    record_experiment(
        ROOT,
        "DONOR-ACTIVATION-TRANSFER-001",
        "A frozen resident donor hidden slice can carry transferable signal into a small Remora-compatible port, beyond prompt-only features, without importing donor weights into the candidate.",
        f"Capture {layer_name} through prompt_format={prompt_format} on externally labeled parity prompts, train only TeacherPortAdapter plus a classifier, and compare with prompt-byte and shuffled-activation controls.",
        "The activation port beats the prompt-only control by at least 5 percentage points on disjoint validation pairs and is no worse on the shifted interface; all donor weights remain frozen and promotion remains false.",
        "The port fails to beat the prompt control, loses on the shifted interface, activation hashes/budget checks fail, labels come from donor text, or the candidate self-promotes.",
        "python -m experiments.resident_activation_transfer --model-path <local-model> --runtime-id <id> --layer-name <module> --allow-model-load",
        seed,
        {key: value for key, value in result.items() if key not in {"promotion"}},
        result["interpretation"],
        "If useful, train a Remora specialist on a larger held-out task and test replacement/rehearsal; if not, try a different declared layer or keep the donor response-only.",
        hardware={"device": str(device), "mode": result["mode"], "runtime_id": runtime_id},
    )
    if not useful:
        record_failure(
            ROOT,
            "DONOR-ACTIVATION-TRANSFER-001",
            {"model_path": result["model_path"], "runtime_id": runtime_id, "layer_name": layer_name},
            {"train_count": train_count, "valid_count": valid_count, "shifted_count": shifted_count, "prompt_format": prompt_format},
            seed,
            "resident donor activation port did not clear the held-out utility gate",
            "The selected layer may not encode parity in a transferable way, or the one-layer final-token port is underpowered; this does not establish that the donor is globally useless.",
            "Retest after changing layer, pooling, prompt protocol, or target task; preserve the activation records and compare against a response verifier before promotion.",
            runtime={"device": str(device), "runtime_id": runtime_id, "prompt_format": prompt_format},
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--layer-name", default="model.layers.0")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--prompt-format", choices=["raw", "chat_template"], default="chat_template")
    parser.add_argument("--chat-template", action="store_const", const="chat_template", dest="prompt_format")
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--valid-count", type=int, default=32)
    parser.add_argument("--shifted-count", type=int, default=32)
    parser.add_argument("--train-steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=931)
    parser.add_argument("--allow-model-load", action="store_true", help="required safety acknowledgment")
    parser.add_argument("--output", default=str(ROOT / "results" / "resident-activation-transfer.json"))
    parser.add_argument("--records-output", default=str(ROOT / "results" / "resident-activation-transfer-records.jsonl"))
    parser.add_argument("--bundle-output", default=str(ROOT / "results" / "resident-activation-transfer.safetensors"))
    args = parser.parse_args()
    if not args.allow_model_load:
        raise SystemExit("refusing resident model load: pass --allow-model-load explicitly")
    result = run(
        args.model_path,
        runtime_id=args.runtime_id,
        layer_name=args.layer_name,
        device_name=args.device,
        trust_remote_code=args.trust_remote_code,
        prompt_format=args.prompt_format,
        train_count=args.train_count,
        valid_count=args.valid_count,
        shifted_count=args.shifted_count,
        train_steps=args.train_steps,
        seed=args.seed,
        output=args.output,
        records_output=args.records_output,
        bundle_output=args.bundle_output,
    )
    print(json.dumps({
        "interpretation": result["interpretation"],
        "candidate_accuracy": result["candidate"]["accuracy"],
        "prompt_byte_control_accuracy": result["prompt_byte_control"]["accuracy"],
        "shuffled_activation_control_accuracy": result["shuffled_activation_control"]["accuracy"],
        "derived": result["derived"],
        "promotion": result["promotion"],
    }, indent=2))


if __name__ == "__main__":
    main()
