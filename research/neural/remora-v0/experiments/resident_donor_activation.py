from __future__ import annotations

"""Capture one declared hidden-state slice from an explicitly loaded donor."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.donors.activation import load_activation_bundle, write_activation_bundle, write_activation_records
from remora.donors.port import TeacherPortAdapter
from remora.donors.runtime import DonorRuntimeSpec, LocalTransformersDonor
from remora.ledger import record_experiment
from remora.utils import choose_device, runtime_context, write_json


_PROMPTS = (
    "Explain why a checksum is useful in one short sentence.",
    "Give one concise example of a reversible experiment.",
    "State one reason to keep provenance separate from a model weight.",
    "What does a held-out evaluation protect against?",
)


def run(
    model_path: str,
    *,
    runtime_id: str,
    layer_name: str,
    device_name: str = "auto",
    trust_remote_code: bool = False,
    prompt_format: str = "raw",
    output: str | Path | None = None,
    records_output: str | Path | None = None,
    bundle_output: str | Path | None = None,
) -> dict:
    device = choose_device(device_name)
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
        _PROMPTS,
        donor_id=f"resident-{Path(model_path).name}",
        layer_name=layer_name,
        lineage_key=f"{runtime_id}:activation-probe-v1:{prompt_format}",
    )
    if records_output:
        write_activation_records(records_output, records)
    if bundle_output:
        payload_bytes = write_activation_bundle(bundle_output, records, activations)
        loaded = load_activation_bundle(
            bundle_output,
            records_output,
            max_payload_bytes=256 * 1024,
            accepted_only=True,
        )
    else:
        payload_bytes = sum(tensor.numel() * tensor.element_size() for tensor in activations.values())
        loaded = {"records": records, "activations": activations, "payload_bytes": payload_bytes}
    port = TeacherPortAdapter(
        teacher_dim=next(iter(loaded["activations"].values())).numel(),
        bus_dim=96,
        bottleneck=96,
    )
    with torch.no_grad():
        port_output = port(
            torch.stack(list(loaded["activations"].values()), dim=0).unsqueeze(1)
        )
    result = {
        "schema": "remora-v0-resident-donor-activation-result",
        "mode": "EXPLICIT_LOCAL_DONOR_ACTIVATION_CAPTURE",
        "model_path": str(Path(model_path).expanduser().resolve()),
        "runtime_id": runtime_id,
        "layer_name": layer_name,
        "prompt_format": prompt_format,
        "chat_template_sha256": donor.chat_template_sha256,
        "config_repairs": list(donor.config_repairs),
        "device": str(device),
        "record_count": len(records),
        "loaded_record_count": len(loaded["records"]),
        "activation_shape": records[0].shape,
        "activation_dtype": records[0].dtype,
        "payload_bytes_written": payload_bytes,
        "payload_bytes_loaded": loaded["payload_bytes"],
        "port_output_shape": list(port_output.latent.shape),
        "port_output_finite": bool(torch.isfinite(port_output.latent).all()),
        "records_path": str(records_output) if records_output else None,
        "bundle_path": str(bundle_output) if bundle_output else None,
        "promotion": {
            "state": "CANDIDATE_ACTIVATION_CAPTURED",
            "promoted": False,
            "decision_authority": "external Remora experiment harness",
        },
        "interpretation": "MEASURED RUNTIME PROBE: one explicitly selected donor layer emitted final-token activations through a bounded hook, the records round-tripped through donor-activation-v1, and the port produced a finite language-bus packet; no donor module was grafted or promoted.",
        "runtime": runtime_context(device),
    }
    if output:
        write_json(output, result)
    record_experiment(
        ROOT,
        "DONOR-RUNTIME-ACTIVATION-001",
        "A resident donor can expose one declared hidden-state slice that Remora can port without importing the donor graph or reading unrelated layers.",
        "Explicitly load a local donor, hook one named layer, retain only final-token activations, enforce a 256 KiB bundle budget, and pass the result through TeacherPortAdapter into the 96-wide bus.",
        "The hook returns bounded finite activations, hashes and layer identity are recorded, port dimensions are valid, and promotion remains false.",
        "An undeclared layer is captured, the bundle budget or hashes are bypassed, the port emits non-finite values, or the candidate self-promotes.",
        "python -m experiments.resident_donor_activation --model-path <local-model> --runtime-id <id> --layer-name <module> --allow-model-load",
        0,
        {
            "model_path": result["model_path"],
            "runtime_id": runtime_id,
            "layer_name": layer_name,
            "record_count": len(records),
            "loaded_record_count": len(loaded["records"]),
            "activation_shape": records[0].shape,
            "payload_bytes_loaded": loaded["payload_bytes"],
            "port_output_shape": result["port_output_shape"],
        },
        result["interpretation"],
        "Repeat with a Qwen-compatible runtime hook only after its serving path is explicitly budgeted; train a port on verifier-held-out tasks before considering any donor-derived module.",
        hardware=runtime_context(device),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--layer-name", default="model.layers.0")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--prompt-format", choices=["raw", "chat_template"], default="raw")
    parser.add_argument("--chat-template", action="store_const", const="chat_template", dest="prompt_format")
    parser.add_argument("--allow-model-load", action="store_true", help="required safety acknowledgment")
    parser.add_argument("--output", default=str(ROOT / "results" / "resident-donor-activation.json"))
    parser.add_argument("--records-output", default=str(ROOT / "results" / "resident-donor-activation-records.jsonl"))
    parser.add_argument("--bundle-output", default=str(ROOT / "results" / "resident-donor-activation.safetensors"))
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
        output=args.output,
        records_output=args.records_output,
        bundle_output=args.bundle_output,
    )
    print(json.dumps({
        "interpretation": result["interpretation"],
        "model_path": result["model_path"],
        "layer_name": result["layer_name"],
        "activation_shape": result["activation_shape"],
        "loaded_record_count": result["loaded_record_count"],
        "payload_bytes_loaded": result["payload_bytes_loaded"],
        "port_output_finite": result["port_output_finite"],
        "promotion": result["promotion"],
    }, indent=2))


if __name__ == "__main__":
    main()
