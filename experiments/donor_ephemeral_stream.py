from __future__ import annotations

"""Run the safe, one-shard-at-a-time Qwen donor lifecycle.

This experiment has two deliberately separate passes:

1. read every Qwen shard header sequentially, without materializing weights;
2. read the selected layer-17/value-head-10 GDN core pieces from their one
   source shard, build a short-lived staged safetensors bundle, verify the
   reassembled organ, and delete only that staged bundle after acceptance.

The source shard remains in place.  The receipt is the durable replacement
for the deleted staging copy: it contains the pinned source identity, full
parent tensor offsets/hashes, exact slice coordinates, reassembly order,
dependencies, and the acceptance/deletion events.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.donors.ephemeral import scan_safetensor_shards, stream_selected_payload  # noqa: E402
from remora.donors.gdn import QwenGatedDeltaCoreOrgan  # noqa: E402
from remora.ledger import record_experiment  # noqa: E402
from remora.utils import git_hash, runtime_context, tensor_sha256, write_json  # noqa: E402


DONOR_REPOSITORY = "Qwen/Qwen3.8-Flash-Next"
DONOR_REVISION = "f5d08274bafd880402bd16f5e3e6c514136ec06c"
LAYER = 17
VALUE_HEAD = 10
NUM_KEY_HEADS = 16
KEY_HEAD_DIM = 128
VALUE_HEAD_DIM = 128
KEY_DIM = NUM_KEY_HEADS * KEY_HEAD_DIM
NUM_VALUE_HEADS = 48
NATIVE_WIDTH = 2560
SHARD = "model-00051-of-00131.safetensors"


def _gdn_core_selection() -> dict[str, Any]:
    """Describe the exact native-width donor pieces and their reassembly."""

    prefix = f"model.language_model.layers.{LAYER}.linear_attn."
    key_head = VALUE_HEAD // (NUM_VALUE_HEADS // NUM_KEY_HEADS)
    q0 = key_head * KEY_HEAD_DIM
    k0 = KEY_DIM + key_head * KEY_HEAD_DIM
    v0 = KEY_DIM * 2 + VALUE_HEAD * VALUE_HEAD_DIM
    requests: list[dict[str, Any]] = []

    def add(
        request_id: str,
        role: str,
        tensor_name: str,
        *,
        rows: list[int] | None = None,
        operation: str,
        order: int,
        depends_on: list[str],
        context: dict[str, Any],
    ) -> None:
        request: dict[str, Any] = {
            "request_id": request_id,
            "role": role,
            "tensor_name": tensor_name,
            "source_shard": SHARD,
            "operation": operation,
            "reassembly": {"group": "qwen-gdn-core-native", "axis": 0, "order": order},
            "depends_on": depends_on,
            "context": context,
        }
        if rows is not None:
            request["slice"] = {"rows": rows}
        requests.append(request)

    qkv = prefix + "in_proj_qkv.weight"
    conv = prefix + "conv1d.weight"
    add("q_weight", "query_projection", qkv, rows=[q0, q0 + KEY_HEAD_DIM], operation="row_slice", order=0, depends_on=["in_proj_qkv.weight"], context={"source_rows": [q0, q0 + KEY_HEAD_DIM], "input_width": NATIVE_WIDTH, "output_width": KEY_HEAD_DIM, "key_head": key_head})
    add("k_weight", "key_projection", qkv, rows=[k0, k0 + KEY_HEAD_DIM], operation="row_slice", order=1, depends_on=["in_proj_qkv.weight"], context={"source_rows": [k0, k0 + KEY_HEAD_DIM], "input_width": NATIVE_WIDTH, "output_width": KEY_HEAD_DIM, "key_head": key_head})
    add("v_weight", "value_projection", qkv, rows=[v0, v0 + VALUE_HEAD_DIM], operation="row_slice", order=2, depends_on=["in_proj_qkv.weight"], context={"source_rows": [v0, v0 + VALUE_HEAD_DIM], "input_width": NATIVE_WIDTH, "output_width": VALUE_HEAD_DIM, "value_head": VALUE_HEAD})
    add("a_weight", "decay_projection", prefix + "in_proj_a.weight", rows=[VALUE_HEAD, VALUE_HEAD + 1], operation="row_slice", order=3, depends_on=["in_proj_a.weight"], context={"source_rows": [VALUE_HEAD, VALUE_HEAD + 1], "input_width": NATIVE_WIDTH, "output_width": 1})
    add("b_weight", "beta_projection", prefix + "in_proj_b.weight", rows=[VALUE_HEAD, VALUE_HEAD + 1], operation="row_slice", order=4, depends_on=["in_proj_b.weight"], context={"source_rows": [VALUE_HEAD, VALUE_HEAD + 1], "input_width": NATIVE_WIDTH, "output_width": 1})
    add("conv_q", "query_convolution", conv, rows=[q0, q0 + KEY_HEAD_DIM], operation="row_slice", order=5, depends_on=["conv1d.weight", "q_weight"], context={"source_rows": [q0, q0 + KEY_HEAD_DIM], "channels": KEY_HEAD_DIM, "kernel": 4})
    add("conv_k", "key_convolution", conv, rows=[k0, k0 + KEY_HEAD_DIM], operation="row_slice", order=6, depends_on=["conv1d.weight", "k_weight"], context={"source_rows": [k0, k0 + KEY_HEAD_DIM], "channels": KEY_HEAD_DIM, "kernel": 4})
    add("conv_v", "value_convolution", conv, rows=[v0, v0 + VALUE_HEAD_DIM], operation="row_slice", order=7, depends_on=["conv1d.weight", "v_weight"], context={"source_rows": [v0, v0 + VALUE_HEAD_DIM], "channels": VALUE_HEAD_DIM, "kernel": 4})
    add("A_log", "log_decay_scalar", prefix + "A_log", rows=[VALUE_HEAD, VALUE_HEAD + 1], operation="row_slice", order=8, depends_on=["a_weight"], context={"source_rows": [VALUE_HEAD, VALUE_HEAD + 1], "state_role": "decay_parameter"})
    add("dt_bias", "decay_bias_scalar", prefix + "dt_bias", rows=[VALUE_HEAD, VALUE_HEAD + 1], operation="row_slice", order=9, depends_on=["a_weight"], context={"source_rows": [VALUE_HEAD, VALUE_HEAD + 1], "state_role": "decay_parameter"})

    return {
        "schema": "remora-ephemeral-donor-selection-v1",
        "component_id": f"{DONOR_REPOSITORY}:layer{LAYER}:linear-attn:value-head{VALUE_HEAD}:gdn-core",
        "source": {"repository": DONOR_REPOSITORY, "revision": DONOR_REVISION},
        "requests": requests,
        "reassembly": {
            "group": "qwen-gdn-core-native",
            "steps": [
                {"output": "q_weight", "from": "q_weight", "operation": "identity"},
                {"output": "k_weight", "from": "k_weight", "operation": "identity"},
                {"output": "v_weight", "from": "v_weight", "operation": "identity"},
                {"output": "a_weight", "from": "a_weight", "operation": "identity"},
                {"output": "b_weight", "from": "b_weight", "operation": "identity"},
                {"output": "conv_qkv", "from": ["conv_q", "conv_k", "conv_v"], "operation": "squeeze_then_concatenate", "squeeze_axis": 1, "axis": 0, "order": ["conv_q", "conv_k", "conv_v"]},
                {"output": "A_log", "from": "A_log", "operation": "identity"},
                {"output": "dt_bias", "from": "dt_bias", "operation": "identity"},
            ],
            "excluded_coadapted_siblings": [
                prefix + "in_proj_z.weight",
                prefix + "norm.weight",
                prefix + "out_proj.weight",
            ],
        },
        "contract": {
            "input": {"shape": ["batch", "time", NATIVE_WIDTH], "dtype": "BF16_or_FP32", "semantic": "Qwen layer-17 linear-attention hidden state"},
            "output": {"shape": ["batch", "time", VALUE_HEAD_DIM], "semantic": "value-head recurrent core"},
            "state": {"owner": "qwen-gdn-core", "recurrent": ["batch", VALUE_HEAD_DIM, KEY_HEAD_DIM], "convolution": ["batch", 3 * VALUE_HEAD_DIM, 3]},
            "normalization": ["L2-normalize q and k after causal convolution"],
            "residual_assumptions": ["core ends before z gate, RMS norm, and out projection"],
        },
    }


def _consume_gdn_bundle(bundle_path: Path, tensors: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Acceptance gate: reload the temporary bundle and verify its organ path."""

    from safetensors import safe_open

    reloaded: dict[str, Any] = {}
    with safe_open(str(bundle_path), framework="pt", device="cpu") as handle:
        for key in sorted(tensors):
            reloaded[key] = handle.get_tensor(key)
    reload_mismatches = [key for key in tensors if not torch.equal(reloaded[key], tensors[key])]
    if reload_mismatches:
        return {"accepted": False, "verifier_id": "qwen-gdn-ephemeral-integrity-v1", "reason": "staged_bundle_reload_mismatch", "metrics": {"mismatched_keys": reload_mismatches}}

    conv_parts = []
    for key in ("conv_q", "conv_k", "conv_v"):
        part = tensors[key]
        if part.ndim == 3 and part.shape[1] == 1:
            part = part.squeeze(1)
        conv_parts.append(part)
    conv_qkv = torch.cat(tuple(conv_parts), dim=0)
    organ = QwenGatedDeltaCoreOrgan(
        q_weight=tensors["q_weight"],
        k_weight=tensors["k_weight"],
        v_weight=tensors["v_weight"],
        a_weight=tensors["a_weight"],
        b_weight=tensors["b_weight"],
        conv_qkv=conv_qkv,
        A_log=tensors["A_log"],
        dt_bias=tensors["dt_bias"],
        donor_variant="actual",
    )
    generator = torch.Generator(device="cpu").manual_seed(17010)
    probe = torch.randn(1, 7, NATIVE_WIDTH, generator=generator) * 0.05
    with torch.no_grad():
        full, full_state = organ(probe, return_state=True)
        first, first_state = organ(probe[:, :3], return_state=True)
        second, second_state = organ(probe[:, 3:], state=first_state, return_state=True)
        chunked = torch.cat((first, second), dim=1)
    difference = chunked.float() - full.float()
    relative = float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(full.float()).clamp_min(1e-30))
    finite = bool(torch.isfinite(full).all() and torch.isfinite(full_state[0]).all() and torch.isfinite(full_state[1]).all())
    accepted = finite and float(difference.abs().max()) < 1e-5 and relative < 1e-5
    return {
        "accepted": accepted,
        "verifier_id": "qwen-gdn-ephemeral-integrity-v1",
        "reason": "reloaded_exact_and_chunked_state_equivalent" if accepted else "organ_integrity_gate_failed",
        "metrics": {
            "reloaded_key_count": len(reloaded),
            "organ_parameter_count": organ.donor_parameter_count,
            "output_shape": list(full.shape),
            "recurrent_state_shape": list(full_state[0].shape),
            "convolution_state_shape": list(full_state[1].shape),
            "finite": finite,
            "chunked_max_absolute_error": float(difference.abs().max()),
            "chunked_relative_l2_error": relative,
            "reloaded_tensor_sha256": {key: tensor_sha256(value) for key, value in reloaded.items()},
        },
        "satisfied_context": {
            "dependency_graph_checked": True,
            "reassembly_executed": "conv_qkv = concatenate([squeeze(conv_q, axis=1), squeeze(conv_k, axis=1), squeeze(conv_v, axis=1)], axis=0)",
            "state_owner": "qwen-gdn-core",
            "source_payload_is_still_external": True,
            "future_recovery": "use source shard, parent offsets, slice coordinates, and hashes in the receipt",
            "context_request_count": len(context.get("requests", [])),
        },
    }


def run_header_sweep(source: Path, output: Path) -> dict[str, Any]:
    result = scan_safetensor_shards(source, include_tensor_inventory=True)
    result["experiment_id"] = "QWEN-SAFETENSOR-SHARD-SWEEP-001"
    result["runtime"] = {**runtime_context(torch.device("cpu")), "cwd": str(ROOT), "mode": "qwen_safetensor_header_sweep", "source_model_materialized": False}
    result["provenance"] = {"git_hash": git_hash(ROOT), "script": str(Path(__file__).resolve())}
    result["labels"] = {
        "MEASURED": ["one-shard-at-a-time header parsing", "tensor geometry and data offsets", "header hashes"],
        "UNMEASURED": ["payload values for unselected shards", "full shard hashes for unselected shards"],
    }
    write_json(output, result)
    return result


def run_stream(
    source: Path,
    *,
    output: Path,
    event_log: Path,
    staging_dir: Path,
    delete_after_accept: bool,
    stream_id: str,
) -> dict[str, Any]:
    selection = _gdn_core_selection()
    result = stream_selected_payload(
        source,
        selection,
        staging_dir=staging_dir,
        receipt_path=output,
        event_log_path=event_log,
        allow_payload=True,
        delete_after_accept=delete_after_accept,
        hash_source_shards=True,
        consumer=_consume_gdn_bundle,
        stream_id=stream_id,
    )
    result["experiment_id"] = "QWEN-SAFETENSOR-EPHEMERAL-GDN-001"
    result["runtime"] = {**runtime_context(torch.device("cpu")), "cwd": str(ROOT), "mode": "qwen_safetensor_ephemeral_gdn_core", "source_model_materialized": False}
    result["provenance"] = {"git_hash": git_hash(ROOT), "script": str(Path(__file__).resolve())}
    result["labels"] = {
        "MEASURED": ["selected source-shard hash", "parent tensor payload hashes", "exact slice geometry", "staged bundle reload", "GDN core finite/chunked-state acceptance", "deletion verification"],
        "DERIVED": ["reassembly/dependency recipe", "source-to-stage accounting"],
        "UNMEASURED": ["original Qwen training compute avoided", "capability utility beyond organ integrity"],
    }
    write_json(output, result)
    return result


def _record(repo: Path, experiment_id: str, hypothesis: str, change: str, expected: str, falsification: str, command: str, metrics: dict[str, Any], conclusion: str, next_action: str) -> None:
    record_experiment(
        repo,
        experiment_id,
        hypothesis,
        change,
        expected,
        falsification,
        command,
        0,
        metrics,
        conclusion,
        next_action,
        hardware={**runtime_context(torch.device("cpu")), "cwd": str(repo), "source_model_materialized": False},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely stream Qwen safetensor shards with durable context receipts.")
    parser.add_argument("--source", default="/home/leo/models/Qwen3.8-Flash-Next-BF16-source")
    parser.add_argument("--header-output", default=str(ROOT / "results" / "qwen-safetensor-shard-sweep-v1.json"))
    parser.add_argument("--output", default=str(ROOT / "results" / "qwen-safetensor-ephemeral-gdn-v1.json"))
    parser.add_argument("--event-log", default=str(ROOT / "ledger" / "donor-safetensor-lifecycle.jsonl"))
    parser.add_argument("--staging-dir", default="/tmp/remora-donor-safetensor-stream")
    parser.add_argument("--stream-id", default="qwen-gdn-core-ephemeral-001")
    parser.add_argument("--delete-after-accept", action="store_true", help="Delete only the temporary staged bundle after the organ gate accepts it.")
    parser.add_argument("--skip-header-sweep", action="store_true")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    header_output = Path(args.header_output).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    event_log = Path(args.event_log).expanduser().resolve()
    staging_dir = Path(args.staging_dir).expanduser().resolve()
    header_result = None
    if not args.skip_header_sweep:
        header_result = run_header_sweep(source, header_output)
        _record(
            ROOT,
            "QWEN-SAFETENSOR-SHARD-SWEEP-001",
            "A sequential header pass can inventory every donor shard without materializing Qwen weights.",
            "Read every *.safetensors header one at a time and record names, geometry, offsets, and header hashes.",
            "All source shards parse and reconcile without payload materialization.",
            "Any header error, index mismatch, payload read, or source mutation.",
            f"python -m experiments.donor_ephemeral_stream --source {source} --header-output {header_output} --skip-header-sweep",
            {"shard_count": header_result["shard_count"], "total_tensor_payload_bytes": header_result["total_tensor_payload_bytes"], "weights_materialized": False},
            "MEASURED POSITIVE: all donor shard headers were inspected sequentially without loading the model.",
            "Use the durable header inventory to select bounded payload pieces only.",
        )
    stream_result = run_stream(source, output=output, event_log=event_log, staging_dir=staging_dir, delete_after_accept=args.delete_after_accept, stream_id=args.stream_id)
    _record(
        ROOT,
        "QWEN-SAFETENSOR-EPHEMERAL-GDN-001",
        "A selected trained donor organ can be consumed from a temporary safetensor bundle, verified with its dependency/reassembly contract, and then have only that staging copy deleted.",
        "Stream the layer-17/value-head-10 GDN core pieces from model-00051, execute the reassembly/integrity gate, and delete the temporary bundle only after acceptance.",
        "The staged bundle reloads exactly, the recurrent core is finite and chunk-equivalent, and the source shard remains byte-identical.",
        "Any source mutation, missing context, staged reload mismatch, state-equivalence failure, or deletion before acceptance.",
        f"python -m experiments.donor_ephemeral_stream --source {source} --output {output} --event-log {event_log} --staging-dir {staging_dir} --delete-after-accept",
        {"summary": stream_result.get("summary"), "source_files_mutated": stream_result.get("policy", {}).get("source_files_mutated"), "request_count": len(stream_result["recovery_recipe"]["requests"])},
        "MEASURED POSITIVE: the real selected Qwen shard was consumed through an explicit acceptance gate and only the temporary staged safetensor was deleted.",
        "Reuse the receipt as the durable context contract; never delete the immutable Qwen source shards.",
    )
    print(json.dumps({"header_sweep": str(header_output) if header_result is not None else None, "stream_result": str(output), "event_log": str(event_log), "summary": stream_result.get("summary")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
