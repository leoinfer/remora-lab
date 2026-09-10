from __future__ import annotations

"""Bounded, provenance-first streaming of foreign safetensor shards.

The Qwen source tree is an immutable donor.  This module never deletes or
writes inside it.  It reads a shard header, materializes only explicitly
requested tensor regions, writes a short-lived staging ``.safetensors``
bundle, lets a caller verify/consume that bundle, and optionally deletes only
the staging file after an explicit acceptance decision.

The receipt is deliberately more detailed than a normal extraction receipt.
It records the information needed to reconstruct a tensor or reassemble a
subcircuit later after the payload copy has been deleted: source revision,
shard and header hashes, relative and absolute data offsets, parent tensor
geometry, slice coordinates, decoded tensor hashes, operation roles, and the
declared dependency/reassembly contract.
"""

import hashlib
import json
import os
import struct
import tempfile
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "C64": 8,
    "C128": 16,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash a file in bounded chunks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_region(path: Path, start: int, end: int) -> str:
    if start < 0 or end < start:
        raise ValueError(f"invalid byte region: {start}:{end}")
    digest = hashlib.sha256()
    remaining = end - start
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise EOFError(f"short read while hashing {path} at {start}:{end}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _json_read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _dtype_name(value: Any) -> str:
    text = str(value).removeprefix("torch.").upper()
    return {"FLOAT16": "F16", "BFLOAT16": "BF16", "FLOAT32": "F32", "FLOAT64": "F64"}.get(text, text)


def _numel(shape: Iterable[int]) -> int:
    result = 1
    for dimension in shape:
        result *= int(dimension)
    return result


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _source_identity(source_root: Path) -> dict[str, Any]:
    manifest = _json_read(source_root / "FLASH_NEXT_BF16_SOURCE_MANIFEST.json") or {}
    source = dict(manifest.get("source") or {})
    index_path = source_root / "model.safetensors.index.json"
    result: dict[str, Any] = {
        "path": str(source_root),
        "repository": source.get("repository"),
        "revision": source.get("revision"),
        "revision_type": source.get("revision_type"),
        "integrity_status": manifest.get("integrity_status"),
        "index_path": str(index_path) if index_path.is_file() else None,
        "index_sha256": sha256_file(index_path) if index_path.is_file() else None,
    }
    metadata_files = ["config.json", "model.safetensors.index.json", "FLASH_NEXT_BF16_SOURCE_MANIFEST.json"]
    result["metadata_files"] = {
        name: {"bytes": int((source_root / name).stat().st_size), "sha256": sha256_file(source_root / name)}
        for name in metadata_files
        if (source_root / name).is_file()
    }
    return result


def read_safetensors_header(path: str | Path) -> dict[str, Any]:
    """Read and validate one safetensors header without touching payload data."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    file_bytes = int(source.stat().st_size)
    with source.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError(f"safetensors file has no 8-byte header length: {source}")
        header_length = int(struct.unpack("<Q", prefix)[0])
        if header_length > file_bytes - 8:
            raise ValueError(f"safetensors header exceeds file: {source}")
        header_bytes = handle.read(header_length)
    if len(header_bytes) != header_length:
        raise EOFError(f"short safetensors header: {source}")
    try:
        raw_header = json.loads(header_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid safetensors JSON header: {source}") from exc
    if not isinstance(raw_header, dict):
        raise ValueError(f"safetensors header is not an object: {source}")

    data_start = 8 + header_length
    metadata = raw_header.get("__metadata__")
    tensors: list[dict[str, Any]] = []
    for name, info in sorted(raw_header.items()):
        if name == "__metadata__":
            continue
        if not isinstance(info, dict):
            raise ValueError(f"invalid tensor header for {name!r} in {source}")
        shape = [int(dimension) for dimension in info.get("shape", [])]
        dtype = _dtype_name(info.get("dtype"))
        offsets = info.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"invalid data_offsets for {name!r} in {source}")
        start, end = int(offsets[0]), int(offsets[1])
        absolute_start, absolute_end = data_start + start, data_start + end
        if start < 0 or end < start or absolute_end > file_bytes:
            raise ValueError(f"tensor offsets outside file for {name!r} in {source}")
        payload_bytes = end - start
        expected_bytes = _numel(shape) * _DTYPE_BYTES[dtype] if dtype in _DTYPE_BYTES else None
        if expected_bytes is not None and expected_bytes != payload_bytes:
            raise ValueError(
                f"tensor byte mismatch for {name!r} in {source}: header={payload_bytes} geometry={expected_bytes}"
            )
        tensors.append(
            {
                "name": name,
                "shape": shape,
                "dtype": dtype,
                "nbytes": payload_bytes,
                "data_offsets": [start, end],
                "absolute_file_offsets": [absolute_start, absolute_end],
            }
        )

    return {
        "filename": source.name,
        "path": str(source),
        "file_bytes": file_bytes,
        "header_length": header_length,
        "data_start": data_start,
        "header_sha256": _sha256_bytes(header_bytes),
        "metadata": metadata,
        "tensor_count": len(tensors),
        "payload_bytes": int(sum(row["nbytes"] for row in tensors)),
        "tensors": tensors,
    }


def scan_safetensor_shards(source_root: str | Path, *, include_tensor_inventory: bool = True) -> dict[str, Any]:
    """Scan every donor shard header sequentially; no payload is materialized."""

    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards in {root}")
    rows = []
    for shard in shards:
        header = read_safetensors_header(shard)
        if not include_tensor_inventory:
            header = {key: value for key, value in header.items() if key != "tensors"}
        rows.append(header)
    return {
        "schema": "remora-donor-safetensor-header-sweep-v1",
        "source": _source_identity(root),
        "inspection": {
            "mode": "one-shard-at-a-time-header-only",
            "weights_materialized": False,
            "model_loader_called": False,
            "source_files_mutated": False,
            "payload_hashes_performed": False,
            "full_shard_hashes_performed": False,
        },
        "shard_count": len(rows),
        "total_shard_bytes": int(sum(row["file_bytes"] for row in rows)),
        "total_tensor_payload_bytes": int(sum(row["payload_bytes"] for row in rows)),
        "shards": rows,
    }


def _normalise_range(value: Any, *, axis: int, shape: list[int]) -> list[int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"slice axis {axis} must be [start, end]")
    start, end = int(value[0]), int(value[1])
    if start < 0 or end < start or end > shape[axis]:
        raise ValueError(f"slice axis {axis} outside shape {shape}: {value}")
    return [start, end]


def _normalise_slice(spec: Any, shape: list[int]) -> dict[str, Any]:
    if spec in (None, {}, {"axes": {}}):
        return {"axes": {}, "kind": "full_tensor"}
    if not isinstance(spec, dict):
        raise ValueError("slice must be an object")
    axes: dict[str, list[int]] = {}
    raw_axes = spec.get("axes")
    if raw_axes is not None:
        if not isinstance(raw_axes, dict):
            raise ValueError("slice.axes must be an object")
        for raw_axis, value in raw_axes.items():
            axis = int(raw_axis)
            if axis < 0 or axis >= len(shape):
                raise ValueError(f"slice axis outside tensor rank: {axis}")
            axes[str(axis)] = _normalise_range(value, axis=axis, shape=shape)
    if "rows" in spec:
        axes["0"] = _normalise_range(spec["rows"], axis=0, shape=shape)
    if "columns" in spec:
        if len(shape) < 2:
            raise ValueError("columns slice requires a rank-2 tensor")
        axes["1"] = _normalise_range(spec["columns"], axis=1, shape=shape)
    return {"axes": dict(sorted(axes.items(), key=lambda item: int(item[0]))), "kind": "slice" if axes else "full_tensor"}


def _slice_handle_value(handle: Any, tensor_name: str, slice_spec: dict[str, Any]):
    view = handle.get_slice(tensor_name)
    axes = slice_spec.get("axes", {})
    if not axes:
        return view[:].contiguous()
    selectors = [slice(None)] * len(view.get_shape())
    for raw_axis, bounds in axes.items():
        selectors[int(raw_axis)] = slice(int(bounds[0]), int(bounds[1]))
    return view[tuple(selectors)].contiguous()


def _tensor_sha256(tensor: Any) -> str:
    import torch

    value = tensor.detach().cpu().contiguous()
    return _sha256_bytes(value.view(torch.uint8).numpy().tobytes())


def _safe_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _append_event(path: Path | None, event: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")


def _normalise_decision(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"accepted": value, "verifier_id": "boolean-callback", "reason": "callback_boolean", "metrics": {}}
    if value is None:
        return {"accepted": False, "verifier_id": "no-decision", "reason": "consumer_returned_none", "metrics": {}}
    if not isinstance(value, dict):
        raise TypeError("consumer decision must be bool, None, or a dictionary")
    accepted = value.get("accepted")
    if not isinstance(accepted, bool):
        raise ValueError("consumer decision must contain boolean 'accepted'")
    return {
        "accepted": accepted,
        "verifier_id": str(value.get("verifier_id", "unspecified")),
        "reason": str(value.get("reason", "unspecified")),
        "metrics": dict(value.get("metrics") or {}),
        "satisfied_context": value.get("satisfied_context", {}),
    }


def _temporary_stage_path(staging_root: Path, stream_id: str, sequence: int) -> Path:
    fd, name = tempfile.mkstemp(prefix=f".remora-{stream_id}-{sequence:04d}-", suffix=".safetensors", dir=staging_root)
    os.close(fd)
    return Path(name)


def _assert_safe_staging(source_root: Path, staging_root: Path) -> None:
    if staging_root == source_root or source_root in staging_root.parents:
        raise ValueError("staging directory must not be inside the immutable donor source tree")
    allowed_roots = {Path(tempfile.gettempdir()).resolve(), Path("/var/tmp").resolve()}
    if not any(staging_root == root or root in staging_root.parents for root in allowed_roots):
        raise ValueError("deletable staging must live below /tmp or /var/tmp")


def stream_selected_payload(
    source_root: str | Path,
    selection: dict[str, Any],
    *,
    staging_dir: str | Path,
    receipt_path: str | Path | None = None,
    event_log_path: str | Path | None = None,
    allow_payload: bool = False,
    delete_after_accept: bool = False,
    hash_source_shards: bool = True,
    consumer: Callable[[Path, dict[str, Any], dict[str, Any]], Any] | None = None,
    stream_id: str | None = None,
) -> dict[str, Any]:
    """Stream explicitly selected payload pieces and delete only accepted staging.

    ``consumer(bundle_path, tensors, context)`` is the acceptance gate.  It
    must return ``{"accepted": True, ...}`` before a temporary bundle can be
    deleted.  The source shard is never a deletion target.  A selection may
    contain slices of larger parent tensors; the receipt logs the parent
    offsets and the exact reassembly contract supplied by the caller.
    """

    if not allow_payload:
        raise PermissionError("payload streaming requires explicit allow_payload=True")
    if delete_after_accept and consumer is None:
        raise ValueError("delete_after_accept requires an explicit consumer acceptance gate")

    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    staging_root = Path(staging_dir).expanduser().resolve()
    staging_root.mkdir(parents=True, exist_ok=True)
    _assert_safe_staging(root, staging_root)
    receipt = Path(receipt_path).expanduser().resolve() if receipt_path is not None else None
    events = Path(event_log_path).expanduser().resolve() if event_log_path is not None else None
    if receipt is not None and (receipt == root or root in receipt.parents):
        raise ValueError("receipt must not be written inside the immutable donor source tree")
    if events is not None and (events == root or root in events.parents):
        raise ValueError("event log must not be written inside the immutable donor source tree")

    requests = list(selection.get("requests") or [])
    if not requests:
        raise ValueError("selection contains no tensor requests")
    request_ids: set[str] = set()
    requested_by_shard: dict[str, list[dict[str, Any]]] = defaultdict(list)
    index = _json_read(root / "model.safetensors.index.json") or {}
    weight_map = dict(index.get("weight_map") or {})
    header_cache: dict[str, dict[str, Any]] = {}
    tensor_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_request in requests:
        if not isinstance(raw_request, dict):
            raise ValueError("every tensor request must be an object")
        request = dict(raw_request)
        request_id = str(request.get("request_id") or request.get("role") or "")
        tensor_name = str(request.get("tensor_name") or "")
        if not request_id or not tensor_name:
            raise ValueError("every tensor request needs request_id and tensor_name")
        if request_id in request_ids or request_id == "__metadata__":
            raise ValueError(f"duplicate/invalid request_id: {request_id}")
        request_ids.add(request_id)
        expected_shard = weight_map.get(tensor_name)
        declared_shard = request.get("source_shard")
        if declared_shard is not None and expected_shard is not None and str(declared_shard) != str(expected_shard):
            raise ValueError(f"request shard disagrees with index for {tensor_name}")
        shard = str(declared_shard or expected_shard or "")
        if not shard:
            raise KeyError(f"tensor is absent from model.safetensors.index.json: {tensor_name}")
        request["request_id"] = request_id
        request["tensor_name"] = tensor_name
        request["source_shard"] = shard
        requested_by_shard[shard].append(request)

    source_info = _source_identity(root)
    stream_id = str(stream_id or f"stream-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}")
    result: dict[str, Any] = {
        "schema": "remora-donor-ephemeral-safetensor-stream-v1",
        "stream_id": stream_id,
        "started_at": _utc_now(),
        "source": source_info,
        "selection": selection,
        "policy": {
            "source_tree_immutable": True,
            "source_files_mutated": False,
            "source_files_deleted": [],
            "one_source_shard_at_a_time": True,
            "one_staging_bundle_per_selected_shard": True,
            "staging_root": str(staging_root),
            "delete_after_accept": bool(delete_after_accept),
            "deletion_scope": "temporary_staging_bundle_only",
            "full_source_shard_hashes": bool(hash_source_shards),
        },
        "shards": [],
        "recovery_recipe": {
            "source": source_info,
            "index_path": str(root / "model.safetensors.index.json"),
            "requests": requests,
            "reassembly_contract": selection.get("reassembly") or selection.get("contract") or {},
            "required_shards": sorted(requested_by_shard),
            "note": "Payload staging copies may be deleted; re-read the pinned source shard and exact offsets/slices to reconstruct.",
        },
    }

    def persist() -> None:
        if receipt is not None:
            _safe_json_write(receipt, result)

    persist()
    for sequence, shard_name in enumerate(sorted(requested_by_shard), start=1):
        shard_path = (root / shard_name).resolve()
        if shard_path.parent != root or not shard_path.is_file():
            raise FileNotFoundError(f"requested shard is not an immediate source child: {shard_path}")
        header = header_cache.setdefault(shard_name, read_safetensors_header(shard_path))
        header_by_name = {row["name"]: row for row in header["tensors"]}
        source_hash = sha256_file(shard_path) if hash_source_shards else None
        shard_record: dict[str, Any] = {
            "sequence": sequence,
            "source_shard": shard_name,
            "source_file": {
                "path": str(shard_path),
                "bytes": int(shard_path.stat().st_size),
                "sha256": source_hash,
                "sha256_status": "MEASURED" if source_hash else "NOT_COMPUTED",
                "source_mutated": False,
            },
            "header": {
                "header_length": header["header_length"],
                "data_start": header["data_start"],
                "header_sha256": header["header_sha256"],
                "tensor_count": header["tensor_count"],
                "payload_bytes": header["payload_bytes"],
            },
            "requests": [],
            "staging": {},
        }
        _append_event(events, {"schema": "remora-donor-safetensor-lifecycle-event-v1", "event": "HEADER_READ", "stream_id": stream_id, "sequence": sequence, "source_shard": shard_name, "header": shard_record["header"], "at": _utc_now()})

        import safetensors.torch
        from safetensors import safe_open

        tensors: dict[str, Any] = {}
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for request in sorted(requested_by_shard[shard_name], key=lambda row: row["request_id"]):
                tensor_name = request["tensor_name"]
                if tensor_name not in header_by_name or tensor_name not in set(handle.keys()):
                    raise KeyError(f"requested tensor {tensor_name!r} is not present in {shard_name}")
                parent = header_by_name[tensor_name]
                normalised_slice = _normalise_slice(request.get("slice"), list(parent["shape"]))
                value = _slice_handle_value(handle, tensor_name, normalised_slice)
                key = str(request["request_id"])
                tensors[key] = value
                parent_start, parent_end = parent["absolute_file_offsets"]
                parent_cache_key = (shard_name, tensor_name)
                parent_context = tensor_cache.get(parent_cache_key)
                if parent_context is None:
                    parent_context = {
                        "source_tensor_payload_sha256": _sha256_region(shard_path, parent_start, parent_end),
                        "source_tensor_payload_bytes": int(parent["nbytes"]),
                    }
                    tensor_cache[parent_cache_key] = parent_context
                request_record = {
                    "request_id": key,
                    "role": request.get("role"),
                    "tensor_name": tensor_name,
                    "source_shard": shard_name,
                    "source_shape": parent["shape"],
                    "source_dtype": parent["dtype"],
                    "source_tensor_nbytes": parent["nbytes"],
                    "source_data_offsets": parent["data_offsets"],
                    "source_absolute_file_offsets": parent["absolute_file_offsets"],
                    "source_tensor_payload_sha256": parent_context["source_tensor_payload_sha256"],
                    "slice": normalised_slice,
                    "materialized_shape": [int(dimension) for dimension in value.shape],
                    "materialized_dtype": _dtype_name(value.dtype),
                    "materialized_nbytes": int(value.numel() * value.element_size()),
                    "materialized_tensor_sha256": _tensor_sha256(value),
                    "operation": request.get("operation"),
                    "reassembly": request.get("reassembly"),
                    "depends_on": list(request.get("depends_on") or []),
                    "context": dict(request.get("context") or {}),
                }
                request_record["source_read_bytes"] = int(value.numel() * value.element_size())
                shard_record["requests"].append(request_record)

        stage_path = _temporary_stage_path(staging_root, stream_id, sequence)
        metadata = {
            "remora_schema": "remora-ephemeral-donor-staging-v1",
            "stream_id": stream_id,
            "component_id": str(selection.get("component_id", "unspecified")),
            "source_shard": shard_name,
            "source_revision": str(source_info.get("revision", "unknown")),
            "request_ids": json.dumps(sorted(tensors)),
        }
        try:
            safetensors.torch.save_file(tensors, str(stage_path), metadata=metadata)
        except Exception:
            stage_path.unlink(missing_ok=True)
            raise
        stage_hash = sha256_file(stage_path)
        shard_record["staging"] = {
            "path": str(stage_path),
            "bytes": int(stage_path.stat().st_size),
            "sha256": stage_hash,
            "payload_keys": sorted(tensors),
            "status": "STAGED",
        }
        _append_event(events, {"schema": "remora-donor-safetensor-lifecycle-event-v1", "event": "STAGED", "stream_id": stream_id, "sequence": sequence, "source_shard": shard_name, "staging_path": str(stage_path), "staging_sha256": stage_hash, "at": _utc_now()})
        result["shards"].append(shard_record)
        persist()

        try:
            decision = _normalise_decision(consumer(stage_path, tensors, shard_record) if consumer is not None else None)
        except Exception as exc:
            # A failed verifier is not permission to delete anything.  Keep
            # the staged bundle and persist the diagnostic before re-raising
            # so a later run can inspect the exact failure without rereading
            # the source blindly.
            shard_record["decision"] = {
                "accepted": False,
                "verifier_id": "consumer-exception",
                "reason": "consumer_raised_exception",
                "metrics": {},
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            shard_record["staging"]["status"] = "RETAINED_CONSUMER_ERROR"
            _append_event(events, {"schema": "remora-donor-safetensor-lifecycle-event-v1", "event": "CONSUMER_ERROR", "stream_id": stream_id, "sequence": sequence, "source_shard": shard_name, "accepted": False, "error_type": type(exc).__name__, "error": str(exc), "staging_path": str(stage_path), "at": _utc_now()})
            persist()
            raise
        shard_record["decision"] = decision
        _append_event(events, {"schema": "remora-donor-safetensor-lifecycle-event-v1", "event": "ACCEPTANCE_DECISION", "stream_id": stream_id, "sequence": sequence, "source_shard": shard_name, "accepted": decision["accepted"], "verifier_id": decision["verifier_id"], "reason": decision["reason"], "at": _utc_now()})
        if decision["accepted"] and delete_after_accept:
            if stage_path.parent != staging_root or not stage_path.name.startswith(f".remora-{stream_id}-"):
                raise RuntimeError(f"refusing to delete unexpected staging path: {stage_path}")
            shard_record["staging"]["status"] = "ACCEPTED_PENDING_DELETE"
            persist()
            stage_path.unlink()
            if stage_path.exists():
                raise OSError(f"staging bundle survived deletion: {stage_path}")
            shard_record["staging"].update({"status": "DELETED_AFTER_ACCEPT", "deleted_at": _utc_now(), "path_exists_after_delete": False})
            _append_event(events, {"schema": "remora-donor-safetensor-lifecycle-event-v1", "event": "EPHEMERAL_BUNDLE_DELETED", "stream_id": stream_id, "sequence": sequence, "source_shard": shard_name, "staging_path": str(stage_path), "source_mutated": False, "at": shard_record["staging"]["deleted_at"]})
        elif decision["accepted"]:
            shard_record["staging"]["status"] = "ACCEPTED_RETAINED_DELETE_NOT_REQUESTED"
        else:
            shard_record["staging"]["status"] = "RETAINED_REJECTED_FOR_REVIEW"
        persist()

    result["completed_at"] = _utc_now()
    result["summary"] = {
        "selected_shard_count": len(result["shards"]),
        "selected_request_count": len(requests),
        "accepted_shard_count": sum(1 for row in result["shards"] if row.get("decision", {}).get("accepted") is True),
        "deleted_ephemeral_bundle_count": sum(1 for row in result["shards"] if row.get("staging", {}).get("status") == "DELETED_AFTER_ACCEPT"),
        "retained_bundle_count": sum(1 for row in result["shards"] if row.get("staging", {}).get("status") != "DELETED_AFTER_ACCEPT"),
        "source_files_mutated": False,
        "source_files_deleted": [],
        "staging_bundles_deleted_for_source_shards": [row["source_shard"] for row in result["shards"] if row.get("staging", {}).get("status") == "DELETED_AFTER_ACCEPT"],
    }
    persist()
    return result
