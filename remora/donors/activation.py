from __future__ import annotations

"""Bounded, provenance-checked hidden-state interchange for donor models.

This module is the activation counterpart to remora.donors.response. It
accepts tensors only after an explicitly launched donor runtime has produced
them; it never constructs or loads the donor model itself.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_tensor(value: torch.Tensor) -> str:
    # A byte view also works for BF16 tensors, whose direct NumPy conversion
    # is not supported by every PyTorch/NumPy pairing.
    data = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _dtype_name(dtype: Any) -> str:
    value = getattr(dtype, "name", None) or str(dtype)
    return (
        value.removeprefix("torch.")
        .upper()
        .replace("BOOL", "BOOL")
        .replace("UINT8", "U8")
        .replace("INT8", "I8")
        .replace("INT16", "I16")
        .replace("INT32", "I32")
        .replace("INT64", "I64")
        .replace("FLOAT16", "F16")
        .replace("BFLOAT16", "BF16")
        .replace("FLOAT32", "F32")
        .replace("FLOAT64", "F64")
        .replace("INT64", "I64")
    )


def _nbytes(shape: Iterable[int], dtype: str) -> int:
    width = _DTYPE_BYTES.get(dtype)
    if width is None:
        raise ValueError(f"unsupported activation dtype {dtype}")
    count = 1
    for dimension in shape:
        count *= int(dimension)
    return count * width


@dataclass(frozen=True)
class DonorActivationRecord:
    record_id: str
    donor_id: str
    layer_name: str
    prompt_sha256: str
    shape: list[int]
    dtype: str
    activation_sha256: str
    lineage_key: str
    runtime: dict[str, Any]
    accepted: bool

    @classmethod
    def create(
        cls,
        record_id: str,
        donor_id: str,
        layer_name: str,
        prompt: str,
        activation: torch.Tensor,
        *,
        lineage_key: str,
        runtime: dict[str, Any],
        accepted: bool,
    ) -> "DonorActivationRecord":
        if not isinstance(activation, torch.Tensor):
            raise TypeError("activation must be a torch.Tensor")
        record = cls(
            record_id=record_id,
            donor_id=donor_id,
            layer_name=layer_name,
            prompt_sha256=_sha256_text(prompt),
            shape=[int(dimension) for dimension in activation.shape],
            dtype=_dtype_name(activation.dtype),
            activation_sha256=_sha256_tensor(activation),
            lineage_key=lineage_key,
            runtime=dict(runtime),
            accepted=bool(accepted),
        )
        record.verify_integrity()
        return record

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "DonorActivationRecord":
        record = cls(**row)
        record.verify_integrity()
        return record

    def verify_integrity(self) -> None:
        if not self.record_id or not self.donor_id or not self.layer_name:
            raise ValueError(f"activation identity is incomplete for {self.record_id}")
        if any(int(dimension) < 0 for dimension in self.shape):
            raise ValueError(f"invalid activation shape for {self.record_id}")
        _nbytes(self.shape, self.dtype)
        if len(self.prompt_sha256) != 64 or len(self.activation_sha256) != 64:
            raise ValueError(f"invalid activation hash for {self.record_id}")
        if not self.lineage_key:
            raise ValueError(f"missing lineage key for {self.record_id}")
        if not self.runtime.get("runtime_id"):
            raise ValueError(f"missing runtime identity for {self.record_id}")


def write_activation_records(path: str | Path, records: Iterable[DonorActivationRecord]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w") as handle:
        for record in records:
            record.verify_integrity()
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
            count += 1
    return count


def load_activation_records(path: str | Path) -> list[DonorActivationRecord]:
    records: list[DonorActivationRecord] = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(DonorActivationRecord.from_dict(json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid donor activation at line {line_number}: {exc}") from exc
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("duplicate donor activation record_id")
    return records


def write_activation_bundle(
    path: str | Path,
    records: Iterable[DonorActivationRecord],
    activations: dict[str, torch.Tensor],
) -> int:
    """Write a standalone safetensors bundle after validating every record."""

    records = list(records)
    record_map = {record.record_id: record for record in records}
    if len(record_map) != len(records):
        raise ValueError("duplicate donor activation record_id")
    if set(record_map) != set(activations):
        raise ValueError("activation bundle keys must exactly match record IDs")
    payload: dict[str, torch.Tensor] = {}
    total = 0
    for record in records:
        record.verify_integrity()
        activation = activations[record.record_id]
        if [int(dimension) for dimension in activation.shape] != record.shape:
            raise ValueError(f"activation shape mismatch for {record.record_id}")
        if _dtype_name(activation.dtype) != record.dtype:
            raise ValueError(f"activation dtype mismatch for {record.record_id}")
        if _sha256_tensor(activation) != record.activation_sha256:
            raise ValueError(f"activation hash mismatch for {record.record_id}")
        total += _nbytes(record.shape, record.dtype)
        payload[record.record_id] = activation.detach().cpu().contiguous()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file

    save_file(
        payload,
        str(output),
        metadata={"schema": "donor-activation-v1", "record_count": str(len(records))},
    )
    return total


def load_activation_bundle(
    bundle_path: str | Path,
    records_path: str | Path,
    *,
    max_payload_bytes: int = 256 * 1024 * 1024,
    accepted_only: bool = True,
) -> dict[str, Any]:
    """Load only listed, accepted activations under an explicit byte budget."""

    records = load_activation_records(records_path)
    selected = [record for record in records if not accepted_only or record.accepted]
    selected_map = {record.record_id: record for record in selected}
    all_ids = {record.record_id for record in records}
    from safetensors import safe_open

    tensors: dict[str, torch.Tensor] = {}
    payload_bytes = 0
    with safe_open(str(bundle_path), framework="pt", device="cpu") as handle:
        bundle_ids = set(handle.keys())
        unexpected = bundle_ids - all_ids
        missing = set(selected_map) - bundle_ids
        if unexpected:
            raise ValueError(f"activation bundle contains unlisted tensors: {sorted(unexpected)}")
        if missing:
            raise ValueError(f"activation bundle is missing records: {sorted(missing)}")
        for record_id, record in selected_map.items():
            tensor_slice = handle.get_slice(record_id)
            shape = [int(dimension) for dimension in tensor_slice.get_shape()]
            dtype = _dtype_name(tensor_slice.get_dtype())
            if shape != record.shape or dtype != record.dtype:
                raise ValueError(f"activation header mismatch for {record_id}")
            payload_bytes += _nbytes(shape, dtype)
        if payload_bytes > max_payload_bytes:
            raise MemoryError(f"activation payload {payload_bytes} exceeds budget {max_payload_bytes}")
        for record_id, record in selected_map.items():
            tensor = handle.get_tensor(record_id)
            if _sha256_tensor(tensor) != record.activation_sha256:
                raise ValueError(f"activation hash mismatch for {record_id}")
            tensors[record_id] = tensor
    return {
        "schema": "donor-activation-v1",
        "records": selected,
        "activations": tensors,
        "payload_bytes": payload_bytes,
        "max_payload_bytes": max_payload_bytes,
        "payload_materialized": True,
        "model_loader_called": False,
    }


def activation_contract_signature() -> dict[str, Any]:
    return {
        "version": "donor-activation-v1",
        "required_record_fields": [
            "record_id",
            "donor_id",
            "layer_name",
            "prompt_sha256",
            "shape",
            "dtype",
            "activation_sha256",
            "lineage_key",
            "runtime",
            "accepted",
        ],
        "policy": "read only listed accepted tensors under a byte budget; donor remains external and frozen; verify headers and content hashes before port training",
    }
