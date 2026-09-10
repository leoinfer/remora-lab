from __future__ import annotations

"""Explicit, bounded extraction of named donor tensors.

This is intentionally separate from :mod:`remora.donors.manifest`.  The
manifest path is value-free; this path may materialize only a preselected,
byte-bounded tensor set after an explicit caller opt-in.
"""

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_selected_tensors(
    source_root: str | Path,
    manifest: dict[str, Any],
    selection: dict[str, Any],
    output_path: str | Path,
    *,
    allow_payload: bool = False,
    max_payload_bytes: int = 256 * 1024 * 1024,
) -> dict[str, Any]:
    """Materialize only selected tensors into a standalone safetensors file.

    ``allow_payload`` is deliberately mandatory.  The caller must supply a
    prior header manifest and a value-free selection generated from it.  The
    source model is never passed to a model loader and the output cannot be
    written inside the source tree.
    """

    if not allow_payload:
        raise PermissionError("payload extraction requires explicit allow_payload=True")

    source = Path(source_root).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    output = Path(output_path).expanduser().resolve()
    if output == source or source in output.parents:
        raise ValueError("refusing to write extracted payload inside donor source tree")
    if output.exists():
        raise FileExistsError(output)

    inventory = manifest.get("headers", {}).get("tensor_inventory", [])
    by_name = {str(row["name"]): row for row in inventory}
    selected_rows = selection.get("selected", [])
    if not selection.get("selection_is_value_free", False):
        raise ValueError("selection must be marked value-free")
    if not selected_rows:
        raise ValueError("selection contains no components")

    requested: list[str] = []
    for row in selected_rows:
        requested.extend(str(name) for name in row.get("tensor_names", []))
    if len(requested) != len(set(requested)):
        raise ValueError("selection contains duplicate tensor names")
    missing = sorted(set(requested) - set(by_name))
    if missing:
        raise KeyError(f"selection names absent from manifest: {missing[:3]}")

    expected_payload_bytes = sum(int(by_name[name].get("nbytes") or 0) for name in requested)
    declared_payload_bytes = int(selection.get("selected_payload_bytes", expected_payload_bytes))
    if declared_payload_bytes != expected_payload_bytes:
        raise ValueError(
            f"selection byte accounting mismatch: declared={declared_payload_bytes} expected={expected_payload_bytes}"
        )
    if expected_payload_bytes > int(max_payload_bytes):
        raise ValueError(
            f"selection exceeds extraction budget: {expected_payload_bytes} > {int(max_payload_bytes)}"
        )

    by_shard: dict[str, list[str]] = defaultdict(list)
    for name in requested:
        by_shard[str(by_name[name]["filename"])].append(name)

    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:  # pragma: no cover - package is a project dependency
        raise RuntimeError("safetensors is required for explicit payload extraction") from exc

    tensors: dict[str, Any] = {}
    observed_payload_bytes = 0
    source_shards: list[str] = []
    for filename in sorted(by_shard):
        shard = source / filename
        if not shard.is_file():
            raise FileNotFoundError(shard)
        source_shards.append(filename)
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for name in sorted(by_shard[filename]):
                if name not in available:
                    raise KeyError(f"tensor {name!r} is not present in shard {filename!r}")
                tensor = handle.get_tensor(name)
                tensors[name] = tensor.contiguous()
                observed_payload_bytes += int(tensor.numel() * tensor.element_size())

    if observed_payload_bytes != expected_payload_bytes:
        raise ValueError(
            f"materialized byte accounting mismatch: observed={observed_payload_bytes} expected={expected_payload_bytes}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{output.name}.", suffix=".partial", dir=output.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        save_file(
            tensors,
            str(temporary_path),
            metadata={
                "remora_schema": "remora-donor-payload-v1",
                "source_repository": str(manifest.get("source", {}).get("repository", "unknown")),
                "source_revision": str(manifest.get("source", {}).get("revision", "unknown")),
                "selection_schema": str(selection.get("schema", "unknown")),
            },
        )
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return {
        "schema": "remora-donor-extraction-v1",
        "source_path": str(source),
        "source_repository": manifest.get("source", {}).get("repository"),
        "source_revision": manifest.get("source", {}).get("revision"),
        "selection_schema": selection.get("schema"),
        "component_keys": [str(row["component_key"]) for row in selected_rows],
        "tensor_names": requested,
        "source_shards": source_shards,
        "expected_payload_bytes": expected_payload_bytes,
        "observed_payload_bytes": observed_payload_bytes,
        "budget_bytes": int(max_payload_bytes),
        "payload_materialized": True,
        "model_loader_called": False,
        "output_path": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": _sha256(output),
        "promotion_state": "CANDIDATE_EXTRACTED",
        "license_review_required": True,
    }


def write_extraction_receipt(path: str | Path, receipt: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
