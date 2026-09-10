from __future__ import annotations

"""Read-only inspection of downloaded model artifacts.

This module deliberately has no model loader.  It reads small JSON metadata
files and, when available, safetensors headers through ``safe_open``.  It
never calls ``get_tensor`` and never constructs a Transformers model.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _dtype_name(dtype: Any) -> str:
    value = getattr(dtype, "name", None) or str(dtype)
    value = value.removeprefix("torch.").upper()
    aliases = {"FLOAT16": "F16", "BFLOAT16": "BF16", "FLOAT32": "F32", "INT64": "I64"}
    return aliases.get(value, value)


def _nbytes(shape: list[int], dtype: str) -> int | None:
    width = _DTYPE_BYTES.get(dtype)
    if width is None:
        return None
    count = 1
    for dimension in shape:
        count *= int(dimension)
    return count * width


def _tensor_class(name: str) -> str:
    lower = name.lower()
    if "visual" in lower or "vision" in lower:
        return "vision"
    if "ngram" in lower or "ple_" in lower:
        return "ngram_embedding"
    if "linear_attn" in lower or "deltanet" in lower or "mamba" in lower:
        return "recurrent_or_gated_linear_attention"
    if "sparse_attention" in lower or ".attn." in lower or "attention" in lower:
        return "attention"
    if "hyper_connection" in lower or "hyper_connections" in lower:
        return "residual_hyperconnection"
    if "expert" in lower or ".mlp." in lower:
        return "expert_or_mlp"
    if "embed" in lower or "lm_head" in lower:
        return "embedding_or_output"
    if "router" in lower:
        return "router"
    return "other"


def _metadata_files(root: Path) -> dict[str, dict[str, Any]]:
    names = (
        ".gitattributes",
        "LICENSE",
        "README.md",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "merges.txt",
        "vocab.json",
        "preprocessor_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
    )
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        path = root / name
        if path.is_file():
            result[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    return result


def _compatibility(config: dict[str, Any] | None, target_config: dict[str, Any] | None) -> dict[str, Any]:
    text_config = (config or {}).get("text_config") or (config or {})
    donor = {
        "model_type": text_config.get("model_type") or (config or {}).get("model_type"),
        "architectures": (config or {}).get("architectures", []),
        "hidden_size": text_config.get("hidden_size"),
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "num_attention_heads": text_config.get("num_attention_heads"),
        "num_key_value_heads": text_config.get("num_key_value_heads"),
        "vocab_size": text_config.get("vocab_size"),
        "max_position_embeddings": text_config.get("max_position_embeddings"),
        "num_experts": text_config.get("num_experts"),
        "num_experts_per_tok": text_config.get("num_experts_per_tok"),
        "layer_types": text_config.get("layer_types"),
    }
    if target_config is None:
        return {
            "status": "NOT_EVALUATED",
            "donor": donor,
            "target": None,
            "hard_mismatches": [],
            "recommended_import": "frozen_teacher_or_activation_distillation",
        }

    target = {
        "model_type": "remora-v0",
        "hidden_size": target_config.get("d_model"),
        "num_hidden_layers": target_config.get("n_layers"),
        "num_attention_heads": target_config.get("n_heads"),
        "vocab_size": target_config.get("vocab_size"),
        "max_position_embeddings": target_config.get("max_seq_len"),
        "bus_dim": target_config.get("bus_dim"),
    }
    mismatch_fields = {
        "hidden_size": (donor["hidden_size"], target["hidden_size"]),
        "num_hidden_layers": (donor["num_hidden_layers"], target["num_hidden_layers"]),
        "num_attention_heads": (donor["num_attention_heads"], target["num_attention_heads"]),
        "vocab_size": (donor["vocab_size"], target["vocab_size"]),
    }
    hard_mismatches = [
        {"field": field, "donor": left, "target": right}
        for field, (left, right) in mismatch_fields.items()
        if left is not None and right is not None and left != right
    ]
    architecture_mismatch = donor["model_type"] not in {None, "remora-v0"}
    if architecture_mismatch:
        hard_mismatches.append({"field": "model_type", "donor": donor["model_type"], "target": "remora-v0"})
    direct_weight_graft = not hard_mismatches and donor["model_type"] == "remora-v0"
    return {
        "status": "DIRECT_GRAFT_SHAPE_COMPATIBLE" if direct_weight_graft else "INCOMPATIBLE_FOR_DIRECT_GRAFT",
        "donor": donor,
        "target": target,
        "hard_mismatches": hard_mismatches,
        "recommended_import": "named_tensor_graft_with_ab_tests" if direct_weight_graft else "frozen_teacher_or_activation_distillation",
    }


def inspect_resident_model(
    path: str | Path,
    target_config: dict[str, Any] | None = None,
    include_tensor_headers: bool = True,
) -> dict[str, Any]:
    """Return a provenance-rich, value-free manifest for a model directory.

    ``include_tensor_headers`` may open every safetensors file, but only the
    format headers are read.  Tensor payloads are not materialized.
    """

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    config = _read_json(root / "config.json")
    index = _read_json(root / "model.safetensors.index.json")
    metadata = _metadata_files(root)
    shard_paths = sorted(root.glob("*.safetensors"))
    shards = [{"filename": p.name, "bytes": p.stat().st_size} for p in shard_paths]
    tensor_headers: list[dict[str, Any]] = []
    header_errors: list[dict[str, str]] = []
    shard_header_summary: list[dict[str, Any]] = []

    if include_tensor_headers and shard_paths:
        try:
            from safetensors import safe_open
        except ImportError:
            safe_open = None
        if safe_open is None:
            header_errors.append({"error": "safetensors package is unavailable"})
        else:
            for shard in shard_paths:
                summary = {"filename": shard.name, "tensor_count": 0, "metadata": None}
                try:
                    # No get_tensor call is permitted in this inspection path.
                    with safe_open(str(shard), framework="pt", device="cpu") as handle:
                        names = sorted(handle.keys())
                        summary["tensor_count"] = len(names)
                        summary["metadata"] = handle.metadata()
                        for name in names:
                            tensor_slice = handle.get_slice(name)
                            shape = [int(dimension) for dimension in tensor_slice.get_shape()]
                            dtype = _dtype_name(tensor_slice.get_dtype())
                            tensor_headers.append(
                                {
                                    "name": name,
                                    "filename": shard.name,
                                    "shape": shape,
                                    "dtype": dtype,
                                    "nbytes": _nbytes(shape, dtype),
                                    "tensor_class": _tensor_class(name),
                                }
                            )
                except Exception as exc:  # preserve a diagnostic, do not load around it
                    header_errors.append({"filename": shard.name, "error": f"{type(exc).__name__}: {exc}"})
                shard_header_summary.append(summary)

    index_names = sorted((index or {}).get("weight_map", {}))
    header_names = sorted(item["name"] for item in tensor_headers)
    missing_from_headers = sorted(set(index_names) - set(header_names)) if include_tensor_headers else []
    extra_in_headers = sorted(set(header_names) - set(index_names)) if include_tensor_headers and index else []
    total_payload = sum(item.get("nbytes") or 0 for item in tensor_headers)
    dtype_counts = Counter(item["dtype"] for item in tensor_headers)
    class_counts: dict[str, dict[str, int]] = {}
    for item in tensor_headers:
        row = class_counts.setdefault(item["tensor_class"], {"tensor_count": 0, "payload_bytes": 0})
        row["tensor_count"] += 1
        row["payload_bytes"] += item.get("nbytes") or 0

    local_source_manifest = _read_json(root / "FLASH_NEXT_BF16_SOURCE_MANIFEST.json")
    source = (local_source_manifest or {}).get("source", {})
    license_text = ""
    license_path = root / "LICENSE"
    if license_path.is_file():
        try:
            license_text = license_path.read_text(errors="replace")
        except OSError:
            license_text = ""
    license_name = license_text.splitlines()[0].strip() if license_text else None

    return {
        "schema": "remora-donor-manifest-v1",
        "path": str(root),
        "inspection": {
            "mode": "metadata_and_safetensors_headers_only",
            "weights_materialized": False,
            "model_loader_called": False,
            "get_tensor_called": False,
            "payload_hashes_performed": False,
        },
        "source": source,
        "license": {"name": license_name, "metadata_file": "LICENSE" if license_name else None, "review_required": not bool(license_name)},
        "files": {
            "metadata": metadata,
            "safetensor_shards": shards,
            "shard_count": len(shards),
            "total_shard_bytes": sum(item["bytes"] for item in shards),
        },
        "config": config,
        "tensor_index": {
            "present": index is not None,
            "indexed_tensor_count": len(index_names),
            "index_metadata": (index or {}).get("metadata"),
        },
        "headers": {
            "requested": include_tensor_headers,
            "parsed_tensor_count": len(tensor_headers),
            "parsed_payload_bytes": total_payload,
            "dtype_counts": dict(sorted(dtype_counts.items())),
            "class_counts": dict(sorted(class_counts.items())),
            "shard_summaries": shard_header_summary,
            "errors": header_errors,
            "index_name_reconciliation": {
                "ok": bool(index) and not header_errors and not missing_from_headers and not extra_in_headers,
                "missing_from_headers": missing_from_headers,
                "extra_in_headers": extra_in_headers,
            },
            "tensor_inventory": tensor_headers,
        },
        "compatibility": _compatibility(config, target_config),
        "local_acquisition": {
            "integrity_status": (local_source_manifest or {}).get("integrity_status"),
            "pinned_revision": source.get("revision"),
            "existing_header_autopsy": (local_source_manifest or {}).get("header_verification"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect resident model metadata without loading weights.")
    parser.add_argument("--path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-config")
    parser.add_argument("--no-tensor-headers", action="store_true")
    args = parser.parse_args()
    target = _read_json(Path(args.target_config)) if args.target_config else None
    result = inspect_resident_model(args.path, target, not args.no_tensor_headers)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "path": result["path"],
        "shard_count": result["files"]["shard_count"],
        "shard_bytes": result["files"]["total_shard_bytes"],
        "parsed_tensor_count": result["headers"]["parsed_tensor_count"],
        "header_errors": len(result["headers"]["errors"]),
        "weights_materialized": result["inspection"]["weights_materialized"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
