from __future__ import annotations

"""Build a value-free neural anatomy graph for a resident donor checkpoint.

The graph is deliberately derived from the donor's config, safetensors index,
and safetensors headers.  It does not call ``get_tensor`` and therefore cannot
silently turn an anatomy pass into a model load.  Payload observations can be
attached later by a separate, explicitly bounded experiment.
"""

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_MODEL_LAYER_PREFIX = "model.language_model.layers."


def _config_view(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, dict) else config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize the two header formats already present in the repository."""

    rows = manifest.get("headers", {}).get("tensor_inventory")
    if rows is None:
        rows = manifest.get("tensor_inventory", [])
    result: list[dict[str, Any]] = []
    for raw in rows:
        name = raw.get("name", raw.get("tensor_name"))
        filename = raw.get("filename", raw.get("shard"))
        shape = raw.get("shape", [])
        if not name or not filename:
            continue
        shape = [int(value) for value in shape]
        nbytes = raw.get("nbytes", raw.get("payload_bytes"))
        if nbytes is None:
            width = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8}.get(str(raw.get("dtype", "")))
            nbytes = math.prod(shape) * width if width is not None else 0
        result.append(
            {
                "tensor_name": str(name),
                "source_shard": str(filename),
                "shape": shape,
                "dtype": str(raw.get("dtype", "UNKNOWN")),
                "payload_bytes": int(nbytes or 0),
                "tensor_class": str(raw.get("tensor_class", "other")),
            }
        )
    return result


def _layer_index(name: str) -> int | None:
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def _namespace(component_key: str) -> str:
    if component_key.startswith("model.language_model."):
        return "language"
    if component_key.startswith("mtp."):
        return "mtp"
    return "global"


def _structural_signature(component_key: str) -> str:
    """Collapse repeated layer instances into one architecture candidate."""

    return re.sub(r"\.layers\.\d+(?=\.)", ".layers.*", component_key)


def _component_key(name: str) -> str:
    """Return a stable component boundary, including packed expert arrays."""

    layer = _layer_index(name)
    if layer is not None and _MODEL_LAYER_PREFIX in name:
        prefix = f"model.language_model.layers.{layer}"
        if ".linear_attn." in name:
            return f"{prefix}.linear_attn"
        if ".attn_hyper_connection." in name:
            return f"{prefix}.attn_hyper_connection"
        if ".mlp_hyper_connection." in name:
            return f"{prefix}.mlp_hyper_connection"
        if ".mlp.shared_expert" in name or ".mlp.shared_expert_gate." in name:
            return f"{prefix}.mlp.shared_expert"
        if ".mlp.experts." in name:
            return f"{prefix}.mlp.routed_experts"
        if ".mlp.gate." in name:
            return f"{prefix}.mlp.router"
        if ".ple." in name:
            return f"{prefix}.ple"
        return f"{prefix}.other"
    if name.startswith("mtp."):
        if ".hyper_connection_mixer." in name:
            return "mtp.hyper_connection_mixer"
        parts = name.split(".")
        if len(parts) >= 3 and parts[1] == "layers" and parts[2].isdigit():
            prefix = ".".join(parts[:3])
            if ".mlp.experts." in name:
                return f"{prefix}.mlp.routed_experts"
            if ".mlp.shared_expert" in name:
                return f"{prefix}.mlp.shared_expert"
            if ".mlp.gate." in name:
                return f"{prefix}.mlp.router"
            return f"{prefix}.other"
    if "visual" in name or "vision" in name:
        return "model.visual_encoder"
    if "embed" in name or "lm_head" in name:
        return "model.token_embedding_or_lm_head"
    return ".".join(name.split(".")[:3]) or name


def _family(component_key: str, names: Iterable[str]) -> str:
    joined = " ".join(names).lower()
    key = component_key.lower()
    if "linear_attn" in key or "deltanet" in joined:
        return "gated_deltanet_linear_attention"
    if "attn_hyper_connection" in key or "mlp_hyper_connection" in key or "hyper_connection" in joined:
        return "gated_residual_hyperconnections"
    if "routed_experts" in key or ".experts." in joined:
        return "packed_routed_moe_experts"
    if "shared_expert" in key:
        return "shared_swiglu_expert"
    if "router" in key or "router" in joined:
        return "moe_router"
    if ".ple" in key or "ngram" in joined:
        return "ngram_ple"
    if "visual" in key or "vision" in key:
        return "vision_encoder"
    if "embedding" in key or "lm_head" in key:
        return "token_embedding_or_output"
    if "norm" in joined or "layernorm" in joined:
        return "normalization"
    return "other"


def _product(shape: list[int]) -> int:
    return int(math.prod(shape)) if shape else 0


def _geometry(family: str, tensors: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    hidden = cfg.get("hidden_size")
    intermediate = cfg.get("shared_expert_intermediate_size") or cfg.get("moe_intermediate_size")
    input_width = None
    output_width = None
    state_dimensions: dict[str, Any] = {}
    for tensor in tensors:
        name = tensor["tensor_name"].lower()
        shape = tensor["shape"]
        if not shape:
            continue
        if name.endswith(".weight") and len(shape) >= 2:
            if "gate_proj" in name or "up_proj" in name:
                output_width = output_width or shape[-2]
                input_width = input_width or shape[-1]
            elif "down_proj" in name or "out_proj" in name:
                output_width = output_width or shape[-2]
                input_width = input_width or shape[-1]
            elif "in_proj" in name:
                output_width = output_width or shape[-2]
                input_width = input_width or shape[-1]
            else:
                # Router, hyperconnection, and other affine tensors retain
                # the ordinary [out, in] Linear convention even when their
                # leaf name is not one of the projection families above.
                output_width = output_width or shape[-2]
                input_width = input_width or shape[-1]
        if "gate_up_proj" in name and len(shape) == 3:
            state_dimensions["packed_experts"] = shape[0]
            state_dimensions["packed_gate_up_width"] = shape[1]
            input_width = input_width or shape[2]
            output_width = output_width or hidden
        if "down_proj" in name and len(shape) == 3:
            state_dimensions["packed_down_shape"] = shape
        if "conv1d.weight" in name:
            state_dimensions["convolution_shape"] = shape
        if "A_log" in tensor["tensor_name"] or "dt_bias" in tensor["tensor_name"]:
            state_dimensions["state_parameter_shape"] = shape
    if family == "shared_swiglu_expert":
        state_dimensions.update({"intermediate_size": intermediate, "activation": cfg.get("hidden_act", "unknown")})
        input_width = hidden if hidden is not None else input_width
        output_width = hidden if hidden is not None else output_width
    elif family == "gated_deltanet_linear_attention":
        state_dimensions.update(
            {
                "key_heads": cfg.get("linear_num_key_heads"),
                "value_heads": cfg.get("linear_num_value_heads"),
                "key_head_dim": cfg.get("linear_key_head_dim"),
                "value_head_dim": cfg.get("linear_value_head_dim"),
                "state_layout": "value_heads x value_head_dim x key_head_dim (inferred from config and implementation family)",
            }
        )
        input_width = hidden if hidden is not None else input_width
        output_width = hidden if hidden is not None else output_width
    elif family == "gated_residual_hyperconnections":
        state_dimensions.update({"hc_count": cfg.get("hc_count"), "hc_lowrank": cfg.get("hc_lowrank")})
    return {"input_width": input_width, "output_width": output_width, "state_dimensions": state_dimensions}


def _assumptions(family: str, component_key: str, tensors: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    names = [item["tensor_name"] for item in tensors]
    nearby_norms = [name for name in names if "norm" in name.lower()]
    if family == "shared_swiglu_expert":
        return {
            "normalization": "caller supplies the block hidden state; no internal normalization in the shared MLP; scalar sigmoid gate is included when shared_expert_gate.weight is present",
            "residual": "output is added to the MoE branch by the donor decoder block",
            "state": "stateless",
            "nearby_normalization_tensors": nearby_norms,
            "activation": cfg.get("hidden_act", "silu"),
        }
    if family == "gated_deltanet_linear_attention":
        return {
            "normalization": "learned value-head normalization tensor is internal; surrounding block supplies hidden-state normalization and rotary/causal context",
            "residual": "output is consumed by the decoder residual/hyperconnection path",
            "state": "recurrent finite state owned by the linear-attention organ",
            "nearby_normalization_tensors": nearby_norms,
        }
    if family == "packed_routed_moe_experts":
        return {
            "normalization": "caller supplies hidden states; routing weights and top-k indices are external inputs",
            "residual": "weighted expert outputs are summed into the MoE residual branch",
            "state": "stateless expert weights; router-owned dispatch state is per-token",
            "nearby_normalization_tensors": nearby_norms,
        }
    if family == "gated_residual_hyperconnections":
        return {
            "normalization": f"grouped RMS normalization is internal ({cfg.get('rms_norm_eps', 'unknown')} eps where declared)",
            "residual": "mixes and injects multiple residual streams",
            "state": "stateless",
            "nearby_normalization_tensors": nearby_norms,
        }
    return {
        "normalization": "not established from headers",
        "residual": "requires source-level or integration-level verification",
        "state": "not established from headers",
        "nearby_normalization_tensors": nearby_norms,
    }


def _difficulty_value(family: str, payload_bytes: int, tensor_count: int) -> tuple[float, float, str]:
    difficulty = {
        "shared_swiglu_expert": 0.35,
        "moe_router": 0.45,
        "gated_residual_hyperconnections": 0.55,
        "gated_deltanet_linear_attention": 0.78,
        "packed_routed_moe_experts": 0.92,
        "ngram_ple": 0.98,
        "vision_encoder": 0.95,
        "token_embedding_or_output": 0.9,
    }.get(family, 0.65)
    value = {
        "shared_swiglu_expert": 0.72,
        "moe_router": 0.62,
        "gated_residual_hyperconnections": 0.76,
        "gated_deltanet_linear_attention": 0.93,
        "packed_routed_moe_experts": 0.88,
        "ngram_ple": 0.68,
        "vision_encoder": 0.45,
        "token_embedding_or_output": 0.4,
    }.get(family, 0.3)
    cost = max(payload_bytes / (1024 * 1024), 1.0)
    score = value * (1.0 - 0.35 * difficulty) / (1.0 + math.log10(cost))
    if tensor_count <= 5:
        score *= 1.15
    return difficulty, value, f"header-only heuristic score={score:.6f}; payload utility not yet measured"


def _compatibility(geometry: dict[str, Any], target_config: dict[str, Any] | None, family: str) -> dict[str, Any]:
    target = target_config or {}
    target_widths = {"bus_dim": target.get("bus_dim"), "d_model": target.get("d_model")}
    donor_input = geometry.get("input_width")
    donor_output = geometry.get("output_width")
    hard_mismatches = []
    if donor_input is not None and target.get("bus_dim") is not None and donor_input != target["bus_dim"]:
        hard_mismatches.append({"field": "input_width_vs_bus_dim", "donor": donor_input, "target": target["bus_dim"]})
    if donor_output is not None and target.get("bus_dim") is not None and donor_output != target["bus_dim"]:
        hard_mismatches.append({"field": "output_width_vs_bus_dim", "donor": donor_output, "target": target["bus_dim"]})
    if donor_input is None or donor_output is None:
        status = "GEOMETRY_UNRESOLVED"
        modes = ["source_level_verification", "wrapped_graft", "mechanism_reconstruction"]
    else:
        status = "DIRECT_SHAPE_COMPATIBLE" if not hard_mismatches else "WIDTH_MISMATCH_REQUIRES_CONVERSION"
        modes = ["direct_graft"] if not hard_mismatches else ["reparameterize", "wrapped_graft", "subspace_graft", "mechanism_reconstruction"]
    return {
        "status": status,
        "target_widths": target_widths,
        "donor_widths": {"input_width": donor_input, "output_width": donor_output},
        "hard_mismatches": hard_mismatches,
        "candidate_import_modes": modes,
        "family_note": "shared expert is a stateless finite-boundary organ" if family == "shared_swiglu_expert" else "family-specific source/state verification required",
    }


def build_anatomy_graph(manifest: dict[str, Any], target_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the machine-readable donor anatomy graph without reading payloads."""

    inventory = _inventory(manifest)
    config = _config_view(manifest.get("config"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for tensor in inventory:
        grouped[_component_key(tensor["tensor_name"])].append(tensor)

    components: list[dict[str, Any]] = []
    for key in sorted(grouped):
        tensors = sorted(grouped[key], key=lambda item: item["tensor_name"])
        family = _family(key, [item["tensor_name"] for item in tensors])
        geometry = _geometry(family, tensors, config)
        difficulty, value, score_note = _difficulty_value(family, sum(item["payload_bytes"] for item in tensors), len(tensors))
        layer = _layer_index(key)
        dependencies: list[dict[str, Any]] = []
        if layer is not None:
            prefix = key.rsplit(".", 1)[0]
            dependencies.append({"component": f"{prefix}.hidden_state", "evidence": "interface inference from layer contract"})
            if family == "shared_swiglu_expert":
                dependencies.append({"component": f"{prefix}.mlp_hyper_connection", "evidence": "donor decoder residual path; source-level contract"})
            if family == "packed_routed_moe_experts":
                dependencies.append({"component": f"{prefix}.mlp.router", "evidence": "top-k indices and weights are router outputs"})
        components.append(
            {
                "component_id": (
                    f"qwen3.8.{_namespace(key)}.layer{layer}.{key.rsplit('.', 1)[-1]}"
                    if layer is not None
                    else f"qwen3.8.{key.replace('.', '_')}"
                ),
                "component_key": key,
                "namespace": _namespace(key),
                "layer": layer,
                "architecture_family": family,
                "tensor_names": [item["tensor_name"] for item in tensors],
                "tensors": tensors,
                "tensor_count": len(tensors),
                "parameter_count": sum(_product(item["shape"]) for item in tensors),
                "payload_bytes": sum(item["payload_bytes"] for item in tensors),
                "source_shards": sorted({item["source_shard"] for item in tensors}),
                "geometry": geometry,
                "dependencies": dependencies,
                "interface_assumptions": _assumptions(family, key, tensors, config),
                "state_ownership": _assumptions(family, key, tensors, config)["state"],
                "compatibility": _compatibility(geometry, target_config, family),
                "estimated_graft_difficulty": difficulty,
                "estimated_value": value,
                "ranking_note": score_note,
                "actual_payload_inspected": False,
                "payload_statistics": None,
                "promotion_state": "DONOR_OBSERVED",
            }
        )

    ranked = sorted(
        [
            item
            for item in components
            if item["architecture_family"]
            not in {"other", "normalization", "token_embedding_or_output", "vision_encoder"}
        ],
        key=lambda item: (
            -float(item["estimated_value"] * (1.0 - 0.35 * item["estimated_graft_difficulty"]))
            / (1.0 + math.log10(max(item["payload_bytes"] / (1024 * 1024), 1.0))),
            item["payload_bytes"],
            item["component_id"],
        ),
    )
    # Repeated layers are instances of one boundary, not independent design
    # candidates. Keep all instances in ``components`` but rank one
    # representative per structural signature.
    unique_ranked: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    for item in ranked:
        signature = _structural_signature(item["component_key"])
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        instances = [
            candidate
            for candidate in components
            if _structural_signature(candidate["component_key"]) == signature
        ]
        representative = dict(item)
        representative["structural_signature"] = signature
        representative["instance_count"] = len(instances)
        representative["representative_layers"] = sorted(
            {candidate["layer"] for candidate in instances if candidate["layer"] is not None}
        )
        unique_ranked.append(representative)

    preferred_first_families = {
        "shared_swiglu_expert",
        "gated_deltanet_linear_attention",
        "gated_residual_hyperconnections",
        "packed_routed_moe_experts",
    }
    first_candidates = [
        item
        for item in components
        if item["architecture_family"] in preferred_first_families
        and item["namespace"] == "language"
        and item["payload_bytes"] <= 256 * 1024 * 1024
        and item["tensor_count"] >= 3
        and item["geometry"]["input_width"] is not None
        and item["geometry"]["output_width"] is not None
    ]
    first_candidates.sort(
        key=lambda item: (
            0 if item["architecture_family"] == "shared_swiglu_expert" else 1,
            item["layer"] if item["layer"] is not None else 10**9,
            item["payload_bytes"],
            item["component_id"],
        )
    )
    first_target = first_candidates[0] if first_candidates else None
    recommendations = [
        {
            "rank": index,
            "component_id": item["component_id"],
            "component_key": item["component_key"],
            "family": item["architecture_family"],
            "payload_bytes": item["payload_bytes"],
            "instance_count": item.get("instance_count", 1),
            "representative_layers": item.get("representative_layers", []),
            "estimated_value": item["estimated_value"],
            "estimated_graft_difficulty": item["estimated_graft_difficulty"],
            "recommended_first_mode": item["compatibility"]["candidate_import_modes"][0],
            "reason": "clear bounded boundary and finite interface; inspect payload before promotion" if item["architecture_family"] == "shared_swiglu_expert" else "candidate ranked by value, boundary clarity, state semantics, and extraction cost",
        }
        for index, item in enumerate(unique_ranked[:24], start=1)
    ]
    source = manifest.get("source", {})
    return {
        "schema": "remora-qwen-neural-anatomy-v1",
        "source": {
            "repository": source.get("repository"),
            "revision": source.get("revision"),
            "path": manifest.get("path"),
            "config_architectures": manifest.get("config", {}).get("architectures", []),
            "config_model_type": manifest.get("config", {}).get("model_type"),
        },
        "inspection": {
            "mode": "config_index_and_safetensors_headers",
            "weights_materialized": False,
            "model_loader_called": False,
            "payload_statistics_present": False,
            "parsed_tensor_count": len(inventory),
            "index_header_reconciliation": manifest.get("headers", {}).get("index_name_reconciliation", manifest.get("verification", {})),
        },
        "config_summary": {
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "layer_types": config.get("layer_types"),
            "linear_num_key_heads": config.get("linear_num_key_heads"),
            "linear_num_value_heads": config.get("linear_num_value_heads"),
            "linear_key_head_dim": config.get("linear_key_head_dim"),
            "linear_value_head_dim": config.get("linear_value_head_dim"),
            "num_experts": config.get("num_experts"),
            "num_experts_per_tok": config.get("num_experts_per_tok"),
            "moe_intermediate_size": config.get("moe_intermediate_size"),
            "shared_expert_intermediate_size": config.get("shared_expert_intermediate_size"),
            "hc_count": config.get("hc_count"),
            "hc_lowrank": config.get("hc_lowrank"),
            "hidden_act": config.get("hidden_act"),
            "rms_norm_eps": config.get("rms_norm_eps"),
        },
        "component_count": len(components),
        "components": components,
        "candidate_ranking": recommendations,
        "initial_target_selection": (
            {
                "component_id": first_target["component_id"],
                "component_key": first_target["component_key"],
                "architecture_family": first_target["architecture_family"],
                "selection_basis": "bounded meaningful organ with a clear affine/SwiGLU boundary; preferred before stateful high-cost candidates",
                "payload_bytes": first_target["payload_bytes"],
                "tensor_names": first_target["tensor_names"],
                "mode": "submodule_graft_then_wrapped_graft_if_width_conversion_is_required",
                "status": "VALUE_FREE_TARGET_SELECTED",
            }
            if first_target is not None
            else None
        ),
        "ranking_status": "ESTIMATED_HEADER_ONLY; actual trained payload analysis is a separate gated experiment",
    }


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def manifest_digest(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def write_anatomy_graph(path: str | Path, graph: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n")
