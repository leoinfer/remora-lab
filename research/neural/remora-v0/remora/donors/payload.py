from __future__ import annotations

"""Bounded analysis and standalone execution of extracted donor payloads."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_stats(tensor: torch.Tensor, *, calculate_spectrum: bool = True) -> dict[str, Any]:
    value = tensor.detach().cpu().float()
    flat = value.reshape(-1)
    result: dict[str, Any] = {
        "shape": [int(dimension) for dimension in tensor.shape],
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "payload_bytes": int(tensor.numel() * tensor.element_size()),
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "l2_norm": float(torch.linalg.vector_norm(flat)),
        "mean_abs": float(flat.abs().mean()),
        "zero_fraction": float((flat == 0).float().mean()),
        "p01": float(torch.quantile(flat, 0.01)),
        "p50": float(torch.quantile(flat, 0.50)),
        "p99": float(torch.quantile(flat, 0.99)),
    }
    if calculate_spectrum and value.ndim == 2 and min(value.shape) > 1:
        singular = torch.linalg.svdvals(value)
        energy = singular.square()
        cumulative = torch.cumsum(energy, dim=0) / energy.sum().clamp_min(1e-30)
        result["spectrum"] = {
            "top_8_singular_values": [float(item) for item in singular[:8]],
            "largest_to_smallest_ratio": float(singular[0] / singular[-1].clamp_min(1e-30)),
            "stable_rank": float(energy.sum() / singular[0].square().clamp_min(1e-30)),
            "rank_at_90pct_energy": int(torch.searchsorted(cumulative, torch.tensor(0.90)).item() + 1),
            "rank_at_99pct_energy": int(torch.searchsorted(cumulative, torch.tensor(0.99)).item() + 1),
            "rank_at_999pct_energy": int(torch.searchsorted(cumulative, torch.tensor(0.999)).item() + 1),
        }
    return result


def load_payload(path: str | Path, *, max_payload_bytes: int = 256 * 1024 * 1024) -> dict[str, torch.Tensor]:
    """Load only an extracted, bounded safetensors bundle."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > int(max_payload_bytes):
        raise MemoryError(f"payload file exceeds bounded load budget: {path.stat().st_size}")
    from safetensors import safe_open

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for name in sorted(handle.keys()):
            tensors[name] = handle.get_tensor(name).contiguous()
    materialized = sum(int(tensor.numel() * tensor.element_size()) for tensor in tensors.values())
    if materialized > int(max_payload_bytes):
        raise MemoryError(f"materialized payload exceeds bounded load budget: {materialized}")
    return tensors


def inspect_payload(path: str | Path, *, max_payload_bytes: int = 256 * 1024 * 1024) -> dict[str, Any]:
    tensors = load_payload(path, max_payload_bytes=max_payload_bytes)
    source_path = Path(path).expanduser().resolve()
    materialized = sum(int(tensor.numel() * tensor.element_size()) for tensor in tensors.values())
    return {
        "schema": "remora-qwen-neural-payload-analysis-v1",
        "payload_path": str(source_path),
        "payload_file_bytes": source_path.stat().st_size,
        "payload_sha256": _sha256(source_path),
        "materialized_payload_bytes": materialized,
        "max_payload_bytes": int(max_payload_bytes),
        "tensor_count": len(tensors),
        "weights_materialized": True,
        "model_loader_called": False,
        "tensors": {name: _tensor_stats(tensor) for name, tensor in tensors.items()},
        "interpretation": "MEASURED: statistics and spectra were computed from the selected trained Qwen payload only. These observations do not establish Remora utility.",
    }


class QwenSharedExpertOrgan(nn.Module):
    """Standalone Qwen shared expert with the actual extracted weights."""

    interface_version = "qwen-shared-expert-organ-v1"

    def __init__(self, tensors: dict[str, torch.Tensor], *, include_scalar_gate: bool = True):
        super().__init__()
        required_suffixes = {
            "gate_proj": "shared_expert.gate_proj.weight",
            "up_proj": "shared_expert.up_proj.weight",
            "down_proj": "shared_expert.down_proj.weight",
        }
        selected: dict[str, torch.Tensor] = {}
        for short, suffix in required_suffixes.items():
            names = [name for name in tensors if name.endswith(suffix)]
            if len(names) != 1:
                raise KeyError(f"expected one {suffix}, found {names}")
            selected[short] = tensors[names[0]].float().contiguous()
        self.register_buffer("gate_proj", selected["gate_proj"])
        self.register_buffer("up_proj", selected["up_proj"])
        self.register_buffer("down_proj", selected["down_proj"])
        self.include_scalar_gate = bool(include_scalar_gate)
        if include_scalar_gate:
            names = [name for name in tensors if name.endswith("shared_expert_gate.weight")]
            if len(names) != 1:
                raise KeyError(f"expected one shared_expert_gate.weight, found {names}")
            self.register_buffer("shared_expert_gate", tensors[names[0]].float().contiguous())
        self.donor_parameter_count = sum(int(buffer.numel()) for buffer in self.buffers())

    def forward(self, hidden: torch.Tensor, *, apply_scalar_gate: bool | None = None) -> torch.Tensor:
        if hidden.shape[-1] != self.gate_proj.shape[-1]:
            raise ValueError(f"expected hidden width {self.gate_proj.shape[-1]}, got {hidden.shape[-1]}")
        gate_pre = torch.matmul(hidden.float(), self.gate_proj.t())
        up_pre = torch.matmul(hidden.float(), self.up_proj.t())
        core = torch.matmul(torch.nn.functional.silu(gate_pre) * up_pre, self.down_proj.t())
        use_gate = self.include_scalar_gate if apply_scalar_gate is None else bool(apply_scalar_gate)
        if use_gate:
            scale = torch.sigmoid(torch.matmul(hidden.float(), self.shared_expert_gate.t()))
            core = core * scale
        return core


def reference_shared_expert(
    hidden: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    *,
    apply_scalar_gate: bool = True,
) -> torch.Tensor:
    """Independent reference path using explicit matrix products."""

    by_suffix: dict[str, torch.Tensor] = {}
    suffixes = {
        "gate": "shared_expert.gate_proj.weight",
        "up": "shared_expert.up_proj.weight",
        "down": "shared_expert.down_proj.weight",
        "scalar": "shared_expert_gate.weight",
    }
    for key, suffix in suffixes.items():
        names = [name for name in tensors if name.endswith(suffix)]
        if key == "scalar" and not names and not apply_scalar_gate:
            continue
        if len(names) != 1:
            raise KeyError(f"expected one {suffix}, found {names}")
        by_suffix[key] = tensors[names[0]].float()
    hidden = hidden.float()
    gate = torch.nn.functional.silu(hidden @ by_suffix["gate"].transpose(-1, -2))
    up = hidden @ by_suffix["up"].transpose(-1, -2)
    output = (gate * up) @ by_suffix["down"].transpose(-1, -2)
    if apply_scalar_gate:
        scalar = torch.sigmoid(hidden @ by_suffix["scalar"].transpose(-1, -2))
        output = output * scalar
    return output


def functional_equivalence(
    tensors: dict[str, torch.Tensor], *, seed: int = 1701, batch: int = 3, time: int = 5
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    gate_name = next(name for name in tensors if name.endswith("shared_expert.gate_proj.weight"))
    input_width = int(tensors[gate_name].shape[-1])
    hidden = torch.randn(batch, time, input_width, generator=generator) * 0.5
    organ = QwenSharedExpertOrgan(tensors)
    converted = organ(hidden)
    reference = reference_shared_expert(hidden, tensors)
    delta = converted - reference
    flat_a = converted.reshape(-1)
    flat_b = reference.reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(flat_a[None, :], flat_b[None, :]).item()
    relative = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference).clamp_min(1e-30)
    top_k = min(16, int(flat_a.numel()))
    top_a = torch.topk(flat_a.abs(), top_k).indices
    top_b = torch.topk(flat_b.abs(), top_k).indices
    return {
        "schema": "remora-qwen-neural-functional-equivalence-v1",
        "input_shape": [batch, time, input_width],
        "input_seed": seed,
        "apply_scalar_gate": True,
        "max_absolute_error": float(delta.abs().max()),
        "mean_absolute_error": float(delta.abs().mean()),
        "relative_l2_error": float(relative),
        "cosine_similarity": float(cosine),
        "absolute_output_l2": float(torch.linalg.vector_norm(flat_a)),
        "top_abs_coordinate_overlap_at_16": int(len(set(top_a.tolist()) & set(top_b.tolist()))),
        "state_update_error": None,
        "state_update_note": "not applicable: selected shared expert is stateless",
        "status": "EQUIVALENT_WITHIN_FLOAT32_REFERENCE_TOLERANCE" if float(relative) < 1e-6 else "NON_EQUIVALENT_REQUIRES_INVESTIGATION",
        "reference_path": "explicit matmul/SILU/sigmoid implementation independent of the standalone module call path",
    }
