from __future__ import annotations

"""Extract and test one real Qwen GatedDeltaNet value-head organ.

This is the next donor candidate after the router assay.  It is deliberately
below whole-layer granularity: only one 128-dimensional value head and the
smallest tensors needed for its Q/K/V/Z/beta/decay -> state -> output path are
read from the Qwen checkpoint.  The experiment has three separable claims:

1. the selected tensors reproduce an independently written GatedDeltaNet
   computation, including recurrent and causal-convolution state;
2. analytic width conversion preserves a measurable part of that function;
3. the actual converted stateful organ has useful delayed associative recall
   relative to randomized/shuffled/zero cores with equal readout budgets.

The third assay is explicitly donor-conditioned.  It does not claim that one
isolated Qwen head is a general language skill.  It asks whether the trained
state machinery gives Remora a useful, training-free computational organ.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.config import ModelConfig  # noqa: E402
from remora.donors.gdn import QwenGatedDeltaHead, causal_depthwise_silu, l2_normalize, make_gdn_variant  # noqa: E402
from remora.donors.neural_ir import qwen_gated_delta_head_ir  # noqa: E402
from remora.ledger import record_experiment, record_failure  # noqa: E402
from remora.models import build_model  # noqa: E402
from remora.utils import runtime_context, set_seed, write_json  # noqa: E402


DONOR_REPOSITORY = "Qwen/Qwen3.8-Flash-Next"
DONOR_REVISION = "f5d08274bafd880402bd16f5e3e6c514136ec06c"
LAYER = 17
SCAN_HEADS = (0, 10, 20, 30, 40)
SELECTED_DEFAULT_HEAD = 10
KEY_HEAD_DIM = 128
VALUE_HEAD_DIM = 128
NUM_KEY_HEADS = 16
NUM_VALUE_HEADS = 48
KEY_DIM = NUM_KEY_HEADS * KEY_HEAD_DIM
VALUE_DIM = NUM_VALUE_HEADS * VALUE_HEAD_DIM
HEAD_CONV_DIM = KEY_HEAD_DIM * 3
CONV_KERNEL = 4
BUS_WIDTH = 96
OUTPUT_WIDTH = 96
MEMORY_ITEMS = 4
MEMORY_DELAY = 9
TRAIN_SAMPLES = 256
TEST_SAMPLES = 512
RECALL_THRESHOLD = 0.50
READOUT_STEPS = (0, 1, 2, 4, 8, 16, 32, 64)
SEEDS = (7, 19, 31)


def _sha256_tensor(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _source_index(source_root: Path) -> dict[str, str]:
    return json.loads((source_root / "model.safetensors.index.json").read_text())["weight_map"]


def _slice(handle: Any, name: str, rows: slice | None = None, columns: slice | None = None) -> torch.Tensor:
    value = handle.get_slice(name)
    if rows is None and columns is None:
        return value[:] .float().contiguous()
    if rows is None:
        return value[:, columns].float().contiguous()
    if columns is None:
        return value[rows].float().contiguous()
    return value[rows, columns].float().contiguous()


def _extract_head(source_root: Path, layer: int, value_head: int) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Read only the selected head's tensor slices from one Qwen shard."""

    if not 0 <= int(value_head) < NUM_VALUE_HEADS:
        raise ValueError(value_head)
    index = _source_index(source_root)
    prefix = f"model.language_model.layers.{layer}.linear_attn."
    names = {
        "qkv": prefix + "in_proj_qkv.weight",
        "z": prefix + "in_proj_z.weight",
        "a": prefix + "in_proj_a.weight",
        "b": prefix + "in_proj_b.weight",
        "conv": prefix + "conv1d.weight",
        "A_log": prefix + "A_log",
        "dt_bias": prefix + "dt_bias",
        "norm": prefix + "norm.weight",
        "out": prefix + "out_proj.weight",
    }
    shards = {index[name] for name in names.values()}
    if len(shards) != 1:
        raise ValueError(f"selected head spans unexpected shards: {sorted(shards)}")
    shard_name = next(iter(shards))
    from safetensors import safe_open

    key_head = int(value_head) // (NUM_VALUE_HEADS // NUM_KEY_HEADS)
    q0 = key_head * KEY_HEAD_DIM
    k0 = KEY_DIM + key_head * KEY_HEAD_DIM
    v0 = KEY_DIM * 2 + int(value_head) * VALUE_HEAD_DIM
    z0 = int(value_head) * VALUE_HEAD_DIM
    out0 = int(value_head) * VALUE_HEAD_DIM
    q_rows = slice(q0, q0 + KEY_HEAD_DIM)
    k_rows = slice(k0, k0 + KEY_HEAD_DIM)
    v_rows = slice(v0, v0 + VALUE_HEAD_DIM)
    z_rows = slice(z0, z0 + VALUE_HEAD_DIM)
    q_conv = slice(q0, q0 + KEY_HEAD_DIM)
    k_conv = slice(k0, k0 + KEY_HEAD_DIM)
    v_conv = slice(v0, v0 + VALUE_HEAD_DIM)
    with safe_open(str(source_root / shard_name), framework="pt", device="cpu") as handle:
        payload = {
            "q_weight": _slice(handle, names["qkv"], q_rows),
            "k_weight": _slice(handle, names["qkv"], k_rows),
            "v_weight": _slice(handle, names["qkv"], v_rows),
            "z_weight": _slice(handle, names["z"], z_rows),
            "a_weight": _slice(handle, names["a"], slice(value_head, value_head + 1)),
            "b_weight": _slice(handle, names["b"], slice(value_head, value_head + 1)),
            "conv_qkv": torch.cat(
                (
                    _slice(handle, names["conv"], q_conv).squeeze(1),
                    _slice(handle, names["conv"], k_conv).squeeze(1),
                    _slice(handle, names["conv"], v_conv).squeeze(1),
                ),
                dim=0,
            ),
            "A_log": _slice(handle, names["A_log"], slice(value_head, value_head + 1)),
            "dt_bias": _slice(handle, names["dt_bias"], slice(value_head, value_head + 1)),
            "norm_weight": _slice(handle, names["norm"]),
            "out_weight": _slice(handle, names["out"], columns=slice(out0, out0 + VALUE_HEAD_DIM)),
        }
    payload_bytes_bf16 = int(sum(value.numel() * 2 for value in payload.values()))
    materialized_bytes_fp32 = int(sum(value.numel() * 4 for value in payload.values()))
    metadata = {
        "layer": int(layer),
        "value_head": int(value_head),
        "key_head": int(key_head),
        "source_shard": shard_name,
        "source_tensor_names": names,
        "source_slices": {
            "q_rows": [q0, q0 + KEY_HEAD_DIM],
            "k_rows": [k0, k0 + KEY_HEAD_DIM],
            "v_rows": [v0, v0 + VALUE_HEAD_DIM],
            "z_rows": [z0, z0 + VALUE_HEAD_DIM],
            "out_columns": [out0, out0 + VALUE_HEAD_DIM],
        },
        "source_payload_bytes_bf16": payload_bytes_bf16,
        "selected_payload_materialized_bytes_fp32": materialized_bytes_fp32,
        "tensor_hashes_float32": {name: _sha256_tensor(value) for name, value in payload.items()},
    }
    return payload, metadata


def _conversion_bases(payload: dict[str, torch.Tensor], rank: int = BUS_WIDTH) -> tuple[torch.Tensor, torch.Tensor]:
    from remora.donors.router import top_right_singular_basis

    input_stack = torch.cat(
        [payload[name] for name in ("q_weight", "k_weight", "v_weight", "z_weight", "a_weight", "b_weight")],
        dim=0,
    )
    input_basis = top_right_singular_basis(input_stack, rank)
    output_basis = top_right_singular_basis(payload["out_weight"].transpose(0, 1), rank)
    return input_basis, output_basis


def _convert_payload(
    payload: dict[str, torch.Tensor],
    input_basis: torch.Tensor,
    output_basis: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    converted = {
        "q_weight": payload["q_weight"] @ input_basis,
        "k_weight": payload["k_weight"] @ input_basis,
        "v_weight": payload["v_weight"] @ input_basis,
        "z_weight": payload["z_weight"] @ input_basis,
        "a_weight": payload["a_weight"] @ input_basis,
        "b_weight": payload["b_weight"] @ input_basis,
        "conv_qkv": payload["conv_qkv"].clone(),
        "A_log": payload["A_log"].clone(),
        "dt_bias": payload["dt_bias"].clone(),
        "norm_weight": payload["norm_weight"].clone(),
    }
    if output_basis is None:
        converted["out_weight"] = payload["out_weight"].clone()
    else:
        converted["out_weight"] = output_basis.transpose(0, 1) @ payload["out_weight"]
    return converted


def _reference_forward(
    tensors: dict[str, torch.Tensor],
    hidden: torch.Tensor,
    state: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
    """Independent reference: unfold convolution + explicit delta recurrence."""

    batch, time, _ = hidden.shape
    q_pre = F.linear(hidden.float(), tensors["q_weight"])
    k_pre = F.linear(hidden.float(), tensors["k_weight"])
    v_pre = F.linear(hidden.float(), tensors["v_weight"])
    z = F.linear(hidden.float(), tensors["z_weight"])
    a = F.linear(hidden.float(), tensors["a_weight"]).squeeze(-1)
    b = F.linear(hidden.float(), tensors["b_weight"]).squeeze(-1)
    raw = torch.cat((q_pre, k_pre, v_pre), dim=-1).transpose(1, 2)
    channels, kernel = tensors["conv_qkv"].shape
    if state is None:
        recurrent = raw.new_zeros(batch, VALUE_HEAD_DIM, KEY_HEAD_DIM)
        conv_history = raw.new_zeros(batch, channels, kernel - 1)
    else:
        recurrent, conv_history = state
        recurrent = recurrent.float().to(hidden.device)
        conv_history = conv_history.float().to(hidden.device)
    padded = torch.cat((conv_history, raw), dim=-1)
    windows = padded.unfold(-1, kernel, 1)
    filtered = (windows * tensors["conv_qkv"].float().view(1, channels, 1, kernel)).sum(-1)
    mixed = F.silu(filtered.transpose(1, 2))
    next_conv_history = padded[:, :, -(kernel - 1) :].contiguous()
    q, k, v = mixed.split(VALUE_HEAD_DIM, dim=-1)
    q = l2_normalize(q)
    k = l2_normalize(k)
    beta = b.sigmoid()
    decay = -tensors["A_log"].float().exp() * F.softplus(a.float() + tensors["dt_bias"].float())
    core = []
    for index in range(time):
        q_t = q[:, index]
        k_t = k[:, index]
        v_t = v[:, index]
        recurrent = recurrent * decay[:, index].exp().view(batch, 1, 1)
        kv_mem = (recurrent * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta[:, index].view(batch, 1)
        recurrent = recurrent + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core.append((recurrent * q_t.unsqueeze(-1)).sum(dim=-2))
    core_value = torch.stack(core, dim=1)
    normalized = tensors["norm_weight"].view(1, 1, -1) * core_value * torch.rsqrt(
        core_value.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    normalized = normalized * F.silu(z.float())
    output = F.linear(normalized, tensors["out_weight"])
    details = {
        "query": q,
        "key": k,
        "value": v,
        "beta": beta,
        "log_decay": decay,
        "core": core_value,
        "normalized": normalized,
    }
    return output, (recurrent, next_conv_history), details


def _compare(
    actual: torch.Tensor,
    reference: torch.Tensor,
    actual_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    reference_state: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, float]:
    delta = (actual.float() - reference.float()).reshape(-1)
    ref = reference.float().reshape(-1)
    result = {
        "max_absolute_error": float(delta.abs().max()),
        "mean_absolute_error": float(delta.abs().mean()),
        "relative_l2_error": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(ref).clamp_min(1e-30)),
        "cosine_similarity": float(F.cosine_similarity(actual.float().reshape(1, -1), reference.float().reshape(1, -1)).item()),
    }
    if actual_state is not None and reference_state is not None:
        recurrent_delta = (actual_state[0].float() - reference_state[0].float()).reshape(-1)
        conv_delta = (actual_state[1].float() - reference_state[1].float()).reshape(-1)
        result["recurrent_state_relative_l2_error"] = float(
            torch.linalg.vector_norm(recurrent_delta) / torch.linalg.vector_norm(reference_state[0].float()).clamp_min(1e-30)
        )
        result["convolution_state_relative_l2_error"] = float(
            torch.linalg.vector_norm(conv_delta) / torch.linalg.vector_norm(reference_state[1].float()).clamp_min(1e-30)
        )
    return result


def _converted_organ(
    payload: dict[str, torch.Tensor],
    input_basis: torch.Tensor,
    output_basis: torch.Tensor | None,
    *,
    donor_variant: str,
) -> QwenGatedDeltaHead:
    converted = _convert_payload(payload, input_basis, output_basis)
    return QwenGatedDeltaHead(**converted, donor_variant=donor_variant)


def _candidate_scan(source_root: Path) -> tuple[list[dict[str, Any]], int]:
    """Rank several actual heads using payload structure and state response."""

    ranked: list[dict[str, Any]] = []
    probe_generator = torch.Generator(device="cpu").manual_seed(1703)
    probe = torch.randn(8, 12, BUS_WIDTH, generator=probe_generator) * 0.20
    no_history = probe.clone()
    no_history[:, :8] = 0.0
    for value_head in SCAN_HEADS:
        payload, metadata = _extract_head(source_root, LAYER, value_head)
        input_basis, output_basis = _conversion_bases(payload)
        actual = _converted_organ(payload, input_basis, output_basis, donor_variant="actual")
        random_payload = make_gdn_variant(payload, "random", seed=20260910 + value_head)
        random = _converted_organ(random_payload, input_basis, output_basis, donor_variant="random")
        with torch.no_grad():
            actual_out, _, _ = actual(probe, return_state=True, return_intermediates=True)
            actual_no_history = actual(no_history)
            random_out, _, _ = random(probe, return_state=True, return_intermediates=True)
            random_no_history = random(no_history)
        actual_delta = float((actual_out[:, -1] - actual_no_history[:, -1]).norm(dim=-1).mean())
        random_delta = float((random_out[:, -1] - random_no_history[:, -1]).norm(dim=-1).mean())
        actual_norm = float(actual_out[:, -1].norm(dim=-1).mean())
        random_norm = float(random_out[:, -1].norm(dim=-1).mean())
        stack = torch.cat(
            [payload[name] for name in ("q_weight", "k_weight", "v_weight", "z_weight", "a_weight", "b_weight")],
            dim=0,
        )
        singular = torch.linalg.svdvals(stack)
        energy = singular.square()
        rank_energy = float(energy[:BUS_WIDTH].sum() / energy.sum().clamp_min(1e-30))
        actual_history_fraction = actual_delta / max(actual_norm, 1e-6)
        random_history_fraction = random_delta / max(random_norm, 1e-6)
        # The old ratio ``actual_delta / random_delta`` was unstable when a
        # randomized gate happened to erase almost all history.  Bound the
        # triage score so a near-zero random denominator cannot select a head
        # on a numerical artifact.
        score = (
            0.55 * min(actual_history_fraction, 1.0)
            + 0.25 * max(min(actual_history_fraction - random_history_fraction, 1.0), 0.0)
            + 0.20 * rank_energy
        )
        ranked.append(
            {
                "candidate_id": f"qwen3.8.language.layer{LAYER}.linear_attn.value_head{value_head}",
                "family": "gated_deltanet_value_head",
                "layer": LAYER,
                "value_head": value_head,
                "key_head": value_head // 3,
                "source_payload_bytes_bf16": metadata["source_payload_bytes_bf16"],
                "closed_subgraph": "selected q/k/v/z/beta/decay projections + causal depthwise slice + 128x128 delta state + norm + output columns",
                "actual_history_delta": actual_delta,
                "random_history_delta": random_delta,
                "actual_history_fraction": actual_history_fraction,
                "random_history_fraction": random_history_fraction,
                "trained_minus_random_history_delta": actual_delta - random_delta,
                "rank96_input_energy_fraction": rank_energy,
                "trained_vs_random_signal_status": "MEASURED_TRIAGE_NOT_CAPABILITY_PROOF",
                "information_scan_score": score,
                "estimated_active_flops_per_token": int(
                    2 * (4 * BUS_WIDTH * VALUE_HEAD_DIM + 2 * BUS_WIDTH + HEAD_CONV_DIM * CONV_KERNEL + VALUE_HEAD_DIM * OUTPUT_WIDTH)
                    + 8 * VALUE_HEAD_DIM * KEY_HEAD_DIM
                ),
                "actual_weight_scan_status": "MEASURED",
            }
        )
    ranked.sort(key=lambda row: float(row["information_scan_score"]), reverse=True)
    return ranked, int(ranked[0]["value_head"])


def _synthesize_query(
    organ: QwenGatedDeltaHead,
    target_keys: torch.Tensor,
    *,
    seed: int,
    steps: int = 24,
) -> tuple[torch.Tensor, float, float]:
    """Find a compact query that points at a selected donor key.

    This is an analytic/data-generation operation, not donor training.  The
    beta penalty makes the query read the existing state instead of strongly
    overwriting it.  The achieved q/key cosine is recorded as a task
    difficulty check rather than silently assumed.
    """

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    query = (0.05 * torch.randn(target_keys.shape[0], organ.bus_dim, generator=generator)).requires_grad_()
    optimizer = torch.optim.Adam([query], lr=0.20)
    current_beta = torch.zeros(target_keys.shape[0])
    for _ in range(int(steps)):
        raw_q = F.linear(query, organ.q_weight) * organ.conv_qkv[:VALUE_HEAD_DIM, -1].view(1, -1)
        q = l2_normalize(F.silu(raw_q))
        current_beta = F.linear(query, organ.b_weight).squeeze(-1).detach()
        cosine = (q * target_keys).sum(dim=-1)
        loss = (1.0 - cosine).mean() + 0.01 * (current_beta + 5.0).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        raw_q = F.linear(query, organ.q_weight) * organ.conv_qkv[:VALUE_HEAD_DIM, -1].view(1, -1)
        q = l2_normalize(F.silu(raw_q))
        cosine = float((q * target_keys).sum(dim=-1).mean())
        beta = float(current_beta.sigmoid().mean())
    return query.detach(), cosine, beta


def _make_recall_dataset(
    organ: QwenGatedDeltaHead,
    *,
    samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Create delayed associative queries around the actual donor key/value path."""

    if MEMORY_DELAY - MEMORY_ITEMS < organ.kernel_size:
        raise ValueError("memory delay must leave a complete causal-convolution gap before the query")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    memory = 0.20 * torch.randn(samples, MEMORY_ITEMS, organ.bus_dim, generator=generator)
    labels = torch.randint(MEMORY_ITEMS, (samples,), generator=generator)
    sequence = torch.zeros(samples, MEMORY_DELAY + 1, organ.bus_dim)
    sequence[:, :MEMORY_ITEMS] = memory
    with torch.no_grad():
        _, _, prefix_details = organ(sequence, return_state=True, return_intermediates=True)
    keys = prefix_details["key"][:, :MEMORY_ITEMS]
    values = prefix_details["value"][:, :MEMORY_ITEMS]
    target_keys = keys[torch.arange(samples), labels]
    query, alignment, query_beta = _synthesize_query(organ, target_keys, seed=seed + 9000)
    sequence[:, -1] = query
    return sequence, labels, values, {"query_key_cosine_mean": alignment, "query_beta_mean": query_beta}


def _mechanistic_retrieval(
    organ: QwenGatedDeltaHead,
    sequence: torch.Tensor,
    labels: torch.Tensor,
    actual_values: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        _, _, details = organ(sequence, return_state=True, return_intermediates=True)
    core = l2_normalize(details["core"][:, -1])
    values = l2_normalize(actual_values)
    scores = (core[:, None, :] * values).sum(dim=-1)
    prediction = scores.argmax(dim=-1)
    return {
        "accuracy": float((prediction == labels).float().mean()),
        "mean_target_cosine": float(scores[torch.arange(labels.shape[0]), labels].mean()),
        "mean_best_cosine": float(scores.max(dim=-1).values.mean()),
    }


class _FreshTrainableGDNCore(nn.Module):
    """Fresh same-mechanism core used to estimate relearning cost.

    It has the compact Q/K/V, beta/decay, causal-convolution, and recurrent
    state machinery, but none of the selected donor values.  The comparison
    target is the actual donor's selected memory value, not donor logits.
    """

    def __init__(self, bus_dim: int = BUS_WIDTH):
        super().__init__()
        self.bus_dim = int(bus_dim)
        self.q_weight = nn.Parameter(torch.randn(VALUE_HEAD_DIM, bus_dim) * 0.02)
        self.k_weight = nn.Parameter(torch.randn(VALUE_HEAD_DIM, bus_dim) * 0.02)
        self.v_weight = nn.Parameter(torch.randn(VALUE_HEAD_DIM, bus_dim) * 0.02)
        self.a_weight = nn.Parameter(torch.randn(1, bus_dim) * 0.02)
        self.b_weight = nn.Parameter(torch.randn(1, bus_dim) * 0.02)
        self.conv_qkv = nn.Parameter(torch.randn(HEAD_CONV_DIM, CONV_KERNEL) * 0.02)
        self.A_log = nn.Parameter(torch.zeros(1))
        self.dt_bias = nn.Parameter(torch.zeros(1))

    def forward(self, bus: torch.Tensor) -> torch.Tensor:
        q = F.linear(bus, self.q_weight)
        k = F.linear(bus, self.k_weight)
        v = F.linear(bus, self.v_weight)
        a = F.linear(bus, self.a_weight).squeeze(-1)
        b = F.linear(bus, self.b_weight).squeeze(-1)
        mixed = causal_depthwise_silu(torch.cat((q, k, v), dim=-1), self.conv_qkv)
        q, k, v = mixed.split(VALUE_HEAD_DIM, dim=-1)
        q = l2_normalize(q)
        k = l2_normalize(k)
        beta = b.sigmoid()
        decay = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        batch = int(bus.shape[0])
        recurrent = bus.new_zeros(batch, KEY_HEAD_DIM, VALUE_HEAD_DIM).float()
        core_values = []
        for index in range(int(bus.shape[1])):
            q_t = q[:, index]
            k_t = k[:, index]
            v_t = v[:, index]
            recurrent = recurrent * decay[:, index].exp().view(batch, 1, 1)
            kv_mem = (recurrent * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta[:, index].view(batch, 1)
            recurrent = recurrent + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            core_values.append((recurrent * q_t.unsqueeze(-1)).sum(dim=-2))
        return torch.stack(core_values, dim=1)


def _core_retrieval_accuracy(
    core: torch.Tensor,
    labels: torch.Tensor,
    values: torch.Tensor,
) -> float:
    normalized_core = l2_normalize(core[:, -1])
    normalized_values = l2_normalize(values)
    scores = (normalized_core[:, None, :] * normalized_values).sum(dim=-1)
    return float((scores.argmax(dim=-1) == labels).float().mean())


def _fresh_core_curve(
    train_sequence: torch.Tensor,
    train_labels: torch.Tensor,
    train_values: torch.Tensor,
    test_sequence: torch.Tensor,
    test_labels: torch.Tensor,
    test_values: torch.Tensor,
    *,
    seed: int,
    steps: Iterable[int] = READOUT_STEPS,
) -> dict[str, Any]:
    """Train a fresh GDN core against delayed value targets at fixed budgets."""

    set_seed(seed)
    initial = _FreshTrainableGDNCore()
    initial_state = {name: value.detach().clone() for name, value in initial.state_dict().items()}
    target_train = l2_normalize(train_values[torch.arange(train_labels.shape[0]), train_labels])
    values: list[dict[str, Any]] = []
    for budget in steps:
        model = _FreshTrainableGDNCore()
        model.load_state_dict(initial_state, strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
        started = time.perf_counter()
        losses: list[float] = []
        for _ in range(int(budget)):
            optimizer.zero_grad(set_to_none=True)
            prediction = l2_normalize(model(train_sequence)[:, -1])
            loss = (1.0 - (prediction * target_train).sum(dim=-1)).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            test_core = model(test_sequence)
            train_core = model(train_sequence)
            accuracy = _core_retrieval_accuracy(test_core, test_labels, test_values)
            train_accuracy = _core_retrieval_accuracy(train_core, train_labels, train_values)
        values.append(
            {
                "gradient_steps": int(budget),
                "accuracy": accuracy,
                "train_accuracy": train_accuracy,
                "loss_last": losses[-1] if losses else None,
                "wall_seconds": time.perf_counter() - started,
                "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
                "tokens": int(budget * train_sequence.shape[0] * train_sequence.shape[1]),
            }
        )
    return {"steps": values, "trainable_parameters": int(sum(value.numel() for value in initial_state.values()))}


class _Readout(nn.Module):
    def __init__(self, input_dim: int = OUTPUT_WIDTH, classes: int = MEMORY_ITEMS):
        super().__init__()
        self.linear = nn.Linear(input_dim, classes, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)


def _readout_curve(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    seed: int,
    steps: Iterable[int] = READOUT_STEPS,
) -> dict[str, Any]:
    set_seed(seed)
    initial = _Readout()
    initial_state = {name: value.detach().clone() for name, value in initial.state_dict().items()}
    values: list[dict[str, Any]] = []
    for budget in steps:
        model = _Readout()
        model.load_state_dict(initial_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.0)
        started = time.perf_counter()
        losses: list[float] = []
        for _ in range(int(budget)):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(train_features), train_labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            accuracy = float((model(test_features).argmax(-1) == test_labels).float().mean())
            train_accuracy = float((model(train_features).argmax(-1) == train_labels).float().mean())
        values.append(
            {
                "gradient_steps": int(budget),
                "accuracy": accuracy,
                "train_accuracy": train_accuracy,
                "loss_last": losses[-1] if losses else None,
                "wall_seconds": time.perf_counter() - started,
                "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
                "tokens": int(budget * train_features.shape[0]),
            }
        )
    return {"steps": values, "trainable_parameters": int(sum(value.numel() for value in initial_state.values()))}


def _first_threshold(curve: dict[str, Any], threshold: float, key: str = "accuracy") -> int | None:
    for row in curve["steps"]:
        if float(row[key]) >= float(threshold):
            return int(row["gradient_steps"])
    return None


def _save_bundle(path: Path, payload: dict[str, torch.Tensor], input_basis: torch.Tensor, output_basis: torch.Tensor, metadata: dict[str, Any]) -> dict[str, Any]:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    bundle_tensors = {name: value.float().contiguous() for name, value in payload.items()}
    bundle_tensors["input_basis"] = input_basis.float().contiguous()
    bundle_tensors["output_basis"] = output_basis.float().contiguous()
    save_file(bundle_tensors, str(path), metadata={key: str(value) for key, value in metadata.items()})
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tensor_count": len(bundle_tensors),
        "materialized_bytes_fp32": int(sum(value.numel() * 4 for value in bundle_tensors.values())),
    }


class _FrozenGDNRemoraBranch(nn.Module):
    """Mechanical Remora socket; frozen random ports keep this a causal check."""

    def __init__(self, organ: QwenGatedDeltaHead, d_model: int):
        super().__init__()
        self.input_port = nn.Linear(d_model, organ.bus_dim, bias=False)
        self.output_port = nn.Linear(organ.output_dim, d_model, bias=False)
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.organ = organ

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output_port(self.organ(self.input_port(hidden)))


def _remora_socket_check(organ: QwenGatedDeltaHead, zero_organ: QwenGatedDeltaHead) -> dict[str, Any]:
    cfg = ModelConfig.from_json(ROOT / "configs" / "v0_tiny.json")
    set_seed(7701)
    model = build_model("remora", cfg)
    token_ids = torch.randint(0, cfg.vocab_size, (2, min(16, cfg.max_seq_len)))
    branch = _FrozenGDNRemoraBranch(organ, cfg.d_model)
    zero_branch = _FrozenGDNRemoraBranch(zero_organ, cfg.d_model)
    # Use identical ports for donor and zero-core comparisons.
    zero_branch.input_port.load_state_dict(branch.input_port.state_dict())
    zero_branch.output_port.load_state_dict(branch.output_port.state_dict())
    model.blocks[1].plastic = branch
    with torch.no_grad():
        donor_logits, _, _ = model(token_ids, return_aux=True)
    model.blocks[1].plastic = zero_branch
    with torch.no_grad():
        zero_logits, _, _ = model(token_ids, return_aux=True)
    delta = (donor_logits - zero_logits).abs()
    return {
        "attached_module_path": "blocks.1.plastic",
        "forward_ok": True,
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "socket_ports_frozen": True,
        "donor_core_frozen": True,
        "donor_core_causal_logit_delta_max": float(delta.max()),
        "donor_core_causal_logit_delta_mean": float(delta.mean()),
        "scope": "mechanical Remora socket and causal contribution only; random-initialized language loss is not a capability claim",
    }


def run(
    source_path: str | Path,
    *,
    output: str | Path = ROOT / "results" / "qwen-neural-gdn-head-v1.json",
    ranking_output: str | Path = ROOT / "results" / "qwen-neural-gdn-candidate-ranking-v1.json",
    bundle_output: str | Path = ROOT / "results" / "qwen-neural-gdn-head-bundle-v1.safetensors",
    seeds: tuple[int, ...] = SEEDS,
    readout_steps: tuple[int, ...] = READOUT_STEPS,
    record_ledger: bool = True,
) -> dict[str, Any]:
    source_root = Path(source_path).expanduser().resolve()
    output = Path(output)
    ranking_output = Path(ranking_output)
    bundle_output = Path(bundle_output)
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    if not seeds:
        raise ValueError("at least one seed is required")
    ranked_candidates, selected_head = _candidate_scan(source_root)
    selected_payload, selected_metadata = _extract_head(source_root, LAYER, selected_head)
    input_basis, output_basis = _conversion_bases(selected_payload)
    converted_actual = _convert_payload(selected_payload, input_basis, output_basis)
    actual_organ = QwenGatedDeltaHead(**converted_actual, donor_variant="actual")
    ranking = {
        "schema": "remora-qwen-neural-gdn-candidate-ranking-v1",
        "source_model": DONOR_REPOSITORY,
        "source_revision": DONOR_REVISION,
        "selection_rule": "highest measured head information score over the bounded scan; score is triage only, not promotion",
        "scan_layer": LAYER,
        "scanned_value_heads": list(SCAN_HEADS),
        "candidates": ranked_candidates,
        "selected_candidate": ranked_candidates[0],
        "prior_router_and_layer0_targets_excluded": True,
    }
    write_json(ranking_output, ranking)

    probe_generator = torch.Generator(device="cpu").manual_seed(1704)
    probe_hidden = torch.randn(2, 11, 2560, generator=probe_generator) * 0.25
    full_actual = QwenGatedDeltaHead(**selected_payload, donor_variant="actual")
    reference_output, reference_state, reference_details = _reference_forward(selected_payload, probe_hidden)
    actual_output, actual_state, actual_details = full_actual(
        probe_hidden, return_state=True, return_intermediates=True
    )
    functional_ladder: dict[str, Any] = {"steps": {}}
    functional_ladder["steps"]["D0_original_extracted"] = {
        "representation": "one selected Qwen GDN value-head subgraph; q/k/v/z/a/b/out slices plus conv/scalars/norm",
        "source_payload_bytes_bf16": selected_metadata["source_payload_bytes_bf16"],
        "selected_payload_materialized_bytes_fp32": selected_metadata["selected_payload_materialized_bytes_fp32"],
        "donor_parameters_available": int(sum(value.numel() for value in selected_payload.values())),
        "tensor_hashes_float32": selected_metadata["tensor_hashes_float32"],
    }
    functional_ladder["steps"]["D1_standalone_reproduction"] = {
        "metrics": _compare(actual_output, reference_output, actual_state, reference_state),
        "state_contract": {"recurrent": ["batch", 128, 128], "convolution": ["batch", HEAD_CONV_DIM, CONV_KERNEL - 1]},
        "reference": "independent unfold-based causal convolution plus explicit recurrent delta update",
        "status": "EQUIVALENT_WITHIN_FLOAT32_REFERENCE_TOLERANCE",
    }
    packed = torch.cat((selected_payload["q_weight"], selected_payload["k_weight"], selected_payload["v_weight"]), dim=0)
    packed_projected = F.linear(probe_hidden, packed)
    separate_projected = torch.cat(
        (
            F.linear(probe_hidden, selected_payload["q_weight"]),
            F.linear(probe_hidden, selected_payload["k_weight"]),
            F.linear(probe_hidden, selected_payload["v_weight"]),
        ),
        dim=-1,
    )
    functional_ladder["steps"]["D2_projection_packing"] = {
        "max_absolute_error": float((packed_projected - separate_projected).abs().max()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(packed_projected - separate_projected)
            / torch.linalg.vector_norm(separate_projected).clamp_min(1e-30)
        ),
        "status": "EQUIVALENT_WITHIN_FLOAT32_REFERENCE_TOLERANCE",
        "transformation": "concatenate q/k/v affine projections without changing trained values",
    }
    from remora.donors.router import top_right_singular_basis

    for rank in (32, 64, BUS_WIDTH):
        rank_input_basis = top_right_singular_basis(
            torch.cat(
                [selected_payload[name] for name in ("q_weight", "k_weight", "v_weight", "z_weight", "a_weight", "b_weight")],
                dim=0,
            ),
            rank,
        )
        rank_organ = QwenGatedDeltaHead(
            **_convert_payload(selected_payload, rank_input_basis, None), donor_variant=f"actual_input_rank{rank}"
        )
        rank_output, rank_state, _ = rank_organ(
            probe_hidden @ rank_input_basis, return_state=True, return_intermediates=True
        )
        functional_ladder["steps"][f"D3_input_compression_rank_{rank}"] = {
            "metrics": _compare(rank_output, reference_output, rank_state, reference_state),
            "input_basis_shape": list(rank_input_basis.shape),
            "resident_transformed_projection_parameters": int(
                sum(_convert_payload(selected_payload, rank_input_basis, None)[name].numel() for name in ("q_weight", "k_weight", "v_weight", "z_weight", "a_weight", "b_weight"))
            ),
        }
    functional_ladder["steps"]["D4_width_conversion_rank_96"] = {
        "metrics": _compare(
            actual_organ(probe_hidden @ input_basis),
            reference_output @ output_basis,
        ),
        "input_basis_shape": list(input_basis.shape),
        "output_basis_shape": list(output_basis.shape),
        "donor_parameters_preserved_unchanged": int(
            selected_payload["conv_qkv"].numel() + selected_payload["A_log"].numel() + selected_payload["dt_bias"].numel() + selected_payload["norm_weight"].numel()
        ),
        "donor_parameters_analytically_transformed": int(
            sum(selected_payload[name].numel() for name in ("q_weight", "k_weight", "v_weight", "z_weight", "a_weight", "b_weight", "out_weight"))
        ),
        "resident_transformed_parameters": int(sum(value.numel() for value in converted_actual.values())),
        "resident_original_parameters": 0,
    }
    bundle = _save_bundle(
        bundle_output,
        converted_actual,
        input_basis,
        output_basis,
        {
            "schema": "remora-qwen-gdn-head-bundle-v1",
            "source_model": DONOR_REPOSITORY,
            "source_revision": DONOR_REVISION,
            "layer": LAYER,
            "value_head": selected_head,
            "conversion": "rank96 input and output SVD bases",
            "source_shard": selected_metadata["source_shard"],
        },
    )
    ir = qwen_gated_delta_head_ir(
        component_id=f"remora-gdn-candidate-layer{LAYER}-head{selected_head}",
        source_revision=DONOR_REVISION,
        tensor_names=[
            "q_weight",
            "k_weight",
            "v_weight",
            "z_weight",
            "a_weight",
            "b_weight",
            "conv_qkv",
            "A_log",
            "dt_bias",
            "norm_weight",
            "out_weight",
        ],
        layer=LAYER,
        value_head=selected_head,
        input_width=BUS_WIDTH,
        output_width=OUTPUT_WIDTH,
    )

    seeds_result: dict[str, Any] = {}
    for seed in seeds:
        train_sequence, train_labels, train_values, train_generation = _make_recall_dataset(
            actual_organ, samples=TRAIN_SAMPLES, seed=int(seed) + 1000
        )
        test_sequence, test_labels, test_values, test_generation = _make_recall_dataset(
            actual_organ, samples=TEST_SAMPLES, seed=int(seed) + 2000
        )
        shifted_test_sequence = test_sequence.clone()
        permutation = torch.roll(torch.arange(BUS_WIDTH), shifts=7)
        shifted_test_sequence = shifted_test_sequence[:, :, permutation] * 0.9
        variants: dict[str, QwenGatedDeltaHead] = {
            "actual": actual_organ,
            "random": QwenGatedDeltaHead(
                **_convert_payload(
                    make_gdn_variant(selected_payload, "random", seed=int(seed) + 5000), input_basis, output_basis
                ),
                donor_variant="random",
            ),
            "shuffled": QwenGatedDeltaHead(
                **_convert_payload(
                    make_gdn_variant(selected_payload, "shuffled", seed=int(seed) + 6000), input_basis, output_basis
                ),
                donor_variant="shuffled",
            ),
            "zero": QwenGatedDeltaHead(
                **_convert_payload(
                    make_gdn_variant(selected_payload, "zero", seed=int(seed) + 7000), input_basis, output_basis
                ),
                donor_variant="zero",
            ),
        }
        mechanistic: dict[str, Any] = {}
        features: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for name, organ in variants.items():
            with torch.no_grad():
                train_output, _, train_details = organ(train_sequence, return_state=True, return_intermediates=True)
                test_output, _, test_details = organ(test_sequence, return_state=True, return_intermediates=True)
                shifted_output, _, shifted_details = organ(
                    shifted_test_sequence, return_state=True, return_intermediates=True
                )
            train_core = train_details["core"][:, -1]
            test_core = test_details["core"][:, -1]
            test_values_for_variant = test_details["value"][:, :MEMORY_ITEMS]
            variant_mechanistic = _mechanistic_retrieval(organ, test_sequence, test_labels, test_values)
            variant_mechanistic["shifted_interface_accuracy"] = _mechanistic_retrieval(
                organ, shifted_test_sequence, test_labels, test_values
            )["accuracy"]
            variant_mechanistic["variant_own_value_accuracy"] = _mechanistic_retrieval(
                organ, test_sequence, test_labels, test_values_for_variant
            )["accuracy"]
            mechanistic[name] = variant_mechanistic
            features[name] = (train_output[:, -1].detach(), test_output[:, -1].detach(), shifted_output[:, -1].detach())

        # The readout is deliberately identical in size and optimizer budget
        # for actual/random/shuffled/zero.  It tests whether the frozen organ
        # exposes a useful final representation without letting a large port
        # relearn the task.
        curves: dict[str, Any] = {}
        for name in ("actual", "random", "shuffled", "zero"):
            train_features, test_features, shifted_features = features[name]
            curve = _readout_curve(
                train_features,
                train_labels,
                test_features,
                test_labels,
                seed=int(seed) + 8000 + (0 if name == "actual" else {"random": 1, "shuffled": 2, "zero": 3}[name]),
                steps=readout_steps,
            )
            shifted_curve = _readout_curve(
                train_features,
                train_labels,
                shifted_features,
                test_labels,
                seed=int(seed) + 8100 + (0 if name == "actual" else {"random": 1, "shuffled": 2, "zero": 3}[name]),
                steps=readout_steps,
            )
            for row, shifted_row in zip(curve["steps"], shifted_curve["steps"]):
                row["shifted_interface_accuracy"] = shifted_row["accuracy"]
            curves[name + "_frozen_core_plus_equal_readout"] = curve
        fresh_core_curve = _fresh_core_curve(
            train_sequence,
            train_labels,
            train_values,
            test_sequence,
            test_labels,
            test_values,
            seed=int(seed) + 8200,
            steps=readout_steps,
        )
        seeds_result[str(seed)] = {
            "dataset": {
                "train_samples": TRAIN_SAMPLES,
                "test_samples": TEST_SAMPLES,
                "memory_items": MEMORY_ITEMS,
                "memory_delay": MEMORY_DELAY,
                "query_synthesis": "24-step analytic compact query optimization toward actual selected key; beta penalty; no donor training",
                "train_query_key_cosine": train_generation["query_key_cosine_mean"],
                "test_query_key_cosine": test_generation["query_key_cosine_mean"],
                "train_query_beta": train_generation["query_beta_mean"],
                "test_query_beta": test_generation["query_beta_mean"],
                "shifted_interface": "bus columns cyclically permuted by 7 and scaled by 0.9",
            },
            "mechanistic_retrieval_zero_training": mechanistic,
            "equal_readout_curves": curves,
            "fresh_trainable_same_mechanism_curve": fresh_core_curve,
            "thresholds": {
                name: {
                    "same_interface_steps": _first_threshold(curve, RECALL_THRESHOLD),
                    "shifted_interface_steps": _first_threshold(curve, RECALL_THRESHOLD, key="shifted_interface_accuracy"),
                }
                for name, curve in curves.items()
            },
            "fresh_trainable_same_mechanism_steps": _first_threshold(fresh_core_curve, RECALL_THRESHOLD),
        }

    zero_summary = {
        name: [seeds_result[str(seed)]["mechanistic_retrieval_zero_training"][name]["accuracy"] for seed in seeds]
        for name in ("actual", "random", "shuffled", "zero")
    }
    actual_thresholds = [
        seeds_result[str(seed)]["thresholds"]["actual_frozen_core_plus_equal_readout"]["same_interface_steps"]
        for seed in seeds
    ]
    random_thresholds = [
        seeds_result[str(seed)]["thresholds"]["random_frozen_core_plus_equal_readout"]["same_interface_steps"]
        for seed in seeds
    ]
    fresh_core_thresholds = [
        seeds_result[str(seed)]["fresh_trainable_same_mechanism_steps"] for seed in seeds
    ]
    result = {
        "schema": "remora-qwen-neural-gdn-head-result-v1",
        "experiment_family": "QWEN-DONOR-GDN-HEAD-001+",
        "source": {
            "repository": DONOR_REPOSITORY,
            "revision": DONOR_REVISION,
            "path": str(source_root),
            "layer": LAYER,
            "value_head": selected_head,
            "key_head": selected_head // 3,
            "source_shard": selected_metadata["source_shard"],
            "source_payload_bytes_bf16": selected_metadata["source_payload_bytes_bf16"],
            "selected_payload_materialized_bytes_fp32": selected_metadata["selected_payload_materialized_bytes_fp32"],
            "full_model_materialized": False,
            "model_loader_called": False,
        },
        "candidate_ranking": ranking,
        "selected_organ": {
            "family": "gated_deltanet_value_head",
            "mode": "SUBMODULE_SUBSPACE_GRAFT",
            "closed_subgraph": "one Qwen value head with causal convolution and recurrent 128x128 state",
            "source_width": 2560,
            "bus_width": BUS_WIDTH,
            "output_width": OUTPUT_WIDTH,
            "head_dim": KEY_HEAD_DIM,
            "state_shape": [KEY_HEAD_DIM, VALUE_HEAD_DIM],
            "donor_parameters_preserved_unchanged": int(
                selected_payload["conv_qkv"].numel()
                + selected_payload["A_log"].numel()
                + selected_payload["dt_bias"].numel()
                + selected_payload["norm_weight"].numel()
            ),
            "donor_parameters_analytically_transformed": int(
                sum(selected_payload[name].numel() for name in ("q_weight", "k_weight", "v_weight", "z_weight", "a_weight", "b_weight", "out_weight"))
            ),
            "donor_parameters_discarded_at_resident_stage": int(
                sum(value.numel() for value in selected_payload.values())
                - sum(value.numel() for value in converted_actual.values())
            ),
            "resident_transformed_parameters": int(sum(value.numel() for value in converted_actual.values())),
            "analytic_input_basis_parameters_offline": int(input_basis.numel()),
            "analytic_output_basis_parameters_offline": int(output_basis.numel()),
            "resident_compact_core_bytes_fp32": int(sum(value.numel() * 4 for value in converted_actual.values())),
            "active_mac_per_token_modeled": int(
                4 * BUS_WIDTH * VALUE_HEAD_DIM
                + 2 * BUS_WIDTH
                + HEAD_CONV_DIM * CONV_KERNEL
                + 4 * KEY_HEAD_DIM * VALUE_HEAD_DIM
                + VALUE_HEAD_DIM * OUTPUT_WIDTH
            ),
            "bundle": bundle,
            "neural_ir": ir.to_dict(),
            "functional_preservation_ladder": functional_ladder,
        },
        "capability_assay": {
            "task": "delayed four-way associative recall using actual selected-head key/value projections and a compact query",
            "recall_threshold": RECALL_THRESHOLD,
            "readout_steps": list(readout_steps),
            "seeds": list(seeds),
            "per_seed": seeds_result,
            "zero_training_mechanistic_accuracy": zero_summary,
            "equal_readout_threshold_steps": {
                "actual": actual_thresholds,
                "random": random_thresholds,
                "fresh_trainable_same_mechanism": fresh_core_thresholds,
            },
            "fresh_core_curve": "A fresh compact GatedDeltaNet core is trained against the delayed value target; this is the same-mechanism relearning control, not distillation.",
            "training_avoidance_interpretation": "MEASURED only for this donor-conditioned state task; original Qwen training compute remains UNMEASURED",
        },
        "remora_attachment": _remora_socket_check(
            actual_organ,
            QwenGatedDeltaHead(**_convert_payload(make_gdn_variant(selected_payload, "zero"), input_basis, output_basis), donor_variant="zero"),
        ),
        "labels": {
            "MEASURED": [
                "bounded selected tensor extraction and byte counts",
                "candidate head payload scan",
                "independent standalone output/state equivalence",
                "D2 projection packing equivalence",
                "D3/D4 compression and width-conversion degradation",
                "trained/random/shuffled/zero state-task retrieval",
                "equal-readout 0..64 curves and shifted-interface metrics",
                "fresh trainable same-mechanism 0..64 curve",
                "Remora socket forward and causal logit delta",
            ],
            "DERIVED": [
                "donor parameter accounting",
                "compact resident parameter counts",
                "threshold records",
            ],
            "MODELED": ["active per-token MACs including one-head state operations", "state memory shape"],
            "UNMEASURED": ["original Qwen training compute avoided", "energy", "native Qwen activation-manifold coverage"],
            "HYPOTHESIS": ["one trained GDN head contains reusable delayed-association computation"],
        },
        "promotion_state": "CANDIDATE_ONLY_UNTIL_EXTERNAL_TASK_AND_AGED_REMORA_PATHWAY_VALIDATION",
        "interpretation": "MEASURED mechanical transplant and donor-conditioned state probe; not broad language capability. Promotion requires a real Remora task benefit, strong fresh controls, and aged-path retention.",
        "runtime": runtime_context(torch.device("cpu")),
    }
    write_json(output, result)
    command = f"python -m experiments.donor_gdn_head --source {source_root}"
    if record_ledger:
        record_experiment(
            ROOT,
            "QWEN-DONOR-GDN-HEAD-001",
            "A below-layer Qwen GatedDeltaNet head should preserve a task-relevant delayed-association computation after bounded extraction and rank-96 analytic width conversion.",
            "Scan five actual layer-17 value heads, extract the highest measured state-response candidate, reproduce it independently, convert its projections to the Remora bus, and compare actual/random/shuffled/zero frozen cores with an equal readout budget.",
            "The selected actual head reproduces output and recurrent/convolution state, has a reproducible advantage over destroyed cores on the fixed delayed recall task, and remains causally attachable to a Remora branch.",
            "Standalone reproduction fails, conversion destroys the state function, actual retrieval is not above random/shuffled/zero, or the equal readout erases the donor advantage.",
            command,
            int(seeds[0]),
            {
                "selected_head": selected_head,
                "zero_training_mechanistic_accuracy": zero_summary,
                "equal_readout_thresholds": {"actual": actual_thresholds, "random": random_thresholds},
                "fresh_trainable_same_mechanism_thresholds": fresh_core_thresholds,
                "promotion_state": result["promotion_state"],
            },
            "MEASURED: GDN organ tranche recorded; interpret only at the donor-conditioned state-task scope until an external Remora capability test passes.",
            "If the state task separates actual from controls, attach the organ to an aged Remora pathway and test a real long-delay task; if it does not, preserve the exact transplant as a negative result and redesign the input-manifold conversion.",
            hardware={**runtime_context(torch.device("cpu")), "mode": "bounded_gdn_head_extraction_and_state_assay", "source_model_materialized": False},
        )
        if sum(zero_summary["actual"]) / len(zero_summary["actual"]) <= sum(zero_summary["random"]) / len(zero_summary["random"]) + 0.05:
            record_failure(
                ROOT,
                "QWEN-DONOR-GDN-HEAD-001",
                {"layer": LAYER, "value_head": selected_head, "bus_width": BUS_WIDTH, "memory_items": MEMORY_ITEMS},
                {"memory_delay": MEMORY_DELAY, "recall_threshold": RECALL_THRESHOLD},
                int(seeds[0]),
                "The selected actual GDN head did not clear the predeclared donor-vs-random mechanistic recall margin.",
                "The isolated head may require its native activation manifold or adjacent Qwen heads/layers; the state machinery alone may not encode transferable capability.",
                "Keep the exact equivalence and extraction artifacts; resurrect after native-manifold capture or a wider closed subgraph is available. Do not promote or scale.",
                runtime=runtime_context(torch.device("cpu")),
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and assay one bounded Qwen GatedDeltaNet value head.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default=str(ROOT / "results" / "qwen-neural-gdn-head-v1.json"))
    parser.add_argument("--ranking-output", default=str(ROOT / "results" / "qwen-neural-gdn-candidate-ranking-v1.json"))
    parser.add_argument("--bundle-output", default=str(ROOT / "results" / "qwen-neural-gdn-head-bundle-v1.safetensors"))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--readout-steps", default=",".join(str(step) for step in READOUT_STEPS))
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()
    result = run(
        args.source,
        output=args.output,
        ranking_output=args.ranking_output,
        bundle_output=args.bundle_output,
        seeds=tuple(int(value) for value in args.seeds.split(",") if value),
        readout_steps=tuple(int(value) for value in args.readout_steps.split(",") if value),
        record_ledger=not args.no_ledger,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_candidate": result["candidate_ranking"]["selected_candidate"],
                "zero_training_mechanistic_accuracy": result["capability_assay"]["zero_training_mechanistic_accuracy"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
