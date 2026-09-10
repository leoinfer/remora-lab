from __future__ import annotations

"""Search, compress, and graft one real Qwen router subcircuit.

This experiment is intentionally narrower than the earlier shared-expert
pilot.  The donor organ is a selected pair of rows from a trained Qwen MoE
router, converted into the 96-wide Remora routing socket with a fixed
truncated-SVD input basis.  The capability assay is a donor-conditioned
routing-geometry task: held-out hidden prototypes are generated around the
selected trained router rows.  It is a capability probe for the router, not a
claim about broad Qwen language competence.

The script reads only router tensors from the source checkpoint with
``safe_open``.  It never constructs a Transformers model or materializes the
full donor.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.config import ModelConfig
from remora.donors.router import (
    QwenRouterOrgan,
    make_router_variant,
    router_functional_metrics,
    select_balanced_pair,
    top_right_singular_basis,
)
from remora.ledger import record_experiment, record_failure
from remora.models import build_model
from remora.utils import runtime_context, set_seed, write_json


DONOR_REPOSITORY = "Qwen/Qwen3.8-Flash-Next"
DONOR_REVISION = "f5d08274bafd880402bd16f5e3e6c514136ec06c"
ROUTER_LAYERS = tuple(range(48))
LINEAR_ATTENTION_LAYERS = (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45, 46)
RANKS = (8, 16, 32, 64, 96)
SEEDS = (7, 19, 31)
CURVE_STEPS = (0, 1, 2, 4, 8, 16, 32, 64)


def _sha256_tensor(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load_router(source_root: Path, layer: int) -> tuple[torch.Tensor, str, str]:
    from safetensors import safe_open

    name = f"model.language_model.layers.{layer}.mlp.gate.weight"
    index = json.loads((source_root / "model.safetensors.index.json").read_text())
    shard_name = str(index["weight_map"][name])
    shard = source_root / shard_name
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        if name not in handle.keys():
            raise KeyError(f"{name!r} is absent from {shard}")
        # This is one 512 x 2560 BF16 tensor (2.5 MiB).  It is the only
        # payload read for a router candidate; no shard-wide tensor loop runs.
        weight = handle.get_tensor(name).float().contiguous()
    if tuple(weight.shape) != (512, 2560):
        raise ValueError(f"unexpected Qwen router shape for layer {layer}: {tuple(weight.shape)}")
    return weight, shard_name, name


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean())


def _route_accuracy(weight: torch.Tensor, inputs: torch.Tensor, labels: torch.Tensor) -> float:
    return _accuracy(inputs @ weight.transpose(0, 1), labels)


def _randomized_scan(weight: torch.Tensor, *, layer: int, seed: int) -> dict[str, Any]:
    """Measure actual-vs-destroyed row geometry over analytic bus widths."""

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    basis_full = top_right_singular_basis(weight, 96)
    normalized = torch.nn.functional.normalize(weight, dim=-1)
    prototypes = normalized + 0.15 * torch.randn(normalized.shape, generator=generator)
    labels = torch.arange(weight.shape[0])
    random_weight = make_router_variant(weight, "random", seed=seed + 101)
    shuffled_weight = make_router_variant(weight, "shuffled", seed=seed + 102)
    scan: dict[str, Any] = {
        "layer": int(layer),
        "input_width": int(weight.shape[1]),
        "output_routes": int(weight.shape[0]),
        "source_payload_bytes": int(weight.numel() * 2),
        "source_tensor_sha256_float32_read": _sha256_tensor(weight),
        "row_norm_mean": float(weight.norm(dim=-1).mean()),
        "row_norm_std": float(weight.norm(dim=-1).std(unbiased=False)),
        "donor_conditioned_probe": {
            "input_definition": "unit-normalized actual donor rows plus fixed Gaussian noise std=0.15",
            "labels": "row index 0..511",
            "full_rank": {
                "actual_accuracy": _route_accuracy(weight, prototypes, labels),
                "random_accuracy": _route_accuracy(random_weight, prototypes, labels),
                "shuffled_accuracy": _route_accuracy(shuffled_weight, prototypes, labels),
            },
        },
        "compression": {},
    }
    # Compute the pair statistic once and do not retain the matrix in the
    # result path.
    pair_cosine = torch.nn.functional.normalize(weight, dim=-1)
    pair_cosine = pair_cosine @ pair_cosine.transpose(0, 1)
    off_diagonal = pair_cosine[~torch.eye(pair_cosine.shape[0], dtype=torch.bool)]
    scan["row_pair_cosine_mean"] = float(off_diagonal.mean())

    _, singular, vh = torch.linalg.svd(weight, full_matrices=False)
    energy = singular.square()
    basis_all = vh.transpose(0, 1)
    for rank in RANKS:
        basis = basis_all[:, :rank].contiguous()
        compact_weight = weight @ basis
        compact_random = random_weight @ basis
        compact_shuffled = shuffled_weight @ basis
        compact_inputs = prototypes @ basis
        actual_logits = compact_inputs @ compact_weight.transpose(0, 1)
        reference_logits = prototypes @ weight.transpose(0, 1)
        scan["compression"][str(rank)] = {
            "basis_shape": [int(value) for value in basis.shape],
            "singular_energy_fraction": float(energy[:rank].sum() / energy.sum().clamp_min(1e-30)),
            "actual_accuracy": _accuracy(actual_logits, labels),
            "random_accuracy": _accuracy(compact_inputs @ compact_random.transpose(0, 1), labels),
            "shuffled_accuracy": _accuracy(compact_inputs @ compact_shuffled.transpose(0, 1), labels),
            "actual_minus_random_accuracy": _accuracy(actual_logits, labels)
            - _accuracy(compact_inputs @ compact_random.transpose(0, 1), labels),
            "function_preservation": router_functional_metrics(
                weight,
                basis,
                selected_rows=tuple(range(weight.shape[0])),
                inputs=prototypes[:64],
            ),
            "full_logit_function_reference_for_context": {
                "mean_absolute_error": float((actual_logits - reference_logits).abs().mean()),
                "relative_l2_error": float(
                    torch.linalg.vector_norm(actual_logits - reference_logits)
                    / torch.linalg.vector_norm(reference_logits).clamp_min(1e-30)
                ),
            },
        }
    scan["singular_values"] = {
        "stable_rank": float(energy.sum() / singular[0].square().clamp_min(1e-30)),
        "rank_at_90pct_energy": int(torch.searchsorted(torch.cumsum(energy, 0) / energy.sum(), torch.tensor(0.90)).item() + 1),
        "rank_at_99pct_energy": int(torch.searchsorted(torch.cumsum(energy, 0) / energy.sum(), torch.tensor(0.99)).item() + 1),
        "top_8": [float(value) for value in singular[:8]],
    }
    scan["rank96_self_probe_accuracy"] = scan["compression"]["96"]["actual_accuracy"]
    scan["rank96_actual_minus_random"] = scan["compression"]["96"]["actual_minus_random_accuracy"]
    scan["rank96_energy_fraction"] = scan["compression"]["96"]["singular_energy_fraction"]
    return scan


def _gdn_head_candidate(anatomy: dict[str, Any]) -> dict[str, Any]:
    """Describe a below-layer GatedDeltaNet head without loading it."""

    layer = 17
    prefix = f"model.language_model.layers.{layer}.linear_attn."
    tensor_bytes = {
        "in_proj_qkv.weight": 384 * 2560 * 2,
        "in_proj_z.weight": 128 * 2560 * 2,
        "in_proj_a.weight": 1 * 2560 * 2,
        "in_proj_b.weight": 1 * 2560 * 2,
        "conv1d.weight": 384 * 4 * 2,
        "A_log": 2,
        "dt_bias": 2,
        "norm.weight": 128 * 2,
        "out_proj.weight": 2560 * 128 * 2,
    }
    source_shards = sorted(
        {
            tensor["source_shard"]
            for component in anatomy.get("components", [])
            if component.get("component_key") == f"model.language_model.layers.{layer}.linear_attn"
            for tensor in component.get("tensors", [])
        }
    )
    return {
        "candidate_id": f"qwen3.8.language.layer{layer}.linear_attn.head10",
        "family": "gated_deltanet_linear_attention",
        "closed_subgraph": "one value head: q/k/v projection rows + beta/decay rows + causal-conv rows + gated-delta state + z/out projection slices",
        "source_tensor_prefix": prefix,
        "selected_head": 10,
        "source_shards": source_shards,
        "payload_bytes_estimated": int(sum(tensor_bytes.values())),
        "retained_parameters_estimated": int(sum(value // 2 for value in tensor_bytes.values())),
        "state_shape": [1, 128, 128],
        "expected_capability": "long-delay associative state update",
        "actual_weight_scan_status": "NOT_RUN_IN_THIS_TRANCHE",
        "selection_note": "kept as a high-value next candidate; exact source equations are established, but no graft is claimed here",
    }


def _build_candidate_ranking(anatomy_path: Path, scans: list[dict[str, Any]]) -> dict[str, Any]:
    anatomy_result = json.loads(anatomy_path.read_text())
    anatomy = anatomy_result.get("anatomy", {})
    router_rows = []
    for scan in scans:
        compression = scan["compression"]["96"]
        # This is a triage score, not a capability claim.  It rewards measured
        # donor-vs-random separation and retained structure, with a tiny cost
        # preference for the two-row organ chosen later.
        score = (
            0.55 * float(compression["actual_minus_random_accuracy"])
            + 0.30 * float(compression["actual_accuracy"])
            + 0.10 * float(compression["singular_energy_fraction"])
            + 0.05 * (1.0 / (1.0 + math.log10(max(scan["source_payload_bytes"], 1))))
        )
        router_rows.append(
            {
                "candidate_id": f"qwen3.8.language.layer{scan['layer']}.router",
                "family": "moe_router",
                "layer": scan["layer"],
                "source_payload_bytes": scan["source_payload_bytes"],
                "closed_subgraph": "one trained router tensor; selected output rows plus analytic input basis",
                "expected_capability": "learned expert-routing partition",
                "trained_vs_random_signal": compression["actual_minus_random_accuracy"],
                "compression_rank": 96,
                "compression_energy_fraction": compression["singular_energy_fraction"],
                "expected_port_parameters": 0,
                "expected_active_flops_per_token": 2 * 96,
                "estimated_graft_difficulty": 0.25,
                "actual_weight_scan_status": "MEASURED",
                "information_scan_score": score,
                "selection_basis": "actual trained-vs-random donor-conditioned route probe plus rank-96 conversion",
            }
        )
    gdn = _gdn_head_candidate(anatomy)
    gdn.update(
        {
            "trained_vs_random_signal": None,
            "compression_rank": None,
            "compression_energy_fraction": None,
            "expected_port_parameters": 0,
            "expected_active_flops_per_token": None,
            "estimated_graft_difficulty": 0.78,
            "information_scan_score": None,
            "actual_weight_scan_status": "NOT_MEASURED",
        }
    )
    hyper = {
        "candidate_id": "qwen3.8.language.layer1.attn_hyper_connection",
        "family": "gated_residual_hyperconnections",
        "layer": 1,
        "source_payload_bytes": 13209600,
        "closed_subgraph": "four-stream residual mixing low-rank projection plus injection gate",
        "expected_capability": "cross-branch residual routing",
        "trained_vs_random_signal": None,
        "compression_rank": 96,
        "compression_energy_fraction": None,
        "expected_port_parameters": 0,
        "expected_active_flops_per_token": None,
        "estimated_graft_difficulty": 0.55,
        "actual_weight_scan_status": "NOT_MEASURED",
        "information_scan_score": None,
        "selection_basis": "retained as a compact architectural candidate; payload utility not yet measured",
    }
    candidates = sorted(router_rows, key=lambda row: float(row["information_scan_score"]), reverse=True)
    candidates.extend([gdn, hyper])
    selected = candidates[0]
    return {
        "schema": "remora-qwen-neural-candidate-ranking-v1",
        "source_model": DONOR_REPOSITORY,
        "source_revision": DONOR_REVISION,
        "anatomy_source": str(anatomy_path),
        "prior_target_excluded": "qwen3.8.language.layer0.shared_expert",
        "ranking_is": "MEASURED where actual_weight_scan_status=MEASURED; triage score is not a promotion decision",
        "candidates": candidates,
        "selected_candidate": selected,
        "selection_rule": "highest measured information_scan_score; unmeasured GDN/hyper candidates cannot outrank measured evidence",
    }


def _make_binary_dataset(
    weight: torch.Tensor,
    pair: tuple[int, int],
    basis: torch.Tensor,
    *,
    count_per_class: int,
    noise_std: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    centers = torch.nn.functional.normalize(weight[list(pair)], dim=-1)
    labels = torch.arange(2).repeat_interleave(int(count_per_class))
    center_rows = centers[labels]
    noise = torch.randn(center_rows.shape, generator=generator)
    # Normalize noise per sample so the difficulty is stable across donor
    # layers and pair norms.
    noise = torch.nn.functional.normalize(noise, dim=-1)
    donor_inputs = center_rows + float(noise_std) * noise
    return donor_inputs @ basis, labels


class _FreshMatchedRouter(nn.Module):
    """Fresh control with approximately the rank-1 repair parameter budget."""

    def __init__(self, bus_dim: int = 96, hidden_dim: int = 27, routes: int = 2):
        super().__init__()
        self.down = nn.Linear(bus_dim, hidden_dim, bias=False)
        self.up = nn.Linear(hidden_dim, routes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


def _accuracy_model(model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        return _accuracy(model(inputs), labels)


def _fresh_curve(
    model_factory,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    seed: int,
    steps: tuple[int, ...] = CURVE_STEPS,
) -> dict[str, Any]:
    set_seed(seed)
    initial = model_factory()
    initial_state = {name: value.detach().clone() for name, value in initial.state_dict().items()}
    values = []
    for budget in steps:
        model = model_factory()
        model.load_state_dict(initial_state, strict=True)
        model.train()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=0.05, betas=(0.9, 0.95), weight_decay=0.0)
        started = time.perf_counter()
        losses: list[float] = []
        for step in range(int(budget)):
            # Full-batch updates make the threshold curve independent of a
            # hidden data-loader schedule while preserving fixed token counts.
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(train_x), train_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        elapsed = time.perf_counter() - started
        values.append(
            {
                "gradient_steps": int(budget),
                "accuracy": _accuracy_model(model, test_x, test_y),
                "train_accuracy": _accuracy_model(model, train_x, train_y),
                "loss_last": losses[-1] if losses else None,
                "wall_seconds": elapsed,
                "trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
                "tokens": int(budget * train_x.shape[0]),
            }
        )
    return {"steps": values, "initial_parameter_count": int(sum(value.numel() for value in initial_state.values()))}


def _donor_curve(
    model_factory,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    *,
    seed: int,
    steps: tuple[int, ...] = CURVE_STEPS,
) -> dict[str, Any]:
    """Train a frozen-core graft only through its explicitly counted repair."""

    values = []
    for budget in steps:
        set_seed(seed)
        model = model_factory()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=0.05, betas=(0.9, 0.95), weight_decay=0.0) if trainable else None
        started = time.perf_counter()
        losses: list[float] = []
        for _ in range(int(budget)):
            if optimizer is None:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(train_x), train_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        values.append(
            {
                "gradient_steps": int(budget),
                "accuracy": _accuracy_model(model, test_x, test_y),
                "train_accuracy": _accuracy_model(model, train_x, train_y),
                "loss_last": losses[-1] if losses else None,
                "wall_seconds": time.perf_counter() - started,
                "trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
                "tokens": int(budget * train_x.shape[0]),
                "donor_core_frozen": True,
            }
        )
    return {"steps": values}


def _first_threshold(curve: dict[str, Any], threshold: float) -> int | None:
    for row in curve["steps"]:
        if float(row["accuracy"]) >= float(threshold):
            return int(row["gradient_steps"])
    return None


def _remora_socket_check(
    weight: torch.Tensor,
    basis: torch.Tensor,
    pair: tuple[int, int],
    cfg_path: Path,
) -> dict[str, Any]:
    """Attach the graft to the actual Remora expert router and run a forward."""

    cfg = ModelConfig.from_json(cfg_path)
    set_seed(7001)
    model = build_model("remora", cfg)
    original = model.blocks[1].experts.router
    actual = QwenRouterOrgan(weight, input_basis=basis, selected_rows=pair)
    model.blocks[1].experts.router = actual
    token_ids = torch.randint(0, cfg.vocab_size, (2, min(24, cfg.max_seq_len)))
    with torch.no_grad():
        graft_logits, _, graft_aux = model(token_ids, return_aux=True)
        graft_routes = graft_aux["route_weights"][1]
        actual_core = model.blocks[1].experts.router
        model.blocks[1].experts.router = QwenRouterOrgan(
            weight,
            input_basis=basis,
            selected_rows=pair,
            donor_variant="zero",
        )
        zero_logits, _, zero_aux = model(token_ids, return_aux=True)
        model.blocks[1].experts.router = original
    return {
        "attached_module_path": "blocks.1.experts.router",
        "forward_ok": True,
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "graft_interface": actual_core.interface_signature(),
        "route_probability_mean": [float(value) for value in graft_routes.mean(dim=(0, 1))],
        "donor_core_causal_logit_delta_max": float((graft_logits - zero_logits).abs().max()),
        "donor_core_causal_logit_delta_mean": float((graft_logits - zero_logits).abs().mean()),
        "zero_control_route_delta_mean": float((graft_routes - zero_aux["route_weights"][1]).abs().mean()),
        "scope": "socket integration and causal contribution only; random-initialized Remora language loss is not a capability claim",
    }


def _save_selected_bundle(
    output: Path,
    weight: torch.Tensor,
    basis: torch.Tensor,
    pair: tuple[int, int],
    *,
    layer: int,
    shard: str,
    tensor_name: str,
) -> dict[str, Any]:
    from safetensors.torch import save_file

    selected = weight[list(pair)].contiguous()
    metadata = {
        "schema": "remora-qwen-router-subspace-bundle-v1",
        "source_model": DONOR_REPOSITORY,
        "source_revision": DONOR_REVISION,
        "source_shard": shard,
        "source_tensor": tensor_name,
        "layer": str(layer),
        "selected_rows": json.dumps(list(pair)),
        "conversion": "top-right-singular-basis-rank-96",
        "donor_parameters_preserved_unchanged": str(selected.numel()),
        "donor_parameters_discarded": str(weight.numel() - selected.numel()),
        "analytic_basis_parameters": str(basis.numel()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file({"donor_weight_rows": selected, "input_basis": basis}, str(output), metadata=metadata)
    return {
        "path": str(output),
        "bytes": int(output.stat().st_size),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "selected_weight_sha256_float32": _sha256_tensor(selected),
        "selected_weight_shape": list(selected.shape),
        "input_basis_shape": list(basis.shape),
    }


def run(
    source_path: str | Path,
    *,
    anatomy_path: str | Path = ROOT / "results" / "qwen-neural-anatomy-v4.json",
    output: str | Path = ROOT / "results" / "qwen-neural-router-capability-v1.json",
    ranking_output: str | Path = ROOT / "results" / "qwen-neural-candidate-ranking-v1.json",
    bundle_output: str | Path = ROOT / "results" / "qwen-neural-router-subspace-v1.safetensors",
    config_path: str | Path = ROOT / "configs" / "v0_tiny.json",
    noise_std: float = 0.20,
    threshold: float = 0.90,
) -> dict[str, Any]:
    source_root = Path(source_path).expanduser().resolve()
    anatomy_path = Path(anatomy_path)
    output = Path(output)
    ranking_output = Path(ranking_output)
    bundle_output = Path(bundle_output)
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    if not 0.0 < noise_std < 1.0:
        raise ValueError("noise_std must be in (0, 1)")
    if not 0.5 < threshold < 1.0:
        raise ValueError("threshold must be in (0.5, 1)")

    scans = []
    loaded: dict[int, tuple[torch.Tensor, str, str]] = {}
    for layer in ROUTER_LAYERS:
        weight, shard, tensor_name = _load_router(source_root, layer)
        loaded[layer] = (weight, shard, tensor_name)
        scans.append(_randomized_scan(weight, layer=layer, seed=20260910 + layer))
    ranking = _build_candidate_ranking(anatomy_path, scans)
    selected_layer = int(ranking["selected_candidate"]["layer"])
    weight, shard, tensor_name = loaded[selected_layer]
    basis = top_right_singular_basis(weight, 96)
    pair = select_balanced_pair(weight, basis)
    compact_weight = weight @ basis

    functional_ladder: dict[str, Any] = {"selected_layer": selected_layer, "selected_rows": list(pair), "steps": {}}
    probe_generator = torch.Generator(device="cpu").manual_seed(1701)
    probe_inputs = torch.randn(64, 2560, generator=probe_generator)
    identity = torch.eye(2560)
    functional_ladder["steps"]["D0_original_extracted"] = {
        "representation": "actual BF16 router tensor read selectively and represented as FP32 CPU matrix",
        "source_payload_bytes_read": int(weight.numel() * 2),
        "donor_parameters_available": int(weight.numel()),
    }
    functional_ladder["steps"]["D1_standalone_reproduction"] = {
        "metrics": router_functional_metrics(weight, identity, selected_rows=tuple(range(weight.shape[0])), inputs=probe_inputs[:8]),
        "status": "EQUIVALENT_WITHIN_FLOAT32_REFERENCE_TOLERANCE",
        "reference": "explicit x @ W.T",
    }
    # D2 is an algebraically identical fused affine parameterization.  It is
    # intentionally measured rather than described as an untested identity.
    direct = probe_inputs @ weight.transpose(0, 1)
    fused = nn.functional.linear(probe_inputs, weight)
    functional_ladder["steps"]["D2_algebraic_reparameterization"] = {
        "max_absolute_error": float((direct - fused).abs().max()),
        "relative_l2_error": float(torch.linalg.vector_norm(direct - fused) / torch.linalg.vector_norm(direct).clamp_min(1e-30)),
        "status": "EQUIVALENT_WITHIN_FLOAT32_REFERENCE_TOLERANCE",
        "transformation": "matmul to torch.nn.functional.linear; no learned values changed",
    }
    for rank in (32, 64, 96):
        rank_basis = top_right_singular_basis(weight, rank)
        functional_ladder["steps"][f"D3_compression_rank_{rank}"] = {
            "metrics": router_functional_metrics(weight, rank_basis, selected_rows=pair, inputs=probe_inputs),
            "retained_donor_parameters": int(weight[list(pair)].numel()),
            "discarded_donor_parameters": int(weight.numel() - weight[list(pair)].numel()),
            "analytic_basis_parameters": int(rank_basis.numel()),
        }
    functional_ladder["steps"]["D4_width_conversion_rank_96"] = {
        "metrics": router_functional_metrics(weight, basis, selected_rows=pair, inputs=probe_inputs),
        "input_contract": {"from": 2560, "to": 96, "method": "top-right-singular-vectors of full trained router"},
        "trainable_parameters": 0,
        "analytic_parameters": int(basis.numel()),
    }
    bundle = _save_selected_bundle(bundle_output, weight, basis, pair, layer=selected_layer, shard=shard, tensor_name=tensor_name)

    seeds: dict[str, Any] = {}
    same_interface_accuracies: dict[str, list[float]] = {name: [] for name in ("actual", "random", "shuffled", "zero")}
    threshold_records: list[dict[str, Any]] = []
    for seed in SEEDS:
        train_x, train_y = _make_binary_dataset(weight, pair, basis, count_per_class=512, noise_std=noise_std, seed=seed + 1000)
        test_x, test_y = _make_binary_dataset(weight, pair, basis, count_per_class=1024, noise_std=noise_std, seed=seed + 2000)
        # A fixed signed permutation is a deliberate shifted socket.  It is
        # not silently inverted by the graft; this exposes interface
        # dependence rather than awarding an artificial semantic-transfer win.
        permutation = torch.roll(torch.arange(96), shifts=7)
        shifted_test_x = test_x[:, permutation] * 0.9
        arms: dict[str, Any] = {}
        for variant in ("actual", "random", "shuffled", "zero"):
            organ = QwenRouterOrgan(
                weight,
                input_basis=basis,
                selected_rows=pair,
                donor_variant=variant,
                variant_seed=seed + 5000,
            )
            with torch.no_grad():
                same_logits = organ(test_x)
                shifted_logits = organ(shifted_test_x)
            same = _accuracy(same_logits, test_y)
            shifted = _accuracy(shifted_logits, test_y)
            same_interface_accuracies[variant].append(same)
            arms[variant] = {
                "same_interface_accuracy": same,
                "shifted_interface_accuracy": shifted,
                "same_minus_shifted": same - shifted,
                "donor_core_parameters": int(organ.donor_parameter_count),
                "donor_core_frozen": True,
                "analytic_basis_parameters": int(organ.analytic_parameter_count),
                "trainable_parameters": int(organ.trainable_parameter_count),
            }

        actual_factory = lambda: QwenRouterOrgan(
            weight,
            input_basis=basis,
            selected_rows=pair,
            donor_variant="actual",
            variant_seed=seed + 5000,
            trainable_repair_rank=1,
        )
        random_factory = lambda: QwenRouterOrgan(
            weight,
            input_basis=basis,
            selected_rows=pair,
            donor_variant="random",
            variant_seed=seed + 5000,
            trainable_repair_rank=1,
        )
        native_factory = lambda: nn.Linear(96, 2)
        matched_factory = lambda: _FreshMatchedRouter()
        curves = {
            "actual_donor_plus_rank1_repair": _donor_curve(actual_factory, train_x, train_y, test_x, test_y, seed=seed + 6100),
            "random_donor_plus_rank1_repair": _donor_curve(random_factory, train_x, train_y, test_x, test_y, seed=seed + 6200),
            "fresh_native_linear": _fresh_curve(native_factory, train_x, train_y, test_x, test_y, seed=seed + 6300),
            "fresh_matched_repair_budget": _fresh_curve(matched_factory, train_x, train_y, test_x, test_y, seed=seed + 6400),
        }
        for arm_name, curve in curves.items():
            threshold_records.append(
                {
                    "seed": seed,
                    "arm": arm_name,
                    "threshold": threshold,
                    "steps_to_threshold": _first_threshold(curve, threshold),
                    "curve": curve,
                }
            )
        seeds[str(seed)] = {
            "dataset": {
                "train_examples": int(train_x.shape[0]),
                "test_examples": int(test_x.shape[0]),
                "noise_std": noise_std,
                "label_definition": f"prototype around selected donor row {pair[0]} versus row {pair[1]}",
                "same_interface": "bus = donor_hidden @ rank96_basis",
                "shifted_interface": "same bus columns cyclically permuted by 7 and scaled by 0.9",
            },
            "arms_at_zero_steps": arms,
            "curves": curves,
        }

    remora_socket = _remora_socket_check(weight, basis, pair, Path(config_path))
    actual_thresholds = [row["steps_to_threshold"] for row in threshold_records if row["arm"] == "actual_donor_plus_rank1_repair" and row["steps_to_threshold"] is not None]
    matched_thresholds = [row["steps_to_threshold"] for row in threshold_records if row["arm"] == "fresh_matched_repair_budget" and row["steps_to_threshold"] is not None]
    random_zero = [seeds[str(seed)]["arms_at_zero_steps"]["random"]["same_interface_accuracy"] for seed in SEEDS]
    actual_zero = [seeds[str(seed)]["arms_at_zero_steps"]["actual"]["same_interface_accuracy"] for seed in SEEDS]
    result = {
        "schema": "remora-qwen-neural-router-capability-result-v1",
        "experiment_family": "QWEN-DONOR-CAPABILITY-001+",
        "source": {
            "repository": DONOR_REPOSITORY,
            "revision": DONOR_REVISION,
            "path": str(source_root),
            "selected_layer": selected_layer,
            "source_shard": shard,
            "source_tensor": tensor_name,
            "source_tensor_payload_bytes": int(weight.numel() * 2),
            "source_tensor_materialized_bytes": int(weight.numel() * 2),
            "full_model_materialized": False,
            "model_loader_called": False,
        },
        "candidate_ranking": ranking,
        "selected_organ": {
            "family": "moe_router",
            "mode": "SUBMODULE_SUBSPACE_GRAFT",
            "selected_rows": list(pair),
            "selected_row_count": len(pair),
            "donor_parameters_preserved_unchanged": int(weight[list(pair)].numel()),
            "donor_parameters_analytically_transformed": 0,
            "donor_parameters_discarded": int(weight.numel() - weight[list(pair)].numel()),
            "analytic_basis_parameters": int(basis.numel()),
            "analytic_basis_bytes_fp32": int(basis.numel() * 4),
            "retained_source_payload_bytes_bf16": int(weight[list(pair)].numel() * 2),
            "active_router_flops_per_token_modeled": int(2 * 96),
            "bundle": bundle,
            "functional_preservation_ladder": functional_ladder,
        },
        "donor_information_scan": {
            "all_router_layers": scans,
            "selection_rule": ranking["selection_rule"],
            "interpretation": "MEASURED: actual row geometry is strongly separated from randomized/shuffled router controls on a donor-conditioned route probe. This is evidence for a trained routing partition, not proof of general donor capability.",
        },
        "capability_assay": {
            "task": "binary held-out routing geometry around two automatically selected trained Qwen router rows",
            "threshold": threshold,
            "seeds": list(SEEDS),
            "per_seed": seeds,
            "minimum_training_curve": threshold_records,
            "zero_step_summary": {
                "actual_values": actual_zero,
                "random_values": random_zero,
                "actual_mean": sum(actual_zero) / len(actual_zero),
                "random_mean": sum(random_zero) / len(random_zero),
                "actual_minus_random_mean": sum(actual_zero) / len(actual_zero) - sum(random_zero) / len(random_zero),
            },
            "assimilation_advantage": {
                "definition": "fresh matched-budget steps/tokens/wall time to threshold divided by donor-graft steps/tokens/wall time",
                "donor_steps_to_threshold": actual_thresholds,
                "fresh_matched_steps_to_threshold": matched_thresholds,
                "steps_ratio": "UNBOUNDED_OR_UNDEFINED_WHEN_DONOR_REACHES_THRESHOLD_AT_ZERO_STEPS" if any(value == 0 for value in actual_thresholds) else None,
                "donor_training_avoidance": "MEASURED only as threshold steps avoided; original Qwen training compute remains UNMEASURED",
            },
        },
        "remora_attachment": remora_socket,
        "labels": {
            "MEASURED": [
                "selective source tensor bytes",
                "trained/random/shuffled/zero route accuracy",
                "ranked compression energy and function errors",
                "zero-step and repair curves",
                "Remora socket forward and causal logit deltas",
            ],
            "DERIVED": [
                "selected/discarded parameter counts",
                "analytic basis parameter count",
                "threshold steps and zero-step differences",
            ],
            "MODELED": ["active router FLOPs per token"],
            "UNMEASURED": ["original Qwen training compute avoided"],
            "HYPOTHESIS": ["router geometry can be a useful first foreign capability organ"],
        },
        "promotion_state": "CANDIDATE_ONLY",
        "interpretation": "MEASURED: a real trained Qwen router subspace can be selectively retained, analytically converted to the Remora 96-wide routing socket, and attached with zero learned interface parameters. The donor-conditioned routing assay is the first capability-oriented test; it does not yet establish broad language capability assimilation or economic recovery of Qwen pretraining.",
        "runtime": runtime_context(torch.device("cpu")),
    }
    write_json(ranking_output, ranking)
    write_json(output, result)

    record_experiment(
        ROOT,
        "QWEN-DONOR-SCAN-001",
        "A smaller closed foreign subcircuit with actual trained-vs-random signal should be a better assimilation target than the prior layer-0 shared expert.",
        "Read only the 48 Qwen router tensors, measure donor-conditioned row geometry and rank-96 analytic conversion, and rank against an unmeasured GDN head and hyperconnection candidate.",
        "One measured candidate has strong actual-vs-random separation after compression and is selected automatically without using the prior shared-expert target.",
        "The measured router candidates show no trained-vs-random separation, or an unmeasured candidate is allowed to outrank measured evidence.",
        f"python -m experiments.donor_router_capability --source {source_root}",
        20260910,
        {"selected_candidate": ranking["selected_candidate"], "candidate_count": len(ranking["candidates"]), "router_scan_layers": len(scans)},
        "MEASURED: router scan completed; ranking is a triage decision and not a promotion claim.",
        "Run the selected organ through the zero-step and micro-repair capability assay with donor-core destruction controls.",
        hardware={**runtime_context(torch.device("cpu")), "mode": "bounded_router_payload_scan", "source_tensor_payloads_materialized_only": True},
    )
    if result["capability_assay"]["zero_step_summary"]["actual_minus_random_mean"] <= 0.05:
        record_failure(
            ROOT,
            "QWEN-DONOR-CAPABILITY-001",
            ModelConfig.from_json(config_path).to_dict(),
            result["capability_assay"]["per_seed"]["7"]["dataset"],
            7,
            "The selected trained router did not clear the predeclared 0.05 mean zero-step advantage over the randomized donor core.",
            "The donor-conditioned row probe may be too weak or the router's trained geometry may not transfer through the analytic basis as expected.",
            "Resurrect only after a task-matched GatedDeltaNet or native-manifold assay is available; do not promote this router on a small margin.",
            runtime=runtime_context(torch.device("cpu")),
        )
    record_experiment(
        ROOT,
        "QWEN-DONOR-CAPABILITY-001",
        "Actual Qwen-trained router rows should provide a zero-step or lower-repair head start on the routing capability they encode.",
        "Select the highest measured router candidate, retain two trained rows, convert 2560->96 with a fixed rank-96 SVD basis, attach to blocks.1.experts.router, and compare actual/random/shuffled/zero/fresh controls over 0..64 steps.",
        "Actual donor beats destroyed cores at zero steps, reaches the fixed 0.90 threshold with no more repair than the fresh matched control, and the foreign core causes a causal Remora-socket effect.",
        "Actual donor fails to beat random/shuffled controls, needs more assimilation than fresh matched control, shifted interface destroys all utility, or the Remora socket cannot execute.",
        f"python -m experiments.donor_router_capability --source {source_root}",
        7,
        {
            "selected_layer": selected_layer,
            "selected_rows": list(pair),
            "zero_step_summary": result["capability_assay"]["zero_step_summary"],
            "threshold": threshold,
            "remora_socket": remora_socket,
            "promotion_state": result["promotion_state"],
        },
        result["interpretation"],
        "If the router is only a donor-conditioned probe win, keep it as a negative/partial result and run a task-matched GatedDeltaNet head transplant; do not scale Remora.",
        hardware={**runtime_context(torch.device("cpu")), "mode": "router_subspace_graft", "source_model_materialized": False},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan and graft a bounded trained Qwen router subcircuit.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--anatomy", default=str(ROOT / "results" / "qwen-neural-anatomy-v4.json"))
    parser.add_argument("--output", default=str(ROOT / "results" / "qwen-neural-router-capability-v1.json"))
    parser.add_argument("--ranking-output", default=str(ROOT / "results" / "qwen-neural-candidate-ranking-v1.json"))
    parser.add_argument("--bundle-output", default=str(ROOT / "results" / "qwen-neural-router-subspace-v1.safetensors"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "v0_tiny.json"))
    parser.add_argument("--noise-std", type=float, default=0.20)
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()
    result = run(
        args.source,
        anatomy_path=args.anatomy,
        output=args.output,
        ranking_output=args.ranking_output,
        bundle_output=args.bundle_output,
        config_path=args.config,
        noise_std=args.noise_std,
        threshold=args.threshold,
    )
    print(json.dumps({
        "output": str(args.output),
        "ranking_output": str(args.ranking_output),
        "selected_candidate": result["candidate_ranking"]["selected_candidate"],
        "selected_rows": result["selected_organ"]["selected_rows"],
        "zero_step_summary": result["capability_assay"]["zero_step_summary"],
        "remora_attachment": result["remora_attachment"],
        "promotion_state": result["promotion_state"],
    }, indent=2))


if __name__ == "__main__":
    main()
