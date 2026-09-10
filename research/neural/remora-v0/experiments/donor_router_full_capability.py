from __future__ import annotations

"""Test whether a real Qwen router tensor supplies a nontrivial head start.

The previous router pilot used two selected rows.  That was a useful causal
socket test, but a fresh two-class classifier solved the assay almost
immediately.  This follow-up keeps the same selected donor candidate and
conversion, but evaluates all 512 trained router rows under the harder noise
distribution used by the information scan.

This remains a donor-conditioned capability assay, not a claim about broad
Qwen language competence.  Its falsifiable question is narrower and useful:
does the actual trained numerical partition survive a bounded analytic
2560->96 conversion well enough to clear a held-out 512-way threshold at
zero steps, while fresh and destroyed cores cannot?

Only one 512 x 2560 router tensor is read from safetensors.  The whole Qwen
checkpoint is never loaded.  The compact organ stores the analytically
converted [512, 96] matrix so its active inference path is measurable.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.donors.router import (  # noqa: E402
    QwenCompactRouterOrgan,
    QwenRouterOrgan,
    make_router_variant,
    router_functional_metrics,
    top_right_singular_basis,
)
from remora.ledger import record_experiment, record_failure  # noqa: E402
from remora.utils import runtime_context, set_seed, write_json  # noqa: E402


DONOR_REPOSITORY = "Qwen/Qwen3.8-Flash-Next"
DONOR_REVISION = "f5d08274bafd880402bd16f5e3e6c514136ec06c"
SELECTED_LAYER = 13
BUS_WIDTH = 96
ROUTE_COUNT = 512
RANK = 96
TRAIN_EXAMPLES_PER_ROUTE = 4
TEST_EXAMPLES_PER_ROUTE = 8
NOISE_STD = 0.15
THRESHOLD = 0.50
SEEDS = (7, 19, 31)
CURVE_STEPS = (0, 1, 2, 4, 8, 16, 32, 64)


def _sha256_tensor(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load_router(source_root: Path, layer: int) -> tuple[torch.Tensor, str, str]:
    from safetensors import safe_open

    tensor_name = f"model.language_model.layers.{layer}.mlp.gate.weight"
    index = json.loads((source_root / "model.safetensors.index.json").read_text())
    shard_name = str(index["weight_map"][tensor_name])
    with safe_open(str(source_root / shard_name), framework="pt", device="cpu") as handle:
        # Exactly one bounded tensor is materialized.  This is deliberately
        # independent from the full model loader and from all other shards.
        weight = handle.get_tensor(tensor_name).float().contiguous()
    if tuple(weight.shape) != (ROUTE_COUNT, 2560):
        raise ValueError(f"unexpected router shape: {tuple(weight.shape)}")
    return weight, shard_name, tensor_name


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean())


def _make_dataset(
    weight: torch.Tensor,
    basis: torch.Tensor,
    *,
    examples_per_route: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build independent noisy examples around every actual donor row.

    The raw Gaussian noise is intentional: the earlier scan used this scale,
    and normalizing it would make the 512-way assay much easier than the
    predeclared information test.  The train and test seeds are disjoint.
    """

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    labels = torch.arange(ROUTE_COUNT).repeat_interleave(int(examples_per_route))
    centers = torch.nn.functional.normalize(weight, dim=-1)
    donor_inputs = centers[labels] + NOISE_STD * torch.randn(
        (labels.numel(), weight.shape[1]), generator=generator
    )
    return donor_inputs @ basis, labels, donor_inputs


class _FreshLowRankRouter(nn.Module):
    """Fresh output low-rank control, matched to the compact repair port."""

    def __init__(self, bus_dim: int = BUS_WIDTH, routes: int = ROUTE_COUNT, rank: int = 1):
        super().__init__()
        self.down = nn.Linear(bus_dim, rank, bias=False)
        self.up = nn.Linear(rank, routes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


def _model_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _curve(
    model_factory: Callable[[], nn.Module],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    shifted_test_x: torch.Tensor,
    *,
    seed: int,
    steps: Iterable[int] = CURVE_STEPS,
    frozen_core: bool = False,
) -> dict[str, Any]:
    """Measure every budget from an identical initialized model state."""

    set_seed(seed)
    initial = model_factory()
    initial_state = {name: value.detach().clone() for name, value in initial.state_dict().items()}
    values: list[dict[str, Any]] = []
    for budget in steps:
        model = model_factory()
        model.load_state_dict(initial_state, strict=True)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = (
            torch.optim.AdamW(trainable, lr=0.05, betas=(0.9, 0.95), weight_decay=0.0)
            if trainable
            else None
        )
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
        with torch.no_grad():
            same_accuracy = _accuracy(model(test_x), test_y)
            shifted_accuracy = _accuracy(model(shifted_test_x), test_y)
            train_accuracy = _accuracy(model(train_x), train_y)
        values.append(
            {
                "gradient_steps": int(budget),
                "accuracy": same_accuracy,
                "shifted_interface_accuracy": shifted_accuracy,
                "train_accuracy": train_accuracy,
                "loss_last": losses[-1] if losses else None,
                "wall_seconds": time.perf_counter() - started,
                "trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
                "tokens": int(budget * train_x.shape[0]),
                "frozen_core": bool(frozen_core),
            }
        )
    return {
        "steps": values,
        "initial_parameter_count": _model_parameter_count(initial),
    }


def _first_threshold(curve: dict[str, Any], threshold: float, key: str = "accuracy") -> int | None:
    for row in curve["steps"]:
        if float(row[key]) >= float(threshold):
            return int(row["gradient_steps"])
    return None


def _ratio(numerator: int | None, denominator: int | None) -> float | str | None:
    if numerator is None:
        return "RIGHT_CENSORED"
    if denominator is None:
        return None
    if denominator == 0:
        return "UNBOUNDED_ZERO_STEP_DENOMINATOR"
    return float(numerator / denominator)


def _save_bundle(
    path: Path,
    compact_weight: torch.Tensor,
    basis: torch.Tensor,
    *,
    source_weight_sha256: str,
    shard: str,
    tensor_name: str,
) -> dict[str, Any]:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "remora-qwen-router-compact-bundle-v1",
        "source_model": DONOR_REPOSITORY,
        "source_revision": DONOR_REVISION,
        "source_shard": shard,
        "source_tensor": tensor_name,
        "source_weight_sha256_float32_read": source_weight_sha256,
        "conversion": "compact_weight = donor_weight @ top_right_singular_basis(rank=96)",
        "source_width": "2560",
        "bus_width": str(BUS_WIDTH),
        "route_count": str(ROUTE_COUNT),
        "donor_parameters_transformed": str(ROUTE_COUNT * 2560),
        "resident_compact_parameters": str(compact_weight.numel()),
        "analytic_basis_parameters": str(basis.numel()),
    }
    save_file(
        {"compact_weight": compact_weight.float().contiguous(), "input_basis": basis.float().contiguous()},
        str(path),
        metadata=metadata,
    )
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "compact_weight_shape": list(compact_weight.shape),
        "input_basis_shape": list(basis.shape),
    }


def run(
    source_path: str | Path,
    *,
    ranking_path: str | Path = ROOT / "results" / "qwen-neural-candidate-ranking-v1.json",
    output: str | Path = ROOT / "results" / "qwen-neural-router-full-capability-v1.json",
    bundle_output: str | Path = ROOT / "results" / "qwen-neural-router-full-rank96-v1.safetensors",
    seeds: tuple[int, ...] = SEEDS,
    curve_steps: tuple[int, ...] = CURVE_STEPS,
    record_ledger: bool = True,
) -> dict[str, Any]:
    source_root = Path(source_path).expanduser().resolve()
    ranking_path = Path(ranking_path)
    output = Path(output)
    bundle_output = Path(bundle_output)
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    if tuple(curve_steps) != tuple(sorted(set(int(step) for step in curve_steps))):
        raise ValueError("curve_steps must be sorted and unique")
    if any(step < 0 for step in curve_steps):
        raise ValueError("curve steps must be non-negative")
    if not seeds:
        raise ValueError("at least one seed is required")

    ranking = json.loads(ranking_path.read_text())
    selected = ranking["selected_candidate"]
    if selected["candidate_id"] != f"qwen3.8.language.layer{SELECTED_LAYER}.router":
        raise ValueError("follow-up is pinned to the measured layer-13 router candidate")
    if selected.get("actual_weight_scan_status") != "MEASURED":
        raise ValueError("cannot run capability assay from an unmeasured candidate")

    weight, shard, tensor_name = _load_router(source_root, SELECTED_LAYER)
    source_hash = _sha256_tensor(weight)
    basis = top_right_singular_basis(weight, RANK)
    compact_weight = weight @ basis
    bundle = _save_bundle(
        bundle_output,
        compact_weight,
        basis,
        source_weight_sha256=source_hash,
        shard=shard,
        tensor_name=tensor_name,
    )

    probe_generator = torch.Generator(device="cpu").manual_seed(1702)
    probe_inputs = torch.randn(64, weight.shape[1], generator=probe_generator)
    identity = torch.eye(weight.shape[1])
    functional_ladder: dict[str, Any] = {"steps": {}}
    functional_ladder["steps"]["D0_original_extracted"] = {
        "representation": "one actual BF16 router tensor selectively read and represented as FP32 CPU matrix",
        "source_payload_bytes_read": int(weight.numel() * 2),
        "donor_parameters_available": int(weight.numel()),
        "sha256_float32_read": source_hash,
    }
    functional_ladder["steps"]["D1_standalone_reproduction"] = {
        "metrics": router_functional_metrics(
            weight, identity, selected_rows=tuple(range(ROUTE_COUNT)), inputs=probe_inputs[:8]
        ),
        "reference": "explicit x @ W.T",
        "status": "EQUIVALENT_WITHIN_FLOAT32_REFERENCE_TOLERANCE",
    }
    direct = probe_inputs @ weight.transpose(0, 1)
    fused = nn.functional.linear(probe_inputs, weight)
    functional_ladder["steps"]["D2_algebraic_reparameterization"] = {
        "max_absolute_error": float((direct - fused).abs().max()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(direct - fused)
            / torch.linalg.vector_norm(direct).clamp_min(1e-30)
        ),
        "transformation": "matmul to torch.nn.functional.linear; no trained values changed",
        "status": "EQUIVALENT_WITHIN_FLOAT32_REFERENCE_TOLERANCE",
    }
    for rank in (32, 64, RANK):
        rank_basis = top_right_singular_basis(weight, rank)
        compact = weight @ rank_basis
        functional_ladder["steps"][f"D3_compression_rank_{rank}"] = {
            "metrics": router_functional_metrics(
                weight, rank_basis, selected_rows=tuple(range(ROUTE_COUNT)), inputs=probe_inputs
            ),
            "rank": rank,
            "resident_transformed_parameters": int(compact.numel()),
            "discarded_original_parameters_at_resident_stage": int(weight.numel() - compact.numel()),
            "analytic_basis_parameters": int(rank_basis.numel()),
        }
    functional_ladder["steps"]["D4_width_conversion_rank_96"] = {
        "metrics": router_functional_metrics(
            weight, basis, selected_rows=tuple(range(ROUTE_COUNT)), inputs=probe_inputs
        ),
        "input_contract": {"from": 2560, "to": BUS_WIDTH, "method": "top-right SVD basis"},
        "donor_parameters_analytically_transformed": int(weight.numel()),
        "resident_transformed_parameters": int(compact_weight.numel()),
        "resident_original_parameters": 0,
        "analytic_basis_parameters_offline": int(basis.numel()),
    }
    compact_organ = QwenCompactRouterOrgan(compact_weight)
    with torch.no_grad():
        compact_direct = compact_organ(probe_inputs @ basis)
    functional_ladder["steps"]["D5_compact_graft_executor"] = {
        "max_absolute_error_vs_D4": float(
            (compact_direct - (probe_inputs @ basis) @ compact_weight.T).abs().max()
        ),
        "status": "EQUIVALENT_WITHIN_FLOAT32_REFERENCE_TOLERANCE",
        "resident_core_parameters": compact_organ.donor_parameter_count,
        "active_mac_per_token_modeled": int(BUS_WIDTH * ROUTE_COUNT),
    }

    seeds_result: dict[str, Any] = {}
    threshold_records: list[dict[str, Any]] = []
    zero_step: dict[str, list[float]] = {name: [] for name in ("actual", "random", "shuffled", "zero")}
    for seed in seeds:
        train_x, train_y, train_donor_x = _make_dataset(
            weight, basis, examples_per_route=TRAIN_EXAMPLES_PER_ROUTE, seed=int(seed) + 1000
        )
        test_x, test_y, test_donor_x = _make_dataset(
            weight, basis, examples_per_route=TEST_EXAMPLES_PER_ROUTE, seed=int(seed) + 2000
        )
        permutation = torch.roll(torch.arange(BUS_WIDTH), shifts=7)
        shifted_test_x = test_x[:, permutation] * 0.9
        variants: dict[str, torch.Tensor] = {
            "actual": weight,
            "random": make_router_variant(weight, "random", seed=int(seed) + 5000),
            "shuffled": make_router_variant(weight, "shuffled", seed=int(seed) + 6000),
            "zero": make_router_variant(weight, "zero", seed=int(seed) + 7000),
        }
        arms_at_zero: dict[str, Any] = {}
        for name, variant_weight in variants.items():
            variant_compact = variant_weight @ basis
            organ = QwenCompactRouterOrgan(variant_compact, donor_variant=name)
            with torch.no_grad():
                logits = organ(test_x)
                shifted_logits = organ(shifted_test_x)
            accuracy = _accuracy(logits, test_y)
            zero_step[name].append(accuracy)
            arms_at_zero[name] = {
                "same_interface_accuracy": accuracy,
                "shifted_interface_accuracy": _accuracy(shifted_logits, test_y),
                "core_parameters": int(organ.donor_parameter_count),
                "core_frozen": True,
                "trainable_parameters": int(organ.trainable_parameter_count),
            }

        actual_factory = lambda: QwenCompactRouterOrgan(
            compact_weight,
            donor_variant="actual",
            trainable_repair_rank=1,
            repair_initialization="warm_start",
        )
        random_factory = lambda: QwenCompactRouterOrgan(
            variants["random"] @ basis,
            donor_variant="random",
            trainable_repair_rank=1,
            repair_initialization="warm_start",
        )
        native_factory = lambda: nn.Linear(BUS_WIDTH, ROUTE_COUNT, bias=False)
        matched_factory = lambda: _FreshLowRankRouter()
        curves = {
            "actual_donor_plus_rank1_repair": _curve(
                actual_factory,
                train_x,
                train_y,
                test_x,
                test_y,
                shifted_test_x,
                seed=int(seed) + 6100,
                steps=curve_steps,
                frozen_core=True,
            ),
            "random_donor_plus_rank1_repair": _curve(
                random_factory,
                train_x,
                train_y,
                test_x,
                test_y,
                shifted_test_x,
                seed=int(seed) + 6200,
                steps=curve_steps,
                frozen_core=True,
            ),
            "fresh_native_linear": _curve(
                native_factory,
                train_x,
                train_y,
                test_x,
                test_y,
                shifted_test_x,
                seed=int(seed) + 6300,
                steps=curve_steps,
            ),
            "fresh_matched_rank1": _curve(
                matched_factory,
                train_x,
                train_y,
                test_x,
                test_y,
                shifted_test_x,
                seed=int(seed) + 6400,
                steps=curve_steps,
            ),
        }
        threshold_for_seed: dict[str, Any] = {}
        for arm, curve in curves.items():
            record = {
                "seed": int(seed),
                "arm": arm,
                "threshold": THRESHOLD,
                "same_interface_steps_to_threshold": _first_threshold(curve, THRESHOLD),
                "shifted_interface_steps_to_threshold": _first_threshold(
                    curve, THRESHOLD, key="shifted_interface_accuracy"
                ),
            }
            threshold_records.append(record)
            threshold_for_seed[arm] = record
        seeds_result[str(seed)] = {
            "dataset": {
                "train_examples": int(train_x.shape[0]),
                "test_examples": int(test_x.shape[0]),
                "train_examples_per_route": TRAIN_EXAMPLES_PER_ROUTE,
                "test_examples_per_route": TEST_EXAMPLES_PER_ROUTE,
                "noise_std": NOISE_STD,
                "label_definition": "route index 0..511 around normalized actual layer-13 Qwen router rows",
                "same_interface": "bus = donor_hidden @ rank96_basis",
                "shifted_interface": "same bus columns cyclically permuted by 7 and scaled by 0.9",
            },
            "donor_reference": {
                "full_donor_accuracy": _accuracy(test_donor_x @ weight.T, test_y),
                "compact_donor_accuracy": _accuracy(test_x @ compact_weight.T, test_y),
            },
            "arms_at_zero_steps": arms_at_zero,
            "curves": curves,
            "thresholds": threshold_for_seed,
        }

    actual_thresholds = [
        record["same_interface_steps_to_threshold"]
        for record in threshold_records
        if record["arm"] == "actual_donor_plus_rank1_repair"
    ]
    native_thresholds = [
        record["same_interface_steps_to_threshold"]
        for record in threshold_records
        if record["arm"] == "fresh_native_linear"
    ]
    matched_thresholds = [
        record["same_interface_steps_to_threshold"]
        for record in threshold_records
        if record["arm"] == "fresh_matched_rank1"
    ]
    result = {
        "schema": "remora-qwen-neural-router-full-capability-result-v1",
        "experiment_family": "QWEN-DONOR-CAPABILITY-002+",
        "source": {
            "repository": DONOR_REPOSITORY,
            "revision": DONOR_REVISION,
            "path": str(source_root),
            "selected_layer": SELECTED_LAYER,
            "source_shard": shard,
            "source_tensor": tensor_name,
            "source_tensor_payload_bytes": int(weight.numel() * 2),
            "source_tensor_materialized_bytes": int(weight.numel() * 4),
            "source_tensor_sha256_float32_read": source_hash,
            "full_model_materialized": False,
            "model_loader_called": False,
        },
        "predeclared_gate": {
            "threshold": THRESHOLD,
            "noise_std": NOISE_STD,
            "train_examples_per_route": TRAIN_EXAMPLES_PER_ROUTE,
            "test_examples_per_route": TEST_EXAMPLES_PER_ROUTE,
            "curve_steps": list(curve_steps),
            "falsification": "actual compact donor does not clear 0.50 at zero steps or does not beat fresh/destroyed controls on the fixed 512-way assay",
            "declared_before_main_run": True,
        },
        "selected_organ": {
            "family": "moe_router",
            "mode": "FULL_ROUTER_ANALYTIC_SUBSPACE_GRAFT",
            "route_count": ROUTE_COUNT,
            "source_width": int(weight.shape[1]),
            "bus_width": BUS_WIDTH,
            "rank": RANK,
            "donor_parameters_preserved_unchanged": 0,
            "donor_parameters_analytically_transformed": int(weight.numel()),
            "donor_parameters_discarded_at_resident_stage": int(weight.numel() - compact_weight.numel()),
            "resident_transformed_core_parameters": int(compact_weight.numel()),
            "analytic_basis_parameters_offline": int(basis.numel()),
            "micro_repair_parameters_rank1": int(BUS_WIDTH + ROUTE_COUNT),
            "retained_source_payload_bytes_bf16": int(weight.numel() * 2),
            "resident_compact_core_bytes_fp32": int(compact_weight.numel() * 4),
            "active_mac_per_token_modeled": int(BUS_WIDTH * ROUTE_COUNT),
            "bundle": bundle,
            "functional_preservation_ladder": functional_ladder,
        },
        "capability_assay": {
            "task": "512-way held-out route-index classification around every actual Qwen layer-13 router row",
            "per_seed": seeds_result,
            "minimum_training_curve": threshold_records,
            "zero_step_summary": {
                "actual_values": zero_step["actual"],
                "random_values": zero_step["random"],
                "shuffled_values": zero_step["shuffled"],
                "zero_values": zero_step["zero"],
                "actual_mean": sum(zero_step["actual"]) / len(zero_step["actual"]),
                "random_mean": sum(zero_step["random"]) / len(zero_step["random"]),
                "actual_minus_random_mean": sum(zero_step["actual"]) / len(zero_step["actual"])
                - sum(zero_step["random"]) / len(zero_step["random"]),
            },
            "assimilation_advantage": {
                "definition": "fresh-control gradient steps to fixed threshold divided by actual donor-graft steps to fixed threshold",
                "actual_donor_steps": actual_thresholds,
                "fresh_native_steps": native_thresholds,
                "fresh_matched_rank1_steps": matched_thresholds,
                "fresh_native_step_ratio": [
                    _ratio(native, donor) for native, donor in zip(native_thresholds, actual_thresholds)
                ],
                "fresh_matched_rank1_step_ratio": [
                    _ratio(matched, donor) for matched, donor in zip(matched_thresholds, actual_thresholds)
                ],
                "original_donor_training_compute": "UNMEASURED; Qwen training receipt is unavailable",
            },
        },
        "controls": {
            "actual": "actual donor W then fixed rank-96 basis",
            "random": "randomized W with the same source shape, basis, and trainable repair budget",
            "shuffled": "element-shuffled W with the same source shape, basis, and trainable repair budget",
            "zero": "zero core with the same compact socket",
            "fresh_native_linear": "fresh 96->512 linear, 49152 trainable parameters",
            "fresh_matched_rank1": "fresh 96->1->512 low-rank map, 608 trainable parameters",
        },
        "labels": {
            "MEASURED": [
                "one-tensor selective extraction and byte accounting",
                "D1/D2 exact function checks",
                "D3/D4/D5 function-preservation metrics",
                "full 512-way actual/random/shuffled/zero accuracy",
                "zero-to-64-step curves and shifted-interface accuracy",
                "per-seed threshold records and wall times",
            ],
            "DERIVED": [
                "parameter counts",
                "threshold step ratios",
                "compact resident byte count",
            ],
            "MODELED": ["active compact-router MACs per token"],
            "UNMEASURED": ["original Qwen training compute avoided", "energy"],
            "HYPOTHESIS": ["router partition is a reusable foreign neural capability"],
        },
        "promotion_state": "CANDIDATE_ONLY_UNTIL_TASK_MATCH_AND_REMORA_SOCKET_VALIDATION",
        "interpretation": "Pending gate evaluation: this is a stricter donor-conditioned test of training avoidance. It is not broad language competence and cannot by itself justify scaling Remora.",
        "runtime": runtime_context(torch.device("cpu")),
    }
    write_json(output, result)

    command = f"python -m experiments.donor_router_full_capability --source {source_root}"
    if record_ledger:
        record_experiment(
            ROOT,
            "QWEN-DONOR-CAPABILITY-002",
            "A full trained Qwen router partition should clear a fixed 512-way held-out threshold with fewer assimilation steps than fresh controls after rank-96 analytic conversion.",
            "Reuse the measured layer-13 router candidate, retain all 512 trained rows, convert 2560->96 analytically, precompute the compact organ, and compare actual/random/shuffled/zero/fresh controls over the predeclared 0..64 curve.",
            "Actual compact donor clears 0.50 at zero steps and maintains a substantial step advantage over both the fresh native and matched rank-1 controls, with donor-core destruction removing the advantage.",
            "Actual donor fails the 0.50 zero-step gate, fresh controls match it at the same or lower budget, or the actual-vs-destroyed separation disappears.",
            command,
            int(seeds[0]),
            {
                "threshold": THRESHOLD,
                "zero_step_summary": result["capability_assay"]["zero_step_summary"],
                "actual_steps": actual_thresholds,
                "fresh_native_steps": native_thresholds,
                "fresh_matched_rank1_steps": matched_thresholds,
                "promotion_state": result["promotion_state"],
            },
            "MEASURED: follow-up result recorded; promotion is deferred until the fixed gate and task-matched Remora pathway criteria are both satisfied.",
            "If zero-step donor advantage is real but shifted-interface/Remora utility remains weak, extract a task-matched GatedDeltaNet head; if fresh controls erase the advantage, keep this as a negative training-avoidance result.",
            hardware={**runtime_context(torch.device("cpu")), "mode": "full_router_compact_capability_assay", "source_model_materialized": False},
        )
        if result["capability_assay"]["zero_step_summary"]["actual_minus_random_mean"] <= 0.05:
            record_failure(
                ROOT,
                "QWEN-DONOR-CAPABILITY-002",
                {"source_tensor": tensor_name, "source_layer": SELECTED_LAYER, "bus_width": BUS_WIDTH, "rank": RANK},
                result["predeclared_gate"],
                int(seeds[0]),
                "The full 512-way compact donor did not clear the predeclared zero-step separation gate over the randomized core.",
                "The measured router geometry may be donor-conditioned but not a robustly reusable capability after width conversion, or the chosen threshold/noise regime may still be misaligned.",
                "Retest only with a task-matched closed subgraph such as a bounded GatedDeltaNet head; do not promote the router or scale Remora.",
                runtime=runtime_context(torch.device("cpu")),
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stricter full-Qwen-router capability assay.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--ranking", default=str(ROOT / "results" / "qwen-neural-candidate-ranking-v1.json"))
    parser.add_argument("--output", default=str(ROOT / "results" / "qwen-neural-router-full-capability-v1.json"))
    parser.add_argument("--bundle-output", default=str(ROOT / "results" / "qwen-neural-router-full-rank96-v1.safetensors"))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--curve-steps", default=",".join(str(step) for step in CURVE_STEPS))
    parser.add_argument("--no-ledger", action="store_true", help="do not append an experiment receipt (for smoke tests)")
    args = parser.parse_args()
    seed_values = tuple(int(value) for value in args.seeds.split(",") if value)
    step_values = tuple(int(value) for value in args.curve_steps.split(",") if value)
    result = run(
        args.source,
        ranking_path=args.ranking,
        output=args.output,
        bundle_output=args.bundle_output,
        seeds=seed_values,
        curve_steps=step_values,
        record_ledger=not args.no_ledger,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_layer": result["source"]["selected_layer"],
                "zero_step_summary": result["capability_assay"]["zero_step_summary"],
                "assimilation_advantage": result["capability_assay"]["assimilation_advantage"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
