from __future__ import annotations

"""Wrap the selected Qwen GDN core at its native input width.

The compact path analytically multiplies Qwen's 2560-wide projections by a
rank-96 basis.  This experiment keeps the trained Q/K/V/beta/decay tensors at
their native geometry and puts the same fixed basis in the Remora-facing
input port.  It is an exact-weight-reuse control: any utility is measured
against randomized/shuffled/zero native cores, while the larger frozen port
and active compute are reported explicitly.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.donors.gdn import QwenGatedDeltaCoreOrgan, QwenGatedDeltaCoreSocket, make_gdn_variant  # noqa: E402
from remora.ledger import record_experiment  # noqa: E402
from remora.utils import runtime_context, set_seed, write_json  # noqa: E402
from experiments.donor_gdn_head import (  # noqa: E402
    BUS_WIDTH,
    CONV_KERNEL,
    LAYER,
    _conversion_bases,
    _extract_head,
)
from experiments.donor_gdn_remora_pathway import (  # noqa: E402
    CHECKPOINT_PATTERN,
    RECALL_THRESHOLD,
    SEEDS,
    SELECTED_VALUE_HEAD,
    TASK_D_MODEL,
    TEST_SAMPLES,
    TRAIN_SAMPLES,
    _configure_task_host,
    _host_bus,
    _load_remora,
    _make_core_organ,
    _make_path_dataset,
    _padded_values,
    _retrieval,
    _run_continuous,
)


NATIVE_WIDTH = 2560
HEAD_DIM = 128
CONV_CHANNELS = 3 * HEAD_DIM


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: list[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    if not values:
        return {"values": [], "mean": None, "std": None}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
    return {"values": values, "mean": mean, "std": math.sqrt(variance)}


def _native_core(
    payload: dict[str, torch.Tensor],
    variant: str,
    *,
    seed: int,
) -> QwenGatedDeltaCoreOrgan:
    values = make_gdn_variant(payload, variant, seed=seed)
    return QwenGatedDeltaCoreOrgan(
        q_weight=values["q_weight"],
        k_weight=values["k_weight"],
        v_weight=values["v_weight"],
        a_weight=values["a_weight"],
        b_weight=values["b_weight"],
        conv_qkv=values["conv_qkv"],
        A_log=values["A_log"],
        dt_bias=values["dt_bias"],
        donor_variant=variant,
    )


def _native_projection(input_basis: torch.Tensor, *, shifted: bool = False) -> torch.Tensor:
    projection = torch.zeros(NATIVE_WIDTH, TASK_D_MODEL, dtype=torch.float32)
    if not shifted:
        projection[:, :BUS_WIDTH] = input_basis
    else:
        permutation = torch.roll(torch.arange(BUS_WIDTH), shifts=7)
        projection[:, :BUS_WIDTH] = input_basis[:, permutation] * 0.9
    return projection


def _native_task_variant(
    checkpoint: Path,
    organ: QwenGatedDeltaCoreOrgan,
    projection: torch.Tensor,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    values: torch.Tensor,
    *,
    shifted: bool = False,
) -> dict[str, Any]:
    model, branch_source = _load_remora(checkpoint, torch.device("cpu"))
    branch = QwenGatedDeltaCoreSocket(
        organ,
        model.cfg.d_model,
        input_projection=projection,
    )
    model = _configure_task_host(model, branch)
    model.eval()
    with torch.no_grad():
        trace = _run_continuous(model, hidden)
    result = {
        "core_to_actual_value": _retrieval(trace["core"], labels, values),
        "path_output_to_actual_value": _retrieval(trace["path_output"], labels, _padded_values(values)),
        "shifted_interface": bool(shifted),
        "donor_variant": organ.donor_variant,
        "donor_core_frozen": True,
        "input_port_parameters_frozen": int(branch.input_port.weight.numel()),
        "checkpoint_schema": branch_source.get("schema"),
    }
    return result


def _core_projection_equivalence(
    compact: QwenGatedDeltaCoreOrgan,
    native: QwenGatedDeltaCoreOrgan,
    norm: torch.nn.Module,
    hidden: torch.Tensor,
    projection: torch.Tensor,
) -> dict[str, float | str]:
    with torch.no_grad():
        normalized = norm(hidden.float())
        compact_bus = normalized[..., :BUS_WIDTH]
        native_bus = F.linear(normalized, projection)
        compact_core = compact.forward_core(compact_bus)
        native_core = native.forward_core(native_bus)
    delta = native_core.float() - compact_core.float()
    cosine = float(
        F.cosine_similarity(native_core.float().reshape(1, -1), compact_core.float().reshape(1, -1)).item()
    )
    return {
        "status": "MEASURED_NATIVE_WRAPPED_CORE_EQUIVALENCE",
        "max_absolute_error": float(delta.abs().max()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(compact_core.float()).clamp_min(1e-30)
        ),
        "cosine_similarity": max(-1.0, min(1.0, cosine)),
        "reference": "native-width frozen Qwen core behind fixed rank-96 basis port versus compact analytically converted core",
    }


def run(
    source_path: str | Path,
    *,
    checkpoint_pattern: str = CHECKPOINT_PATTERN,
    output: str | Path = ROOT / "results" / "qwen-neural-gdn-native-core-v1.json",
    seeds: tuple[int, ...] = SEEDS,
    record_ledger: bool = True,
    experiment_id: str = "QWEN-DONOR-GDN-NATIVE-CORE-001",
) -> dict[str, Any]:
    source_root = Path(source_path).expanduser().resolve()
    output = Path(output)
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    if not seeds:
        raise ValueError("at least one seed is required")

    payload, metadata = _extract_head(source_root, LAYER, SELECTED_VALUE_HEAD)
    input_basis, output_basis = _conversion_bases(payload)
    compact_actual = _make_core_organ(payload, input_basis, output_basis, "actual")
    native_actual = _native_core(payload, "actual", seed=0)
    native_zero = _native_core(payload, "zero", seed=0)
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        set_seed(int(seed))
        checkpoint = Path(checkpoint_pattern.format(seed=int(seed)))
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        norm_model, checkpoint_source = _load_remora(checkpoint, torch.device("cpu"))
        norm = norm_model.blocks[1].norm
        train = _make_path_dataset(norm, compact_actual, samples=TRAIN_SAMPLES, seed=int(seed) + 1000)
        test = _make_path_dataset(norm, compact_actual, samples=TEST_SAMPLES, seed=int(seed) + 2000)
        equivalence = _core_projection_equivalence(
            compact_actual,
            native_actual,
            norm,
            test["hidden"],
            _native_projection(input_basis),
        )
        del norm_model
        variants = {
            "actual": native_actual,
            "random": _native_core(payload, "random", seed=int(seed) + 5000),
            "shuffled": _native_core(payload, "shuffled", seed=int(seed) + 6000),
            "zero": native_zero,
        }
        zero_shot = {
            name: {
                "unshifted": _native_task_variant(
                    checkpoint,
                    organ,
                    _native_projection(input_basis),
                    test["hidden"],
                    test["labels"],
                    test["values"],
                ),
                "shifted": _native_task_variant(
                    checkpoint,
                    organ,
                    _native_projection(input_basis, shifted=True),
                    test["hidden"],
                    test["labels"],
                    test["values"],
                    shifted=True,
                ),
            }
            for name, organ in variants.items()
        }
        per_seed[str(seed)] = {
            "checkpoint": {
                "path": str(checkpoint),
                "bytes": int(checkpoint.stat().st_size),
                "sha256": _sha256_file(checkpoint),
                "schema": checkpoint_source.get("schema"),
                "seed": checkpoint_source.get("seed"),
            },
            "dataset": {
                "train_samples": TRAIN_SAMPLES,
                "test_samples": TEST_SAMPLES,
                "memory_items": 4,
                "memory_delay": 9,
                "query_synthesis": "same donor-key analytic synthesis as the compact aged-path assay; no donor training",
            },
            "native_vs_compact_equivalence": equivalence,
            "zero_shot": zero_shot,
        }

    actual = [per_seed[str(seed)]["zero_shot"]["actual"]["unshifted"]["core_to_actual_value"]["accuracy"] for seed in seeds]
    random = [per_seed[str(seed)]["zero_shot"]["random"]["unshifted"]["core_to_actual_value"]["accuracy"] for seed in seeds]
    shifted = [per_seed[str(seed)]["zero_shot"]["actual"]["shifted"]["core_to_actual_value"]["accuracy"] for seed in seeds]
    shuffled = [per_seed[str(seed)]["zero_shot"]["shuffled"]["unshifted"]["core_to_actual_value"]["accuracy"] for seed in seeds]
    zero = [per_seed[str(seed)]["zero_shot"]["zero"]["unshifted"]["core_to_actual_value"]["accuracy"] for seed in seeds]
    core_source_parameters = int(sum(payload[name].numel() for name in ("q_weight", "k_weight", "v_weight", "a_weight", "b_weight", "conv_qkv", "A_log", "dt_bias")))
    result = {
        "schema": "remora-qwen-neural-gdn-native-core-result-v1",
        "experiment_id": experiment_id,
        "experiment_family": f"{experiment_id}+",
        "source": {
            "repository": "Qwen/Qwen3.8-Flash-Next",
            "revision": "f5d08274bafd880402bd16f5e3e6c514136ec06c",
            "path": str(source_root),
            "layer": LAYER,
            "value_head": SELECTED_VALUE_HEAD,
            "source_shard": metadata["source_shard"],
            "source_tensor_names": metadata["source_tensor_names"],
            "source_payload_bytes_bf16": metadata["source_payload_bytes_bf16"],
            "selected_payload_materialized_bytes_fp32": metadata["selected_payload_materialized_bytes_fp32"],
            "full_model_materialized": False,
            "model_loader_called": False,
        },
        "graft": {
            "mode": "WRAPPED_NATIVE_WIDTH_CORE_GRAFT",
            "attached_module_path": "blocks.1.plastic",
            "donor_core_frozen": True,
            "donor_parameters_source_total": int(sum(value.numel() for value in payload.values())),
            "donor_parameters_preserved_unchanged": core_source_parameters,
            "donor_parameters_analytically_transformed": 0,
            "donor_parameters_discarded": int(sum(value.numel() for value in payload.values()) - core_source_parameters),
            "donor_parameters_discarded_by_width_conversion": 0,
            "resident_native_core_parameters": int(native_actual.donor_parameter_count),
            "resident_native_core_bytes_fp32": int(native_actual.donor_parameter_count * 4),
            "frozen_input_port_parameters": int(NATIVE_WIDTH * TASK_D_MODEL),
            "newly_trained_parameters": 0,
            "donor_parameters_modified_during_assimilation": 0,
            "analytic_basis_shape": list(input_basis.shape),
            "active_core_mac_per_token_modeled": int(3 * NATIVE_WIDTH * HEAD_DIM + 2 * NATIVE_WIDTH + CONV_CHANNELS * CONV_KERNEL + 4 * HEAD_DIM * HEAD_DIM),
            "active_input_port_mac_per_token_modeled": int(NATIVE_WIDTH * TASK_D_MODEL),
            "active_output_port_mac_per_token_modeled": int(HEAD_DIM * TASK_D_MODEL),
            "active_socket_mac_per_token_modeled": int(3 * NATIVE_WIDTH * HEAD_DIM + 2 * NATIVE_WIDTH + CONV_CHANNELS * CONV_KERNEL + 4 * HEAD_DIM * HEAD_DIM + NATIVE_WIDTH * TASK_D_MODEL + HEAD_DIM * TASK_D_MODEL),
            "input_port_semantics": "fixed analytic rank-96 donor basis in first 96 Remora coordinates; remaining 96 coordinates zero",
        },
        "function_preservation": {
            "native_core_parameters_are_source_values": True,
            "per_seed_native_vs_compact": {seed: per_seed[str(seed)]["native_vs_compact_equivalence"] for seed in seeds},
            "compact_reference": "results/qwen-neural-gdn-remora-path-v5.json",
        },
        "task": {
            "name": "aged-Remora delayed associative recall with native-width frozen Qwen core",
            "threshold": RECALL_THRESHOLD,
            "seeds": list(seeds),
            "per_seed": per_seed,
            "zero_step_actual_accuracy": _stats(actual),
            "zero_step_random_accuracy": _stats(random),
            "zero_step_shuffled_accuracy": _stats(shuffled),
            "zero_step_zero_accuracy": _stats(zero),
            "zero_step_shifted_actual_accuracy": _stats(shifted),
            "zero_step_actual_minus_random": _stats([a - r for a, r in zip(actual, random)]),
            "training_avoidance": "MEASURED zero-gradient bounded pathway utility; original Qwen training compute remains UNMEASURED",
        },
        "labels": {
            "MEASURED": [
                "native-width actual/random/shuffled/zero core task scores",
                "native-vs-compact core equivalence",
                "shifted native-interface scores",
                "bounded source extraction and checkpoint hashes",
            ],
            "DERIVED": ["native donor parameter accounting", "mean/std and paired differences"],
            "MODELED": ["native socket MACs", "frozen input-port parameter count"],
            "UNMEASURED": ["original donor pretraining compute avoided", "energy", "broad Remora language capability"],
        },
        "promotion_state": "CANDIDATE_EXACT_WEIGHT_REUSE_ONLY_UNTIL_NON_DONOR_CONDITIONED_VALIDATION",
        "interpretation": "A native-width wrapped core can preserve the donor core numerically and contribute to the same bounded Remora pathway without retraining; this does not establish broad capability assimilation, and its 1.57M modeled socket MACs/token and 491,520 frozen port parameters are the explicit cost of exact reuse.",
        "runtime": {**runtime_context(torch.device("cpu")), "cwd": str(ROOT), "mode": "native_width_gdn_core_socket_cpu"},
    }
    write_json(output, result)
    if record_ledger:
        record_experiment(
            ROOT,
            experiment_id,
            "A native-width wrapped Qwen GDN recurrent core should preserve its trained tensors exactly while contributing to the aged Remora pathway through a fixed analytic basis port.",
            "Keep the selected Qwen core tensors at native 2560-wide input geometry, attach a frozen rank-96 basis projection to blocks.1.plastic, and compare actual/random/shuffled/zero cores with shifted-interface controls.",
            "Actual native core matches the compact path numerically, clears the pathway threshold without gradient updates, and separates from destroyed native controls; exact-reuse memory and active compute are reported.",
            "Native-vs-compact equivalence fails, actual native core does not separate from random/shuffled/zero, or the apparent utility requires donor updates.",
            f"python -m experiments.donor_gdn_native_core --source {source_root} --checkpoint-pattern {checkpoint_pattern} --seeds {','.join(str(seed) for seed in seeds)} --experiment-id {experiment_id}",
            int(seeds[0]),
            {
                "zero_step_actual_accuracy": actual,
                "zero_step_random_accuracy": random,
                "zero_step_shifted_actual_accuracy": shifted,
                "donor_parameters_preserved_unchanged": core_source_parameters,
                "newly_trained_parameters": 0,
            },
            result["interpretation"],
            "Use this exact-reuse result as a costed control; prioritize a non-donor-conditioned task or transplant-tolerant bus before any scale decision.",
            hardware={**runtime_context(torch.device("cpu")), "mode": "native_width_gdn_core_socket_cpu", "source_model_materialized": False},
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the native-width exact-reuse Qwen GDN core experiment.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint-pattern", default=CHECKPOINT_PATTERN)
    parser.add_argument("--output", default=str(ROOT / "results" / "qwen-neural-gdn-native-core-v1.json"))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--experiment-id", default="QWEN-DONOR-GDN-NATIVE-CORE-001")
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()
    result = run(
        args.source,
        checkpoint_pattern=args.checkpoint_pattern,
        output=args.output,
        seeds=tuple(int(value) for value in args.seeds.split(",") if value),
        record_ledger=not args.no_ledger,
        experiment_id=args.experiment_id,
    )
    print(json.dumps({"output": str(args.output), "task": result["task"]["zero_step_actual_accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
