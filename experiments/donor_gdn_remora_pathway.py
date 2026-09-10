from __future__ import annotations

"""Put the selected Qwen GDN state core inside an aged Remora pathway.

The preceding GDN experiment established a donor-conditioned state task, but
the full Qwen head output was a poor Remora representation after analytic
width conversion.  This experiment therefore tests the smallest better-
justified boundary: retain the trained Q/K/V, causal-convolution, gate, and
recurrent state machinery, export its 128-wide core through a fixed Remora
branch port, and evaluate that socket inside ``blocks.1.plastic``.

The continuous input harness is deliberate.  It removes tokenizer/surface
protocol confounds while still executing the real Remora block stack and the
real aged checkpoint.  It is a bounded pathway capability test, not a claim
about language quality.  The task target is generated from the donor's
actual pre-query value vectors and is never used to change the donor weights.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environments.transfer_suite import build_transfer_suite  # noqa: E402
from remora.config import ModelConfig  # noqa: E402
from remora.donors.gdn import (  # noqa: E402
    QwenGatedDeltaCoreOrgan,
    QwenGatedDeltaCoreSocket,
    QwenGatedDeltaHead,
    TrainableGatedDeltaCoreOrgan,
    l2_normalize,
    make_gdn_variant,
)
from remora.donors.neural_ir import qwen_gated_delta_core_ir  # noqa: E402
from remora.metrics import evaluate_stream  # noqa: E402
from remora.models import build_model  # noqa: E402
from remora.modules import SwiGLUExpert  # noqa: E402
from remora.ledger import record_experiment, record_failure  # noqa: E402
from remora.utils import count_parameters, runtime_context, set_seed, write_json  # noqa: E402

from experiments.donor_gdn_head import (  # noqa: E402
    BUS_WIDTH,
    CONV_KERNEL,
    LAYER,
    _conversion_bases,
    _convert_payload,
    _extract_head,
)


TARGET_LAYER = 1
SELECTED_VALUE_HEAD = 10
MEMORY_ITEMS = 4
MEMORY_DELAY = 9
TASK_TIME = MEMORY_DELAY + 1
TASK_D_MODEL = 192
TRAIN_SAMPLES = 64
TEST_SAMPLES = 128
QUERY_SYNTHESIS_STEPS = 64
REPAIR_RANK = 4
REPAIR_STEPS = (0, 1, 2, 4, 8, 16, 32, 64)
SEEDS = (7, 19, 31)
RECALL_THRESHOLD = 0.50
CHECKPOINT_PATTERN = "checkpoints/aged-surgery-remora-seed{seed}.pt"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: Iterable[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    if not values:
        return {"values": [], "mean": None, "std": None}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
    return {"values": values, "mean": mean, "std": math.sqrt(variance)}


def _load_remora(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**payload["config"])
    model = build_model("remora", cfg)
    state_dict = payload["state_dict"]
    # Aged-surgery checkpoints contain a real SwiGLU expert replacement.  Make
    # the host geometry match before loading; no checkpoint tensor is altered.
    if any(name.startswith("blocks.1.experts.experts.0.value.") for name in state_dict):
        model.replace_expert(1, 0, SwiGLUExpert(cfg.bus_dim, cfg.d_ff, cfg.bus_dim))
    model.load_state_dict(state_dict)
    return model.to(device), payload


def _make_organ(
    payload: dict[str, torch.Tensor],
    input_basis: torch.Tensor,
    output_basis: torch.Tensor,
    variant: str,
    seed: int,
) -> QwenGatedDeltaCoreOrgan:
    variant_payload = make_gdn_variant(payload, variant, seed=seed)
    converted = _convert_payload(variant_payload, input_basis, output_basis)
    return QwenGatedDeltaCoreOrgan(
        **{name: converted[name] for name in ("q_weight", "k_weight", "v_weight", "a_weight", "b_weight", "conv_qkv", "A_log", "dt_bias")},
        donor_variant=variant,
    )


def _make_core_organ(
    payload: dict[str, torch.Tensor],
    input_basis: torch.Tensor,
    output_basis: torch.Tensor,
    variant: str,
    seed: int = 0,
) -> QwenGatedDeltaCoreOrgan:
    return _make_organ(payload, input_basis, output_basis, variant, seed)


CORE_TENSOR_NAMES = ("q_weight", "k_weight", "v_weight", "a_weight", "b_weight", "conv_qkv", "A_log", "dt_bias")


def _core_equivalence(
    selected_payload: dict[str, torch.Tensor],
    input_basis: torch.Tensor,
    output_basis: torch.Tensor,
) -> dict[str, Any]:
    """Verify that the compact core organ still matches the converted full head's state path."""

    converted = _convert_payload(selected_payload, input_basis, output_basis)
    full_head = QwenGatedDeltaHead(**converted, donor_variant="actual")
    generator = torch.Generator(device="cpu").manual_seed(1710)
    probe = torch.randn(2, 7, int(input_basis.shape[1]), generator=generator) * 0.20
    with torch.no_grad():
        _full_output, full_state, full_details = full_head(
            probe, return_state=True, return_intermediates=True
        )
        core_output, core_state, core_details = QwenGatedDeltaCoreOrgan(
            **{name: converted[name] for name in CORE_TENSOR_NAMES}
        )(probe, return_state=True, return_intermediates=True)
    delta = core_output.float() - full_details["core"].float()
    reference = full_details["core"].float()
    recurrent_delta = core_state[0].float() - full_state[0].float()
    convolution_delta = core_state[1].float() - full_state[1].float()
    return {
        "status": "MEASURED_CORE_EQUIVALENCE_AFTER_ANALYTIC_WIDTH_CONVERSION",
        "max_absolute_error": float(delta.abs().max()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference).clamp_min(1e-30)
        ),
        "cosine_similarity": float(
            F.cosine_similarity(core_output.float().reshape(1, -1), full_details["core"].float().reshape(1, -1)).item()
        ),
        "recurrent_state_relative_l2_error": float(
            torch.linalg.vector_norm(recurrent_delta)
            / torch.linalg.vector_norm(full_state[0].float()).clamp_min(1e-30)
        ),
        "convolution_state_relative_l2_error": float(
            torch.linalg.vector_norm(convolution_delta)
            / torch.linalg.vector_norm(full_state[1].float()).clamp_min(1e-30)
        ),
        "reference": "converted full Qwen head forward_core versus core-only organ on identical compact-bus probes",
    }


def _configure_task_host(
    model: torch.nn.Module,
    branch: QwenGatedDeltaCoreSocket,
) -> torch.nn.Module:
    """Isolate the installed branch while retaining the actual Remora stack."""

    with torch.no_grad():
        for block in model.blocks:
            block.branch_scale.zero_()
        model.blocks[TARGET_LAYER].branch_scale[3] = 1.0
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.blocks[TARGET_LAYER].plastic = branch
    for parameter in branch.port_parameters():
        parameter.requires_grad = True
    return model


def _run_continuous(model: torch.nn.Module, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run continuous inputs through the ordinary Remora block path."""

    x = hidden
    target_branch: QwenGatedDeltaCoreSocket | None = None
    for index, block in enumerate(model.blocks):
        x, _state, _aux = block(x, None)
        if index == TARGET_LAYER:
            target_branch = block.plastic
    if target_branch is None or target_branch.last_output is None or target_branch.last_core is None:
        raise RuntimeError("target Remora branch did not expose a donor trace")
    return {
        "path_output": target_branch.last_output,
        "core": target_branch.last_core,
        "value": target_branch.last_value,
        "pre_final_hidden": x,
    }


def _host_bus(norm: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    # The task's fixed port consumes the first bus-width coordinates of the
    # same LayerNorm output that the real Remora block would produce.
    return norm(hidden.float())[..., :BUS_WIDTH]


def _make_path_dataset(
    norm: torch.nn.Module,
    organ: QwenGatedDeltaCoreOrgan,
    *,
    samples: int,
    seed: int,
    query_steps: int = QUERY_SYNTHESIS_STEPS,
) -> dict[str, Any]:
    """Generate an external delayed-recall task on the actual donor manifold."""

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    sequence = torch.zeros(samples, TASK_TIME, TASK_D_MODEL)
    sequence[:, :MEMORY_ITEMS] = 0.20 * torch.randn(
        samples, MEMORY_ITEMS, TASK_D_MODEL, generator=generator
    )
    labels = torch.randint(MEMORY_ITEMS, (samples,), generator=generator)
    with torch.no_grad():
        prefix_bus = _host_bus(norm, sequence)
        _, _, prefix_details = organ(prefix_bus, return_state=True, return_intermediates=True)
        target_keys = prefix_details["key"][torch.arange(samples), labels].detach()

    query = (0.05 * torch.randn(samples, TASK_D_MODEL, generator=generator)).requires_grad_()
    optimizer = torch.optim.Adam([query], lr=0.20)
    beta_mean = 0.0
    for _ in range(int(query_steps)):
        candidate = sequence.clone()
        candidate[:, -1] = query
        bus = _host_bus(norm, candidate)
        _, _, details = organ(bus, return_state=True, return_intermediates=True)
        query_key = details["key"][:, -1]
        beta = details["beta"][:, -1].clamp(1e-5, 1.0 - 1e-5)
        loss = (1.0 - (l2_normalize(query_key) * target_keys).sum(dim=-1)).mean()
        loss = loss + 0.01 * (torch.logit(beta) + 5.0).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        beta_mean = float(beta.detach().mean())

    sequence[:, -1] = query.detach()
    with torch.no_grad():
        bus = _host_bus(norm, sequence)
        _, _, details = organ(bus, return_state=True, return_intermediates=True)
        query_alignment = float(
            (l2_normalize(details["key"][:, -1]) * target_keys).sum(dim=-1).mean()
        )
        values = details["value"][:, :MEMORY_ITEMS].detach()
        query_beta = float(details["beta"][:, -1].mean())
    return {
        "hidden": sequence.detach(),
        "labels": labels,
        "values": values,
        "query_key_cosine_mean": query_alignment,
        "query_beta_mean": query_beta,
        "query_synthesis_steps": int(query_steps),
        "query_beta_last_optimization_mean": beta_mean,
    }


def _padded_values(values: torch.Tensor, d_model: int = TASK_D_MODEL) -> torch.Tensor:
    if values.shape[-1] > d_model:
        raise ValueError("value vectors exceed Remora output width")
    return F.pad(values, (0, d_model - values.shape[-1]))


def _retrieval(core: torch.Tensor, labels: torch.Tensor, values: torch.Tensor) -> dict[str, float]:
    query = l2_normalize(core[:, -1])
    candidates = l2_normalize(values)
    scores = (query[:, None, :] * candidates).sum(dim=-1)
    prediction = scores.argmax(dim=-1)
    return {
        "accuracy": float((prediction == labels).float().mean()),
        "mean_target_cosine": float(scores[torch.arange(labels.shape[0]), labels].mean()),
        "mean_best_cosine": float(scores.max(dim=-1).values.mean()),
    }


def _socket_result(
    trace: dict[str, torch.Tensor],
    labels: torch.Tensor,
    values: torch.Tensor,
    *,
    own_values: torch.Tensor | None = None,
) -> dict[str, Any]:
    reference = _padded_values(values)
    result = {
        "core_to_actual_value": _retrieval(trace["core"], labels, values),
        "path_output_to_actual_value": _retrieval(trace["path_output"], labels, reference),
    }
    if own_values is not None:
        result["core_to_own_value"] = _retrieval(trace["core"], labels, own_values)
    return result


def _make_task_host(
    checkpoint: Path,
    organ: QwenGatedDeltaCoreOrgan,
    *,
    repair_rank: int = 0,
    shifted: bool = False,
) -> tuple[torch.nn.Module, QwenGatedDeltaCoreSocket, dict[str, Any]]:
    model, source = _load_remora(checkpoint, torch.device("cpu"))
    permutation = tuple(int(value) for value in torch.roll(torch.arange(BUS_WIDTH), shifts=7)) if shifted else None
    branch = QwenGatedDeltaCoreSocket(
        organ,
        model.cfg.d_model,
        repair_rank=repair_rank,
        interface_permutation=permutation,
        interface_scale=0.9 if shifted else 1.0,
    )
    return _configure_task_host(model, branch), branch, source


def _task_variant(
    checkpoint: Path,
    organ: QwenGatedDeltaCoreOrgan,
    dataset: dict[str, Any],
    *,
    shifted: bool = False,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model, branch, _source = _make_task_host(checkpoint, organ, shifted=shifted)
    model.eval()
    with torch.no_grad():
        trace = _run_continuous(model, dataset["hidden"])
    result = _socket_result(trace, dataset["labels"], dataset["values"], own_values=trace["value"][:, :MEMORY_ITEMS])
    result.update(
        {
            "shifted_interface": bool(shifted),
            "donor_core_frozen": True,
            "fixed_input_port": branch.trainable_parameter_count == 0,
            "trainable_parameters": branch.trainable_parameter_count,
        }
    )
    return result, trace


def _repair_state(branch: QwenGatedDeltaCoreSocket) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in branch.named_parameters()
        if parameter.requires_grad
    }


def _load_repair_state(branch: QwenGatedDeltaCoreSocket, state: dict[str, torch.Tensor]) -> None:
    named = dict(branch.named_parameters())
    for name, value in state.items():
        named[name].data.copy_(value)


def _repair_curve(
    checkpoint: Path,
    organ: QwenGatedDeltaCoreOrgan,
    train: dict[str, Any],
    test: dict[str, Any],
    *,
    seed: int,
    steps: tuple[int, ...] = REPAIR_STEPS,
) -> dict[str, Any]:
    """Measure how much low-rank compatibility repair is needed at the socket."""

    set_seed(seed)
    template_model, template_branch, _source = _make_task_host(
        checkpoint, organ, repair_rank=REPAIR_RANK
    )
    initial_state = _repair_state(template_branch)
    del template_model, template_branch
    reference_train = _padded_values(train["values"])
    reference_test = _padded_values(test["values"])
    rows: list[dict[str, Any]] = []
    for budget in steps:
        # Construct a clean host for every budget so each row is a true
        # logarithmic budget from the same no-op port initialization.
        model, branch, _source = _make_task_host(
            checkpoint, _clone_organ(organ), repair_rank=REPAIR_RANK
        )
        _load_repair_state(branch, initial_state)
        parameters = branch.port_parameters()
        optimizer = torch.optim.AdamW(parameters, lr=0.03, weight_decay=0.0)
        started = time.perf_counter()
        losses: list[float] = []
        model.train()
        for _ in range(int(budget)):
            trace = _run_continuous(model, train["hidden"])
            scores = (
                l2_normalize(trace["path_output"][:, -1])[:, None, :]
                * l2_normalize(reference_train)
            ).sum(dim=-1)
            loss = F.cross_entropy(scores * 10.0, train["labels"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            trace = _run_continuous(model, test["hidden"])
        output_metrics = _retrieval(trace["path_output"], test["labels"], reference_test)
        changed = sum(
            int((parameter.detach() - initial_state[name]).abs().gt(1e-12).sum())
            for name, parameter in branch.named_parameters()
            if name in initial_state
        )
        total_model = count_parameters(model)
        row = {
            "gradient_steps": int(budget),
            "accuracy": output_metrics["accuracy"],
            "mean_target_cosine": output_metrics["mean_target_cosine"],
            "loss_last": losses[-1] if losses else None,
            "wall_seconds": time.perf_counter() - started,
            "tokens": int(budget * train["hidden"].shape[0] * train["hidden"].shape[1]),
            "trainable_parameters": int(sum(parameter.numel() for parameter in parameters)),
            "changed_parameters": int(changed),
            "changed_fraction_total_model": changed / max(total_model, 1),
        }
        if int(budget) == max(steps):
            shifted_model, shifted_branch, _source = _make_task_host(
                checkpoint, _clone_organ(organ), repair_rank=REPAIR_RANK, shifted=True
            )
            _load_repair_state(shifted_branch, {
                name: parameter.detach().clone()
                for name, parameter in branch.named_parameters()
                if name in initial_state
            })
            shifted_model.eval()
            with torch.no_grad():
                shifted_trace = _run_continuous(shifted_model, test["hidden"])
            row["shifted_interface_accuracy"] = _retrieval(
                shifted_trace["path_output"], test["labels"], reference_test
            )["accuracy"]
        rows.append(row)
        del model, branch
    return {
        "repair_rank": REPAIR_RANK,
        "trainable_parameters": int(sum(value.numel() for value in initial_state.values())),
        "steps": rows,
        "target": "frozen actual-value memory vectors with a low-rank socket repair; no donor-core updates",
    }


def _make_fresh_organ(
    payload: dict[str, torch.Tensor],
    input_basis: torch.Tensor,
    output_basis: torch.Tensor,
    *,
    seed: int,
) -> TrainableGatedDeltaCoreOrgan:
    random_payload = make_gdn_variant(payload, "random", seed=seed)
    converted = _convert_payload(random_payload, input_basis, output_basis)
    return TrainableGatedDeltaCoreOrgan(
        **{
            name: converted[name]
            for name in CORE_TENSOR_NAMES
        },
        donor_variant="fresh",
    )


def _clone_fresh_organ(organ: TrainableGatedDeltaCoreOrgan) -> TrainableGatedDeltaCoreOrgan:
    return TrainableGatedDeltaCoreOrgan(
        **{
            name: getattr(organ, name).detach().clone()
            for name in CORE_TENSOR_NAMES
        },
        donor_variant="fresh",
    )


def _fresh_same_mechanism_curve(
    checkpoint: Path,
    payload: dict[str, torch.Tensor],
    input_basis: torch.Tensor,
    output_basis: torch.Tensor,
    train: dict[str, Any],
    test: dict[str, Any],
    *,
    seed: int,
    steps: tuple[int, ...] = REPAIR_STEPS,
) -> dict[str, Any]:
    """Measure full-core relearning inside the exact same aged Remora path."""

    set_seed(seed)
    template = _make_fresh_organ(payload, input_basis, output_basis, seed=seed)
    initial_state = {
        name: parameter.detach().clone() for name, parameter in template.named_parameters()
    }
    rows: list[dict[str, Any]] = []
    reference_train = _padded_values(train["values"])
    reference_test = _padded_values(test["values"])
    for budget in steps:
        model, branch, _source = _make_task_host(
            checkpoint, _clone_fresh_organ(template), repair_rank=0
        )
        fresh = branch.organ
        parameters = list(fresh.parameters())
        for parameter in parameters:
            parameter.requires_grad = True
        optimizer = torch.optim.AdamW(parameters, lr=0.03, weight_decay=0.0)
        started = time.perf_counter()
        losses: list[float] = []
        model.train()
        for _ in range(int(budget)):
            trace = _run_continuous(model, train["hidden"])
            scores = (
                l2_normalize(trace["path_output"][:, -1])[:, None, :]
                * l2_normalize(reference_train)
            ).sum(dim=-1)
            loss = F.cross_entropy(scores * 10.0, train["labels"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            trace = _run_continuous(model, test["hidden"])
        output_metrics = _retrieval(trace["path_output"], test["labels"], reference_test)
        changed = sum(
            int((parameter.detach() - initial_state[name]).abs().gt(1e-12).sum())
            for name, parameter in fresh.named_parameters()
        )
        total_model = count_parameters(model)
        row = {
            "gradient_steps": int(budget),
            "accuracy": output_metrics["accuracy"],
            "mean_target_cosine": output_metrics["mean_target_cosine"],
            "loss_last": losses[-1] if losses else None,
            "wall_seconds": time.perf_counter() - started,
            "tokens": int(budget * train["hidden"].shape[0] * train["hidden"].shape[1]),
            "trainable_parameters": int(sum(parameter.numel() for parameter in parameters)),
            "changed_parameters": int(changed),
            "changed_fraction_total_model": changed / max(total_model, 1),
        }
        if int(budget) == max(steps):
            shifted_model, shifted_branch, _source = _make_task_host(
                checkpoint, _clone_fresh_organ(template), shifted=True
            )
            for name, value in fresh.named_parameters():
                getattr(shifted_branch.organ, name).data.copy_(value.detach())
            shifted_model.eval()
            with torch.no_grad():
                shifted_trace = _run_continuous(shifted_model, test["hidden"])
            row["shifted_interface_accuracy"] = _retrieval(
                shifted_trace["path_output"], test["labels"], reference_test
            )["accuracy"]
        rows.append(row)
        del model, branch
    return {
        "initialization": "fresh same-geometry GDN core using actual-payload standard deviations; no donor values",
        "trainable_parameters": int(sum(value.numel() for value in initial_state.values())),
        "steps": rows,
        "target": "frozen actual-value memory vectors with the full fresh core trainable; no donor-core weights",
    }


def _clone_organ(organ: QwenGatedDeltaCoreOrgan) -> QwenGatedDeltaCoreOrgan:
    tensors = {name: value.detach().clone() for name, value in organ.named_buffers()}
    return QwenGatedDeltaCoreOrgan(
        q_weight=tensors["q_weight"],
        k_weight=tensors["k_weight"],
        v_weight=tensors["v_weight"],
        a_weight=tensors["a_weight"],
        b_weight=tensors["b_weight"],
        conv_qkv=tensors["conv_qkv"],
        A_log=tensors["A_log"],
        dt_bias=tensors["dt_bias"],
        donor_variant=organ.donor_variant,
    )


def _old_capability(
    checkpoint: Path,
    actual_organ: QwenGatedDeltaCoreOrgan,
    zero_organ: QwenGatedDeltaCoreOrgan,
    suite: Any,
    *,
    tokens: int,
) -> dict[str, Any]:
    """Measure insertion damage on the aged model's ordinary LM path."""

    def evaluate(model: torch.nn.Module, stream: torch.Tensor) -> dict[str, Any]:
        bounded = stream[: tokens + 1]
        return evaluate_stream(model, bounded, batch_size=4, seq_len=32, device=torch.device("cpu"))

    base_model, _base_source = _load_remora(checkpoint, torch.device("cpu"))
    base_model.eval()
    baseline = {
        domain: evaluate(base_model, suite.valid_streams[domain]) for domain in ("text", "code")
    }
    del base_model

    actual_model, _actual_source = _load_remora(checkpoint, torch.device("cpu"))
    actual_model.blocks[TARGET_LAYER].plastic = QwenGatedDeltaCoreSocket(
        actual_organ, actual_model.cfg.d_model
    )
    actual_model.eval()
    actual = {
        domain: evaluate(actual_model, suite.valid_streams[domain]) for domain in ("text", "code")
    }
    del actual_model

    zero_model, _zero_source = _load_remora(checkpoint, torch.device("cpu"))
    zero_model.blocks[TARGET_LAYER].plastic = QwenGatedDeltaCoreSocket(
        zero_organ, zero_model.cfg.d_model
    )
    zero_model.eval()
    zero = {
        domain: evaluate(zero_model, suite.valid_streams[domain]) for domain in ("text", "code")
    }
    del zero_model
    return {
        "evaluation_tokens_per_domain": int(tokens),
        "baseline_aged_checkpoint": baseline,
        "actual_core_socket": actual,
        "zero_core_socket": zero,
        "loss_delta_actual_minus_baseline": {
            domain: actual[domain]["loss"] - baseline[domain]["loss"] for domain in ("text", "code")
        },
        "loss_delta_actual_minus_zero": {
            domain: actual[domain]["loss"] - zero[domain]["loss"] for domain in ("text", "code")
        },
    }


def _first_threshold(rows: list[dict[str, Any]], threshold: float = RECALL_THRESHOLD) -> int | None:
    for row in rows:
        if float(row["accuracy"]) >= float(threshold):
            return int(row["gradient_steps"])
    return None


def run(
    source_path: str | Path,
    *,
    checkpoint_pattern: str = CHECKPOINT_PATTERN,
    wiki_root: str | Path = "/home/leo/tmp/wikitext-2-raw",
    output: str | Path = ROOT / "results" / "qwen-neural-gdn-remora-path-v4.json",
    seeds: tuple[int, ...] = SEEDS,
    repair_steps: tuple[int, ...] = REPAIR_STEPS,
    old_eval_tokens: int = 512,
    record_ledger: bool = True,
    experiment_id: str = "QWEN-DONOR-GDN-REMORA-PATH-003",
) -> dict[str, Any]:
    source_root = Path(source_path).expanduser().resolve()
    output = Path(output)
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    if not seeds:
        raise ValueError("at least one seed is required")
    if not experiment_id.strip():
        raise ValueError("experiment_id must be non-empty")
    if not repair_steps or min(repair_steps) < 0 or tuple(sorted(set(repair_steps))) != tuple(repair_steps):
        raise ValueError("repair_steps must be sorted, unique, and non-negative")

    selected_payload, selected_metadata = _extract_head(source_root, LAYER, SELECTED_VALUE_HEAD)
    input_basis, output_basis = _conversion_bases(selected_payload)
    actual_organ = _make_core_organ(
        selected_payload, input_basis, output_basis, "actual"
    )
    zero_organ = _make_core_organ(
        selected_payload, input_basis, output_basis, "zero"
    )
    core_equivalence = _core_equivalence(selected_payload, input_basis, output_basis)
    suite = build_transfer_suite(ROOT, wiki_root)
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        # Model construction initializes tensors before loading the immutable
        # checkpoint.  Seed that construction explicitly so reported seed
        # comparisons do not inherit process-global RNG state.
        set_seed(int(seed))
        checkpoint = Path(checkpoint_pattern.format(seed=int(seed)))
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        norm_model, checkpoint_source = _load_remora(checkpoint, torch.device("cpu"))
        norm = norm_model.blocks[TARGET_LAYER].norm
        train = _make_path_dataset(norm, actual_organ, samples=TRAIN_SAMPLES, seed=int(seed) + 1000)
        test = _make_path_dataset(norm, actual_organ, samples=TEST_SAMPLES, seed=int(seed) + 2000)
        del norm_model

        variants = {
            "actual": actual_organ,
            "random": _make_organ(
                selected_payload, input_basis, output_basis, "random", int(seed) + 5000
            ),
            "shuffled": _make_organ(
                selected_payload, input_basis, output_basis, "shuffled", int(seed) + 6000
            ),
            "zero": zero_organ,
        }
        zero_shot: dict[str, Any] = {}
        traces: dict[str, dict[str, torch.Tensor]] = {}
        for name, organ in variants.items():
            result, trace = _task_variant(checkpoint, organ, test)
            shifted_result, _shifted_trace = _task_variant(checkpoint, organ, test, shifted=True)
            result["shifted_interface"] = shifted_result["core_to_actual_value"]
            result["shifted_interface_path_output"] = shifted_result["path_output_to_actual_value"]
            zero_shot[name] = result
            traces[name] = trace
        causal_delta = (traces["actual"]["pre_final_hidden"] - traces["zero"]["pre_final_hidden"]).abs()
        repair_curves = {
            name: _repair_curve(
                checkpoint,
                organ,
                train,
                test,
                seed=int(seed) + (8100 if name == "actual" else 8200),
                steps=repair_steps,
            )
            for name, organ in (("actual", actual_organ), ("random", variants["random"]))
        }
        fresh_curve = _fresh_same_mechanism_curve(
            checkpoint,
            selected_payload,
            input_basis,
            output_basis,
            train,
            test,
            seed=int(seed) + 8300,
            steps=repair_steps,
        )
        old = _old_capability(
            checkpoint,
            actual_organ,
            zero_organ,
            suite,
            tokens=old_eval_tokens,
        )
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
                "memory_items": MEMORY_ITEMS,
                "memory_delay": MEMORY_DELAY,
                "time": TASK_TIME,
                "input_contract": ["batch", "time", TASK_D_MODEL],
                "query_synthesis": "64-step analytic optimization through aged Remora target LayerNorm and actual converted donor key; no donor training",
                "train_query_key_cosine_mean": train["query_key_cosine_mean"],
                "test_query_key_cosine_mean": test["query_key_cosine_mean"],
                "train_query_beta_mean": train["query_beta_mean"],
                "test_query_beta_mean": test["query_beta_mean"],
                "shifted_interface": "fixed GDN input port cyclically permuted by 7 and scaled by 0.9",
            },
            "zero_shot": zero_shot,
            "causal_path_contribution": {
                "actual_vs_zero_pre_final_hidden_delta_max": float(causal_delta.max()),
                "actual_vs_zero_pre_final_hidden_delta_mean": float(causal_delta.mean()),
                "scope": "isolated blocks.1.plastic branch inside the real Remora block stack",
            },
            "micro_repair_curves": repair_curves,
            "micro_repair_threshold_steps": {
                name: _first_threshold(curve["steps"]) for name, curve in repair_curves.items()
            },
            "fresh_trainable_same_mechanism_curve": fresh_curve,
            "fresh_trainable_same_mechanism_threshold_steps": _first_threshold(fresh_curve["steps"]),
            "old_capability": old,
        }

    actual_zero = [per_seed[str(seed)]["zero_shot"]["actual"]["core_to_actual_value"]["accuracy"] for seed in seeds]
    random_zero = [per_seed[str(seed)]["zero_shot"]["random"]["core_to_actual_value"]["accuracy"] for seed in seeds]
    shifted_actual = [
        per_seed[str(seed)]["zero_shot"]["actual"]["shifted_interface"]["accuracy"] for seed in seeds
    ]
    fresh_thresholds = [
        per_seed[str(seed)]["fresh_trainable_same_mechanism_threshold_steps"] for seed in seeds
    ]
    result = {
        "schema": "remora-qwen-neural-gdn-remora-path-result-v2",
        "experiment_id": experiment_id,
        "experiment_family": f"{experiment_id}+",
        "source": {
            "repository": "Qwen/Qwen3.8-Flash-Next",
            "revision": "f5d08274bafd880402bd16f5e3e6c514136ec06c",
            "path": str(source_root),
            "layer": LAYER,
            "value_head": SELECTED_VALUE_HEAD,
            "key_head": SELECTED_VALUE_HEAD // 3,
            "source_shard": selected_metadata["source_shard"],
            "source_tensor_names": selected_metadata["source_tensor_names"],
            "source_slices": selected_metadata["source_slices"],
            "source_payload_bytes_bf16": selected_metadata["source_payload_bytes_bf16"],
            "selected_payload_materialized_bytes_fp32": selected_metadata["selected_payload_materialized_bytes_fp32"],
            "tensor_hashes_float32": selected_metadata["tensor_hashes_float32"],
            "full_model_materialized": False,
            "model_loader_called": False,
            "selection_evidence": "results/qwen-neural-gdn-candidate-ranking-v2.json; highest bounded state-response score",
        },
        "graft": {
            "mode": "SUBMODULE_SUBSPACE_GRAFT_CORE_SOCKET",
            "attached_module_path": "blocks.1.plastic",
            "socket": QwenGatedDeltaCoreSocket(
                actual_organ, TASK_D_MODEL, repair_rank=REPAIR_RANK
            ).interface_signature(),
            "donor_core_frozen": True,
            "base_ports_frozen": True,
            "micro_repair_rank": REPAIR_RANK,
            "fresh_control_trainable_parameters": int(3 * 128 * BUS_WIDTH + 2 * BUS_WIDTH + 3 * 128 * CONV_KERNEL + 2),
            "function_preservation_ladder": {
                "D0_original_extracted": {
                    "status": "MEASURED",
                    "source_parameters": int(sum(value.numel() for value in selected_payload.values())),
                    "source_payload_bytes_bf16": int(selected_metadata["source_payload_bytes_bf16"]),
                    "reference": "selected Qwen layer-17 value-head-10 payload",
                },
                "D1_standalone_reproduction": {
                    "status": "MEASURED_EQUIVALENT",
                    "reference": "results/qwen-neural-gdn-head-v2.json selected_organ.functional_preservation_ladder.steps.D1_standalone_reproduction",
                },
                "D2_projection_packing": {
                    "status": "MEASURED_EQUIVALENT",
                    "reference": "results/qwen-neural-gdn-head-v2.json selected_organ.functional_preservation_ladder.steps.D2_projection_packing",
                },
                "D3_D4_full_head_compression_boundary": {
                    "status": "MEASURED_LOSSY_BOUNDARY_CONTROL",
                    "reference": "results/qwen-neural-gdn-head-v2.json selected_organ.functional_preservation_ladder.steps.D3_input_compression_rank_96 and D4_width_conversion_rank_96",
                },
                "D5_core_subgraph_extraction": core_equivalence,
                "D6_attached_core_socket": {
                    "status": "MEASURED_CONDITIONAL_UTILITY",
                    "zero_step_actual_accuracy": _stats(actual_zero),
                    "zero_step_random_accuracy": _stats(random_zero),
                    "shifted_interface_actual_accuracy": _stats(shifted_actual),
                    "note": "task utility is not a function-equivalence metric; it is the downstream aged-Remora pathway test",
                },
            },
            "neural_ir": qwen_gated_delta_core_ir(
                component_id=f"remora-gdn-core-candidate-layer{LAYER}-head{SELECTED_VALUE_HEAD}",
                source_revision="f5d08274bafd880402bd16f5e3e6c514136ec06c",
                tensor_names=list(CORE_TENSOR_NAMES),
                layer=LAYER,
                value_head=SELECTED_VALUE_HEAD,
                input_width=BUS_WIDTH,
                output_width=128,
                head_dim=128,
                conv_channels=3 * 128,
                conv_kernel=CONV_KERNEL,
            ).to_dict(),
            "donor_parameters_source_total": int(sum(value.numel() for value in selected_payload.values())),
            "donor_parameters_preserved_unchanged_source": int(
                selected_payload["conv_qkv"].numel()
                + selected_payload["A_log"].numel()
                + selected_payload["dt_bias"].numel()
            ),
            "donor_parameters_analytically_transformed_source": int(
                sum(
                    selected_payload[name].numel()
                    for name in (
                        "q_weight",
                        "k_weight",
                        "v_weight",
                        "a_weight",
                        "b_weight",
                    )
                )
            ),
            "donor_parameters_excluded_noncore_source": int(
                selected_payload["z_weight"].numel()
                + selected_payload["norm_weight"].numel()
                + selected_payload["out_weight"].numel()
            ),
            "donor_parameters_discarded_by_width_conversion": int(
                sum(selected_payload[name].numel() for name in ("q_weight", "k_weight", "v_weight", "a_weight", "b_weight"))
                - sum(
                    value.numel()
                    for name, value in _convert_payload(selected_payload, input_basis, output_basis).items()
                    if name in ("q_weight", "k_weight", "v_weight", "a_weight", "b_weight")
                )
            ),
            "donor_parameters_preserved_unchanged": int(
                selected_payload["conv_qkv"].numel()
                + selected_payload["A_log"].numel()
                + selected_payload["dt_bias"].numel()
            ),
            "donor_parameters_analytically_transformed": int(
                sum(
                    value.numel()
                    for name, value in _convert_payload(selected_payload, input_basis, output_basis).items()
                    if name in ("q_weight", "k_weight", "v_weight", "a_weight", "b_weight")
                )
            ),
            "donor_parameters_analytically_transformed_resident": int(
                sum(
                    value.numel()
                    for name, value in _convert_payload(selected_payload, input_basis, output_basis).items()
                    if name in ("q_weight", "k_weight", "v_weight", "a_weight", "b_weight")
                )
            ),
            "donor_parameters_discarded": int(
                sum(value.numel() for value in selected_payload.values())
                - (
                    selected_payload["conv_qkv"].numel()
                    + selected_payload["A_log"].numel()
                    + selected_payload["dt_bias"].numel()
                )
                - sum(
                    value.numel()
                    for name, value in _convert_payload(selected_payload, input_basis, output_basis).items()
                    if name in ("q_weight", "k_weight", "v_weight", "a_weight", "b_weight")
                )
            ),
            "donor_parameters_resident_compact": int(
                sum(
                    value.numel()
                    for name, value in _convert_payload(selected_payload, input_basis, output_basis).items()
                    if name in CORE_TENSOR_NAMES
                )
            ),
            "resident_compact_bytes_fp32": int(
                sum(
                    value.numel()
                    for name, value in _convert_payload(selected_payload, input_basis, output_basis).items()
                    if name in CORE_TENSOR_NAMES
                )
                * 4
            ),
            "donor_parameters_modified_during_assimilation": 0,
            "newly_trained_parameters_max_micro_repair": int(
                REPAIR_RANK * (TASK_D_MODEL + BUS_WIDTH + 128 + TASK_D_MODEL)
            ),
            "newly_trained_parameters_fresh_same_mechanism_control": int(
                3 * 128 * BUS_WIDTH + 2 * BUS_WIDTH + 3 * 128 * CONV_KERNEL + 2
            ),
            "original_donor_training_compute": None,
            "donor_parameters_discarded_from_resident_full_head": int(
                sum(value.numel() for value in selected_payload.values())
                - sum(
                    value.numel()
                    for name, value in _convert_payload(selected_payload, input_basis, output_basis).items()
                    if name in CORE_TENSOR_NAMES
                )
            ),
            "micro_repair_trainable_parameters": int(
                REPAIR_RANK * (TASK_D_MODEL + BUS_WIDTH + 128 + TASK_D_MODEL)
            ),
            "active_core_mac_per_token_modeled": int(
                3 * BUS_WIDTH * 128
                + 2 * BUS_WIDTH
                + 3 * 128 * CONV_KERNEL
                + 4 * 128 * 128
            ),
            "active_output_port_mac_per_token_modeled": int(128 * TASK_D_MODEL),
            "full_head_width_conversion_reference": "results/qwen-neural-gdn-head-v2.json D4; full output is retained as a negative boundary control",
        },
        "task": {
            "name": "aged-Remora delayed associative recall through core socket",
            "threshold": RECALL_THRESHOLD,
            "seeds": list(seeds),
            "repair_steps": list(repair_steps),
            "per_seed": per_seed,
            "zero_step_actual_accuracy": _stats(actual_zero),
            "zero_step_random_accuracy": _stats(random_zero),
            "zero_step_actual_minus_random": _stats([a - r for a, r in zip(actual_zero, random_zero)]),
            "zero_step_shifted_actual_accuracy": _stats(shifted_actual),
            "fresh_trainable_same_mechanism_threshold_steps": fresh_thresholds,
            "training_avoidance": "MEASURED for the bounded Remora pathway task only; original Qwen training compute remains UNMEASURED",
        },
        "labels": {
            "MEASURED": [
                "actual/random/shuffled/zero core socket recall inside blocks.1.plastic",
                "shifted-interface recall",
                "causal pre-final hidden intervention",
                "low-rank micro-repair curves and changed-parameter counts",
                "fresh same-mechanism full-core relearning curve",
                "compact-core equivalence against converted full-head state path",
                "aged-checkpoint text/code insertion losses",
                "bounded selective source extraction and tensor hashes",
            ],
            "DERIVED": [
                "donor parameter accounting",
                "repair threshold steps",
                "mean/std across seeds",
                "right-censored threshold records",
            ],
            "MODELED": ["active core MACs", "socket state shape"],
            "UNMEASURED": ["original donor pretraining compute", "energy", "full Remora language capability gain"],
            "HYPOTHESIS": ["a trained donor GDN core can export useful state computation through a Remora branch socket"],
        },
        "promotion_state": "CANDIDATE_ONLY_UNTIL_SHIFTED_GENERALIZATION_AND_NONTRIVIAL_TASK_HEAD_VALIDATION",
        "interpretation": (
            "MEASURED CONDITIONAL POSITIVE: the actual trained GDN core clears the fixed recall threshold through the real aged Remora plastic branch and separates from destroyed-core controls at zero repair steps; this is not broad language capability or proof of reclaimed Qwen pretraining compute."
            if sum(actual_zero) / len(actual_zero) > RECALL_THRESHOLD
            and sum(actual_zero) / len(actual_zero) > sum(random_zero) / len(random_zero) + 0.20
            else "MEASURED NEGATIVE/PARTIAL: the actual GDN core did not clear the predeclared aged-Remora pathway gate; preserve the donor machinery and redesign the socket or task boundary."
        ),
        "runtime": {**runtime_context(torch.device("cpu")), "cwd": str(ROOT), "mode": "aged_remora_gdn_core_socket_cpu"},
    }
    write_json(output, result)
    if record_ledger:
        record_experiment(
            ROOT,
            experiment_id,
            "A real Qwen-trained GatedDeltaNet recurrent core should export a delayed-association capability through a fixed core socket inside an aged Remora plastic branch without donor retraining.",
            "Use the selected layer-17/value-head-10 compact organ, end the graft at its trained 128-wide recurrent core, attach it to blocks.1.plastic with frozen identity geometry ports, and compare actual/random/shuffled/zero controls plus low-rank micro-repair.",
            "The actual core reaches the fixed recall threshold inside the Remora pathway at zero repair steps, separates from destroyed controls, and remains causally present after attachment; old checkpoint capability is measured separately.",
            "Actual core fails the threshold or loses its margin over random/shuffled/zero, shifted interface is indistinguishable from same-interface success, or donor contribution disappears under intervention.",
            f"python -m experiments.donor_gdn_remora_pathway --source {source_root} --checkpoint-pattern {checkpoint_pattern} --seeds {','.join(str(seed) for seed in seeds)} --experiment-id {experiment_id}",
            int(seeds[0]),
            {
                "zero_step_actual_accuracy": actual_zero,
                "zero_step_random_accuracy": random_zero,
                "zero_step_shifted_actual_accuracy": shifted_actual,
                "micro_repair_threshold_steps": {
                    name: [per_seed[str(seed)]["micro_repair_threshold_steps"][name] for seed in seeds]
                    for name in ("actual", "random")
                },
                "fresh_trainable_same_mechanism_threshold_steps": [
                    per_seed[str(seed)]["fresh_trainable_same_mechanism_threshold_steps"] for seed in seeds
                ],
                "promotion_state": result["promotion_state"],
            },
            result["interpretation"],
            "If the core socket passes, run a task-head/output-contract validation with randomized surface interfaces; if it fails, retain the exact core-state positive as donor-conditioned only and target native-manifold capture or a wider state subgraph.",
            hardware={**runtime_context(torch.device("cpu")), "mode": "aged_remora_gdn_core_socket_cpu", "source_model_materialized": False},
        )
        if sum(actual_zero) / len(actual_zero) <= sum(random_zero) / len(random_zero) + 0.20:
            record_failure(
                ROOT,
                experiment_id,
                {"checkpoint_pattern": checkpoint_pattern, "target_layer": TARGET_LAYER, "bus_width": BUS_WIDTH},
                {"memory_items": MEMORY_ITEMS, "memory_delay": MEMORY_DELAY, "threshold": RECALL_THRESHOLD},
                int(seeds[0]),
                "The actual donor core did not clear the predeclared margin over the random core inside the aged Remora socket.",
                "The donor state function may be real but not exportable through the fixed Remora path, or the task remains too donor-conditioned.",
                "Try native activation-manifold capture, a learned semantic bus trained for transplant tolerance, or a larger closed GDN state subgraph; do not scale.",
                runtime={**runtime_context(torch.device("cpu")), "mode": "aged_remora_gdn_core_socket_cpu"},
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach a bounded Qwen GDN core organ to an aged Remora pathway.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint-pattern", default=CHECKPOINT_PATTERN)
    parser.add_argument("--wiki-root", default="/home/leo/tmp/wikitext-2-raw")
    parser.add_argument("--output", default=str(ROOT / "results" / "qwen-neural-gdn-remora-path-v4.json"))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--repair-steps", default=",".join(str(step) for step in REPAIR_STEPS))
    parser.add_argument("--old-eval-tokens", type=int, default=512)
    parser.add_argument("--experiment-id", default="QWEN-DONOR-GDN-REMORA-PATH-003")
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()
    result = run(
        args.source,
        checkpoint_pattern=args.checkpoint_pattern,
        wiki_root=args.wiki_root,
        output=args.output,
        seeds=tuple(int(value) for value in args.seeds.split(",") if value),
        repair_steps=tuple(int(value) for value in args.repair_steps.split(",") if value),
        old_eval_tokens=args.old_eval_tokens,
        record_ledger=not args.no_ledger,
        experiment_id=args.experiment_id,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "zero_step_actual_accuracy": result["task"]["zero_step_actual_accuracy"],
                "zero_step_random_accuracy": result["task"]["zero_step_random_accuracy"],
                "zero_step_shifted_actual_accuracy": result["task"]["zero_step_shifted_actual_accuracy"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
