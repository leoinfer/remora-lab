from __future__ import annotations

"""Bounded Qwen router organs and analytic width conversion.

The router is deliberately a smaller donor target than the earlier shared
expert.  A selected row set is the closed subgraph: it owns a finite affine
decision function, has no hidden recurrent state, and can be attached to the
existing Remora expert-router socket.  The full donor tensor is only used
while constructing an analytic input basis and is never required at runtime.
"""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def top_right_singular_basis(weight: torch.Tensor, width: int) -> torch.Tensor:
    """Return an orthonormal input basis for a bounded analytic conversion.

    ``weight`` is expected to use the ordinary ``[out, in]`` convention.  The
    returned matrix has shape ``[in, width]``.  It contains no trainable
    parameters; it is an analytic coordinate change derived from the donor
    tensor.  ``torch.linalg.svd`` is intentionally used here because the
    router is only 512 x 2560 and the conversion is performed once, offline.
    """

    if weight.ndim != 2:
        raise ValueError(f"expected a matrix, got shape {tuple(weight.shape)}")
    if not 0 < int(width) <= min(weight.shape):
        raise ValueError(f"basis width must be in [1, {min(weight.shape)}], got {width}")
    _, _, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    return vh[: int(width)].transpose(0, 1).contiguous()


def select_balanced_pair(weight: torch.Tensor, basis: torch.Tensor | None = None) -> tuple[int, int]:
    """Choose two donor routes with the largest balanced held-out margin.

    The selection is deterministic and does not use the Remora task labels.
    It chooses the pair whose normalized donor-row prototypes are easiest for
    the donor router to distinguish, after the supplied width conversion.  A
    pair rather than a whole 512-way router is used for the first graft so the
    organ fits the existing two-expert Remora socket exactly.
    """

    if weight.ndim != 2 or weight.shape[0] < 2:
        raise ValueError("pair selection requires at least two output rows")
    score_weight = weight.float() if basis is None else weight.float() @ basis.float()
    normalized = torch.nn.functional.normalize(score_weight, dim=-1)
    norms = score_weight.norm(dim=-1)
    # Scores for a unit prototype centred on row i under rows i and j.
    scores = score_weight @ normalized.transpose(0, 1)
    margins = scores.diagonal().unsqueeze(1) - scores
    balanced = torch.minimum(margins, margins.transpose(0, 1))
    norm_ratio = torch.minimum(norms.unsqueeze(1), norms.unsqueeze(0)) / norms.clamp_min(1e-12).maximum(
        norms.unsqueeze(1),
    )
    # Keep the margin primary but avoid a pair with a nearly zero row.
    balanced = balanced + 1e-3 * norm_ratio
    balanced.fill_diagonal_(-torch.inf)
    index = int(torch.argmax(balanced).item())
    first, second = divmod(index, int(weight.shape[0]))
    if first == second:
        raise RuntimeError("pair selector returned a diagonal entry")
    return (first, second) if first < second else (second, first)


def make_router_variant(
    weight: torch.Tensor,
    variant: str,
    *,
    seed: int = 0,
) -> torch.Tensor:
    """Create matched donor-core controls without mutating the source tensor."""

    if variant not in {"actual", "random", "shuffled", "zero"}:
        raise ValueError(variant)
    source = weight.detach().cpu().float().contiguous()
    if variant == "actual":
        return source.clone()
    if variant == "zero":
        return torch.zeros_like(source)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    if variant == "random":
        return torch.randn(source.shape, generator=generator) * source.std(unbiased=False).clamp_min(1e-6)
    flat = source.reshape(-1)
    return flat[torch.randperm(flat.numel(), generator=generator)].reshape(source.shape).contiguous()


class QwenRouterOrgan(nn.Module):
    """A frozen, selectively retained Qwen router behind an analytic port.

    ``donor_weight`` may contain the full router or a selected row set.  The
    organ retains only those rows at runtime.  ``input_basis`` maps a Remora
    bus vector into the selected donor input subspace.  With an identity basis
    this is the exact donor affine function; with a truncated SVD basis it is
    the explicitly measured compact approximation.
    """

    interface_version = "qwen-router-organ-v1"

    def __init__(
        self,
        donor_weight: torch.Tensor,
        *,
        input_basis: torch.Tensor,
        selected_rows: tuple[int, ...] | list[int] | None = None,
        donor_variant: str = "actual",
        variant_seed: int = 0,
        trainable_repair_rank: int = 0,
        repair_initialization: str = "zero",
    ):
        super().__init__()
        if donor_weight.ndim != 2:
            raise ValueError("donor router must be a matrix")
        if input_basis.ndim != 2 or input_basis.shape[0] != donor_weight.shape[1]:
            raise ValueError("input basis must have shape [donor_width, bus_width]")
        if selected_rows is None:
            selected = tuple(range(int(donor_weight.shape[0])))
        else:
            selected = tuple(int(index) for index in selected_rows)
        if not selected or any(index < 0 or index >= donor_weight.shape[0] for index in selected):
            raise ValueError("selected rows must be non-empty and in range")
        if len(set(selected)) != len(selected):
            raise ValueError("selected rows must be unique")
        variant = make_router_variant(donor_weight, donor_variant, seed=variant_seed)
        selected_weight = variant[list(selected)].contiguous()
        self.donor_variant = str(donor_variant)
        self.donor_variant_seed = int(variant_seed)
        self.selected_rows = selected
        self.donor_width = int(donor_weight.shape[1])
        self.bus_dim = int(input_basis.shape[1])
        self.donor_route_count = int(donor_weight.shape[0])
        self.register_buffer("donor_weight", selected_weight)
        self.register_buffer("input_basis", input_basis.detach().cpu().float().contiguous())
        self.repair_rank = int(trainable_repair_rank)
        if self.repair_rank < 0:
            raise ValueError("repair rank must be non-negative")
        if repair_initialization not in {"zero", "warm_start"}:
            raise ValueError("repair_initialization must be 'zero' or 'warm_start'")
        self.repair_initialization = repair_initialization
        if self.repair_rank:
            self.repair_down = nn.Linear(self.bus_dim, self.repair_rank, bias=False)
            self.repair_up = nn.Linear(self.repair_rank, self.donor_width, bias=False)
            # ``zero`` preserves the original v1 pilot exactly.  It is a
            # strict no-op initialization, but also makes a two-factor port
            # temporarily gradient-dead.  ``warm_start`` keeps the same
            # zero-valued function while giving the up projection a gradient
            # at step one; it is used only by later micro-repair curves.
            if repair_initialization == "zero":
                nn.init.zeros_(self.repair_down.weight)
            else:
                nn.init.normal_(self.repair_down.weight, mean=0.0, std=1.0 / max(self.bus_dim, 1) ** 0.5)
            nn.init.zeros_(self.repair_up.weight)

    def _donor_input(self, bus: torch.Tensor) -> torch.Tensor:
        if bus.shape[-1] != self.bus_dim:
            raise ValueError(f"expected bus width {self.bus_dim}, got {bus.shape[-1]}")
        donor_input = bus.float() @ self.input_basis.transpose(0, 1)
        if self.repair_rank:
            donor_input = donor_input + self.repair_up(self.repair_down(bus.float()))
        return donor_input

    def forward(self, bus: torch.Tensor) -> torch.Tensor:
        donor_input = self._donor_input(bus)
        return donor_input @ self.donor_weight.transpose(0, 1)

    def port_parameters(self) -> list[nn.Parameter]:
        if not self.repair_rank:
            return []
        return list(self.repair_down.parameters()) + list(self.repair_up.parameters())

    @property
    def donor_parameter_count(self) -> int:
        return int(self.donor_weight.numel())

    @property
    def analytic_parameter_count(self) -> int:
        return int(self.input_basis.numel())

    @property
    def trainable_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.port_parameters()))

    def interface_signature(self) -> dict[str, Any]:
        return {
            "version": self.interface_version,
            "input": ["batch", "time", self.bus_dim],
            "output": ["batch", "time", len(self.selected_rows)],
            "donor_width": self.donor_width,
            "selected_rows": list(self.selected_rows),
            "donor_route_count": self.donor_route_count,
            "donor_core_frozen": True,
            "donor_variant": self.donor_variant,
            "analytic_basis": "top-right-singular-vectors",
            "repair_rank": self.repair_rank,
            "repair_initialization": self.repair_initialization,
        }


class QwenCompactRouterOrgan(nn.Module):
    """A precomputed width-converted router with a frozen donor core.

    ``compact_weight`` is normally ``donor_weight @ input_basis``.  The
    multiplication is performed once during bounded offline surgery, so the
    runtime path is the ordinary Remora-width affine map rather than a
    donor-width expansion followed by a second large matmul.  This class is
    deliberately separate from :class:`QwenRouterOrgan`: its resident core is
    an analytically transformed foreign tensor, not the original full-width
    tensor.

    The optional repair is an output-logit low-rank port.  ``warm_start``
    keeps the initial function unchanged while making the up projection
    trainable at the first optimization step.
    """

    interface_version = "qwen-compact-router-organ-v1"

    def __init__(
        self,
        compact_weight: torch.Tensor,
        *,
        donor_route_count: int | None = None,
        donor_variant: str = "actual",
        variant_seed: int = 0,
        trainable_repair_rank: int = 0,
        repair_initialization: str = "zero",
    ):
        super().__init__()
        if compact_weight.ndim != 2:
            raise ValueError("compact router must be a matrix")
        if donor_route_count is None:
            donor_route_count = int(compact_weight.shape[0])
        if int(donor_route_count) != int(compact_weight.shape[0]):
            raise ValueError("donor route count must match compact router rows")
        if repair_initialization not in {"zero", "warm_start"}:
            raise ValueError("repair_initialization must be 'zero' or 'warm_start'")
        self.donor_variant = str(donor_variant)
        self.donor_variant_seed = int(variant_seed)
        self.donor_route_count = int(donor_route_count)
        self.bus_dim = int(compact_weight.shape[1])
        self.repair_rank = int(trainable_repair_rank)
        if self.repair_rank < 0:
            raise ValueError("repair rank must be non-negative")
        self.repair_initialization = repair_initialization
        self.register_buffer("compact_weight", compact_weight.detach().cpu().float().contiguous())
        if self.repair_rank:
            self.repair_down = nn.Linear(self.bus_dim, self.repair_rank, bias=False)
            self.repair_up = nn.Linear(self.repair_rank, self.donor_route_count, bias=False)
            if repair_initialization == "zero":
                nn.init.zeros_(self.repair_down.weight)
            else:
                nn.init.normal_(self.repair_down.weight, mean=0.0, std=1.0 / max(self.bus_dim, 1) ** 0.5)
            nn.init.zeros_(self.repair_up.weight)

    def forward(self, bus: torch.Tensor) -> torch.Tensor:
        if bus.shape[-1] != self.bus_dim:
            raise ValueError(f"expected bus width {self.bus_dim}, got {bus.shape[-1]}")
        logits = bus.float() @ self.compact_weight.transpose(0, 1)
        if self.repair_rank:
            logits = logits + self.repair_up(self.repair_down(bus.float()))
        return logits

    def port_parameters(self) -> list[nn.Parameter]:
        if not self.repair_rank:
            return []
        return list(self.repair_down.parameters()) + list(self.repair_up.parameters())

    @property
    def donor_parameter_count(self) -> int:
        return int(self.compact_weight.numel())

    @property
    def analytic_parameter_count(self) -> int:
        return 0

    @property
    def trainable_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.port_parameters()))

    def interface_signature(self) -> dict[str, Any]:
        return {
            "version": self.interface_version,
            "input": ["batch", "time", self.bus_dim],
            "output": ["batch", "time", self.donor_route_count],
            "donor_width": self.bus_dim,
            "donor_route_count": self.donor_route_count,
            "donor_core_frozen": True,
            "donor_variant": self.donor_variant,
            "conversion": "precomputed-analytic-width-conversion",
            "repair_rank": self.repair_rank,
            "repair_initialization": self.repair_initialization,
        }


def router_functional_metrics(
    full_weight: torch.Tensor,
    basis: torch.Tensor,
    *,
    selected_rows: tuple[int, ...] | list[int],
    inputs: torch.Tensor,
) -> dict[str, float]:
    """Compare full donor rows with the analytically converted organ."""

    rows = list(selected_rows)
    reference = inputs.float() @ full_weight[rows].float().transpose(0, 1)
    converted = (inputs.float() @ basis.float()) @ (full_weight[rows].float() @ basis.float()).transpose(0, 1)
    delta = converted - reference
    ref_flat = reference.reshape(-1)
    delta_flat = delta.reshape(-1)
    return {
        "max_absolute_error": float(delta.abs().max()),
        "mean_absolute_error": float(delta.abs().mean()),
        "relative_l2_error": float(torch.linalg.vector_norm(delta_flat) / torch.linalg.vector_norm(ref_flat).clamp_min(1e-30)),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(converted.reshape(1, -1), reference.reshape(1, -1)).item()),
        "reference_l2": float(torch.linalg.vector_norm(ref_flat)),
    }


def serialize_router_metadata(
    *,
    source_model: str,
    source_revision: str,
    source_shard: str,
    source_tensor: str,
    selected_rows: tuple[int, ...],
    donor_weight: torch.Tensor,
    input_basis: torch.Tensor,
) -> dict[str, Any]:
    return {
        "source_model": source_model,
        "source_revision": source_revision,
        "source_shard": source_shard,
        "source_tensor": source_tensor,
        "selected_rows": list(selected_rows),
        "source_dtype": "BF16",
        "retained_donor_parameters": int(donor_weight.numel()),
        "discarded_donor_parameters": int(512 * 2560 - donor_weight.numel()),
        "analytic_basis_shape": list(input_basis.shape),
        "analytic_basis_parameters": int(input_basis.numel()),
        "runtime_donor_core_dtype": "FP32",
        "runtime_donor_core_bytes": int(donor_weight.numel() * 4),
    }
