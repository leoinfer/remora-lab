from __future__ import annotations

import torch

from ..utils import changed_parameter_stats, parameter_snapshot


def parameter_update_fraction(before: dict[str, torch.Tensor], module: torch.nn.Module) -> float:
    return float(changed_parameter_stats(before, module)["changed_fraction"])


def consolidation_report(
    before: dict[str, torch.Tensor], module: torch.nn.Module, provenance_ids: list[str]
) -> dict:
    stats = changed_parameter_stats(before, module)
    return {"parameter_update": stats, "provenance_ids": provenance_ids, "retrieval_dependency": "disabled_after_consolidation"}
