from __future__ import annotations

import torch


def route_summary(weights: list[torch.Tensor]) -> dict:
    if not weights:
        return {}
    loads = torch.stack([w.detach().mean(dim=(0, 1)).cpu() for w in weights])
    return {
        "per_layer_load": loads.tolist(),
        "mean_load": loads.mean(dim=0).tolist(),
        "load_entropy": float((-(loads.clamp_min(1e-8) * loads.clamp_min(1e-8).log()).sum(-1)).mean()),
    }
