from __future__ import annotations

import time
from typing import Callable, Iterable

import torch
from torch import nn

from ..utils import freeze_all, unfreeze_prefixes


def trainable_parameter_names(module: nn.Module) -> list[str]:
    return [name for name, p in module.named_parameters() if p.requires_grad]


def adapt_module(
    module: nn.Module,
    loss_fn: Callable[[], torch.Tensor],
    prefixes: Iterable[str],
    steps: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> list[dict]:
    """Freeze everything except declared parameter islands and optimize locally."""
    freeze_all(module)
    selected = unfreeze_prefixes(module, prefixes)
    if not selected:
        raise ValueError(f"no parameters matched prefixes {list(prefixes)}")
    optimizer = torch.optim.AdamW(
        [p for p in module.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )
    history = []
    for step in range(1, steps + 1):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in module.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        history.append({"step": step, "loss": float(loss.detach()), "wall_seconds": time.perf_counter() - started})
    return history
