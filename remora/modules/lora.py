from __future__ import annotations

"""Small, explicit LoRA modules used by the adversarial control arms.

The wrapper keeps the original linear layer as a frozen submodule and adds a
zero-initialized low-rank residual.  This makes the parameter accounting
auditable: base parameters remain present and frozen, while only the two
factor matrices are trainable during lifetime adaptation.
"""

from collections.abc import Callable, Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank residual."""

    interface_version = "lora-linear-v1"

    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / self.rank
        # Dynamic surgery may wrap a layer after the host model has already
        # moved to an accelerator.  Allocate the new factors beside the base
        # weight so the wrapper is immediately executable; relying on a later
        # model-wide ``.to(...)`` is not safe because callers may insert the
        # wrapper into an already-placed model.
        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                base.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base.out_features,
                self.rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.lora_A), self.lora_B
        )
        return self.base(x) + self.scaling * residual

    def interface_signature(self) -> dict:
        return {
            "version": self.interface_version,
            "input_dim": self.base.in_features,
            "output_dim": self.base.out_features,
            "rank": self.rank,
            "alpha": self.alpha,
        }


def _set_child_module(root: nn.Module, qualified_name: str, replacement: nn.Module) -> None:
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    leaf = parts[-1]
    if leaf not in parent._modules:
        raise KeyError(f"module path is not a child module: {qualified_name}")
    setattr(parent, leaf, replacement)


def replace_linear_with_lora(
    model: nn.Module,
    predicate: Callable[[str, nn.Linear], bool],
    *,
    rank: int,
    alpha: float | None = None,
) -> list[str]:
    """Replace selected linear leaves and return their original qualified names."""

    selected: list[str] = []
    for name, module in list(model.named_modules()):
        if name and isinstance(module, nn.Linear) and predicate(name, module):
            _set_child_module(model, name, LoRALinear(module, rank=rank, alpha=alpha))
            selected.append(name)
    return selected


def matching_suffixes(suffixes: Iterable[str]) -> Callable[[str, nn.Linear], bool]:
    suffixes = tuple(suffixes)
    if not suffixes:
        raise ValueError("at least one LoRA suffix is required")
    return lambda name, _module: any(name.endswith(suffix) for suffix in suffixes)


def lora_parameter_names(model: nn.Module) -> list[str]:
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and (".lora_A" in name or ".lora_B" in name)
    ]
