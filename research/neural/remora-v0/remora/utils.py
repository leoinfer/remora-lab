from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"cuda", "gpu", "auto"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_hash(cwd: str | Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNCOMMITTED_OR_UNKNOWN"


def runtime_context(device: torch.device) -> dict:
    ctx = {
        "timestamp_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "pid": os.getpid(),
    }
    if torch.cuda.is_available():
        ctx["gpu_name"] = torch.cuda.get_device_name(0)
        ctx["hip_version"] = getattr(torch.version, "hip", None)
        ctx["vram_allocated_bytes"] = int(torch.cuda.memory_allocated())
        ctx["vram_reserved_bytes"] = int(torch.cuda.memory_reserved())
    return ctx


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(
        p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only)
    )


def parameter_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().cpu().clone() for name, p in module.named_parameters()}


def changed_parameter_stats(
    before: dict[str, torch.Tensor], module: torch.nn.Module, atol: float = 1e-12
) -> dict:
    changed = 0
    total = 0
    per_module: dict[str, dict[str, int | float]] = {}
    for name, p in module.named_parameters():
        old = before.get(name)
        if old is None:
            continue
        delta = (p.detach().cpu() - old).abs()
        n = int(p.numel())
        c = int((delta > atol).sum())
        total += n
        changed += c
        prefix = name.split(".")[0]
        item = per_module.setdefault(prefix, {"changed": 0, "total": 0, "l1": 0.0})
        item["changed"] += c
        item["total"] += n
        item["l1"] += float(delta.sum())
    return {
        "changed_parameters": changed,
        "total_parameters": total,
        "changed_fraction": changed / total if total else 0.0,
        "per_top_level_module": per_module,
    }


def tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=json_default)
        f.write("\n")


def append_jsonl(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        json.dump(record, f, sort_keys=True, default=json_default)
        f.write("\n")


def freeze_all(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_prefixes(module: torch.nn.Module, prefixes: Iterable[str]) -> list[str]:
    prefixes = tuple(prefixes)
    selected = []
    for name, p in module.named_parameters():
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            p.requires_grad = True
            selected.append(name)
    return selected
