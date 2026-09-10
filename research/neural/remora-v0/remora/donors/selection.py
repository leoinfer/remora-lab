from __future__ import annotations

"""Deterministic, value-free selection of donor component groups."""

from typing import Any, Iterable


def component_key(name: str) -> str:
    parts = name.split(".")
    for container in ("layers", "blocks"):
        if container not in parts:
            continue
        index = parts.index(container)
        if index + 2 >= len(parts):
            continue

        # Keep the layer/block, implementation namespace, and (where present)
        # expert identity.  The parameter leaf is deliberately excluded:
        # ``...layers.0.linear_attn.A_log`` and ``...layers.0.linear_attn.out_proj``
        # must be one selectable unit, while ``...mlp.experts.3.*`` remains a
        # separately replaceable expert.
        prefix = parts[: index + 2]
        tail = parts[index + 2 :]
        if not tail:
            return ".".join(prefix)
        key_parts = prefix + [tail[0]]
        if (
            tail[0] in {"mlp", "moe"}
            and len(tail) > 2
            and tail[1] in {"experts", "expert", "shared_experts"}
            and tail[2].isdigit()
        ):
            key_parts.extend(tail[1:3])
        elif tail[0] in {"experts", "expert"} and len(tail) > 1 and tail[1].isdigit():
            key_parts.append(tail[1])
        return ".".join(key_parts)
    return ".".join(parts[: max(1, min(3, len(parts)))])


def group_components(tensor_inventory: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for tensor in tensor_inventory:
        key = component_key(str(tensor["name"]))
        row = groups.setdefault(key, {
            "component_key": key,
            "tensor_class": tensor.get("tensor_class", "other"),
            "tensor_names": [],
            "payload_bytes": 0,
        })
        row["tensor_names"].append(tensor["name"])
        row["payload_bytes"] += int(tensor.get("nbytes") or 0)
        if row["tensor_class"] != tensor.get("tensor_class", "other"):
            row["tensor_class"] = "mixed"
    return sorted(groups.values(), key=lambda row: (row["tensor_class"], row["component_key"]))


def select_components(
    manifest: dict[str, Any],
    preferred_classes: Iterable[str],
    max_payload_bytes: int,
    max_components: int = 4,
) -> dict[str, Any]:
    inventory = manifest.get("headers", {}).get("tensor_inventory", [])
    if not inventory:
        raise ValueError("component selection requires parsed tensor headers")
    preferred = list(dict.fromkeys(preferred_classes))
    rank = {name: index for index, name in enumerate(preferred)}
    groups = [group for group in group_components(inventory) if group["tensor_class"] in rank]
    groups.sort(key=lambda row: (rank[row["tensor_class"]], row["component_key"]))
    selected = []
    used = 0
    skipped = []
    for group in groups:
        if len(selected) >= max_components:
            skipped.append({"component_key": group["component_key"], "reason": "max_components"})
            continue
        if used + group["payload_bytes"] > max_payload_bytes:
            skipped.append({"component_key": group["component_key"], "reason": "byte_budget", "payload_bytes": group["payload_bytes"]})
            continue
        selected.append(group)
        used += group["payload_bytes"]
    compatible = manifest.get("compatibility", {}).get("status") == "DIRECT_GRAFT_SHAPE_COMPATIBLE"
    return {
        "schema": "remora-donor-selection-v1",
        "preferred_classes": preferred,
        "max_payload_bytes": int(max_payload_bytes),
        "max_components": int(max_components),
        "selected": selected,
        "skipped": skipped,
        "selected_payload_bytes": used,
        "selection_is_value_free": True,
        "recommended_import": "named_tensor_graft_with_ab_tests" if compatible else "frozen_teacher_or_activation_distillation",
    }
