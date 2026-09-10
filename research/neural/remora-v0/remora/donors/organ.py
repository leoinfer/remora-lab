from __future__ import annotations

"""Value-free organ selection built on top of the donor anatomy graph."""

from typing import Any


def select_anatomy_component(
    graph: dict[str, Any],
    component_id: str,
    *,
    max_payload_bytes: int = 256 * 1024 * 1024,
) -> dict[str, Any]:
    """Create an extraction manifest without consulting tensor values."""

    components = graph.get("components", [])
    matches = [item for item in components if item.get("component_id") == component_id]
    if len(matches) != 1:
        raise KeyError(f"expected one anatomy component {component_id!r}, found {len(matches)}")
    component = matches[0]
    payload_bytes = int(component["payload_bytes"])
    if payload_bytes > int(max_payload_bytes):
        raise ValueError(
            f"component {component_id!r} exceeds extraction budget: {payload_bytes} > {int(max_payload_bytes)}"
        )
    return {
        "schema": "remora-qwen-neural-organ-selection-v1",
        "selection_is_value_free": True,
        "donor_weights_consulted": False,
        "selection_basis": "architecture/header evidence only: bounded component with explicit tensor boundary, finite computation contract, and manageable extraction cost",
        "max_payload_bytes": int(max_payload_bytes),
        "selected_payload_bytes": payload_bytes,
        "selected": [
            {
                "component_id": component["component_id"],
                "component_key": component["component_key"],
                "architecture_family": component["architecture_family"],
                "tensor_names": list(component["tensor_names"]),
                "payload_bytes": payload_bytes,
                "source_shards": list(component["source_shards"]),
                "geometry": component["geometry"],
                "declared_import_modes": component["compatibility"]["candidate_import_modes"],
            }
        ],
        "promotion_state": "DONOR_OBSERVED",
    }
