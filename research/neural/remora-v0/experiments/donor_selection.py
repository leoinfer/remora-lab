from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.donors.registry import DonorRegistry
from remora.donors.selection import select_components
from remora.ledger import record_experiment
from remora.utils import write_json


def run(
    manifest_path: str | Path,
    preferred_classes: list[str],
    max_payload_bytes: int = 256 * 1024 * 1024,
    max_components: int = 4,
    output: str | Path | None = None,
) -> dict:
    outer = json.loads(Path(manifest_path).read_text())
    manifest = outer.get("full_manifest", outer)
    selection = select_components(manifest, preferred_classes, max_payload_bytes, max_components)
    registry = DonorRegistry()
    donor_id = registry.register_manifest(manifest)
    registry.propose_candidate(
        donor_id,
        f"{donor_id}-component-candidate",
        selection["recommended_import"],
        "donor-component-candidate",
        "donor-port-v1",
        {"selected_component_keys": [row["component_key"] for row in selection["selected"]]},
    )
    result = {
        "schema": "remora-v0-donor-selection-result",
        "manifest": str(manifest_path),
        "selection": selection,
        "registry": registry.to_dict(),
        "interpretation": "MEASURED MECHANISM CONTROL: donor component selection is deterministic and value-free; no selected tensor has been copied or promoted.",
    }
    if output:
        write_json(output, result)
    record_experiment(
        ROOT,
        "DONOR-SELECTION-003",
        "A heterogeneous resident donor can be narrowed to complete architectural component groups under a byte budget before any extraction is attempted.",
        "Group parsed header inventory by layer/component namespace, select preferred classes within a fixed payload budget, and register the result as an unpromoted candidate.",
        "Selection is deterministic, stays within budget, preserves whole component groups, and recommends a compatible import mode.",
        "Selection copies tensor values, chooses arbitrary parameter fragments, exceeds the budget, or promotes the candidate without an external decision.",
        f"python -m experiments.donor_selection --manifest {manifest_path}",
        0,
        {"selection": selection, "registry": registry.to_dict()},
        result["interpretation"],
        "Use the selected groups to design a hidden-state/response extraction query; only then authorize a bounded runtime and evaluator.",
        hardware={"mode": "metadata_only"},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--classes", default="recurrent_or_gated_linear_attention,residual_hyperconnection")
    parser.add_argument("--max-payload-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--max-components", type=int, default=4)
    parser.add_argument("--output", default=str(ROOT / "results" / "donor-selection.json"))
    args = parser.parse_args()
    result = run(args.manifest, [x for x in args.classes.split(",") if x], args.max_payload_bytes, args.max_components, args.output)
    print(json.dumps({"schema": result["schema"], "selection": result["selection"], "registry": result["registry"]}, indent=2))


if __name__ == "__main__":
    main()
