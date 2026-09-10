from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.donors.extract import extract_selected_tensors, write_extraction_receipt
from remora.ledger import record_experiment


def run(
    manifest_path: str | Path,
    selection_path: str | Path,
    output_path: str | Path,
    result_path: str | Path,
    *,
    allow_payload: bool = False,
    max_payload_bytes: int = 256 * 1024 * 1024,
) -> dict:
    outer_manifest = json.loads(Path(manifest_path).read_text())
    manifest = outer_manifest.get("full_manifest", outer_manifest)
    selection_outer = json.loads(Path(selection_path).read_text())
    selection = selection_outer.get("selection", selection_outer)
    receipt = extract_selected_tensors(
        manifest.get("path"),
        manifest,
        selection,
        output_path,
        allow_payload=allow_payload,
        max_payload_bytes=max_payload_bytes,
    )
    result = {
        "schema": "remora-v0-donor-extraction-result",
        "manifest": str(manifest_path),
        "selection": str(selection_path),
        "extraction": receipt,
        "interpretation": "MEASURED MECHANISM CONTROL: only the explicitly selected, byte-bounded donor tensors were materialized; no full model loader or candidate promotion was performed.",
    }
    write_extraction_receipt(result_path, result)
    record_experiment(
        ROOT,
        "DONOR-EXTRACTION-001",
        "A resident donor can be opened surgically after a value-free compatibility and budget gate.",
        "Materialize only the selected tensor names with safetensors under an explicit payload opt-in and fixed byte budget; write a standalone receipt.",
        "The selected tensor set is complete, the observed bytes match the header accounting, the output is outside the source tree, and the candidate remains unpromoted.",
        "Extraction occurs without explicit opt-in, reads tensors outside the selection, exceeds the budget, mismatches byte accounting, or promotes the candidate.",
        f"python -m experiments.donor_extract --manifest {manifest_path} --selection {selection_path} --output {output_path} --result {result_path} --allow-payload",
        0,
        {"extraction": receipt},
        result["interpretation"],
        "Use the extracted bundle only in a license-reviewed frozen-teacher/port experiment; do not direct-graft it into Remora-v0.",
        hardware={"mode": "bounded_selective_payload", "model_loader_called": False},
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Explicitly extract a bounded donor tensor selection.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output", required=True, help="Standalone .safetensors output; must be outside donor tree.")
    parser.add_argument("--result", default=str(ROOT / "results" / "donor-extraction.json"))
    parser.add_argument("--allow-payload", action="store_true", help="Required opt-in to materialize tensor values.")
    parser.add_argument("--max-payload-bytes", type=int, default=256 * 1024 * 1024)
    args = parser.parse_args()
    result = run(
        args.manifest,
        args.selection,
        args.output,
        args.result,
        allow_payload=args.allow_payload,
        max_payload_bytes=args.max_payload_bytes,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
