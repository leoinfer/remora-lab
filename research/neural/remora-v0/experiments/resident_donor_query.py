from __future__ import annotations

"""Query one explicitly selected resident donor and emit verified records.

This is intentionally opt-in. It is safe to run the rest of the Remora
repository without importing Transformers or loading any donor weights.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remora.donors.response import write_response_records
from remora.donors.runtime import DonorRuntimeSpec, LocalTransformersDonor
from remora.ledger import record_experiment, record_failure
from remora.utils import choose_device, runtime_context, write_json


_PROMPT_EXPECTATIONS = {
    "Compute 2 + 3. Answer with only the integer.": 5,
    "Compute 7 + 8. Answer with only the integer.": 15,
    "Compute 12 + 30. Answer with only the integer.": 42,
    "Compute 19 + 6. Answer with only the integer.": 25,
    "Compute 40 + 2. Answer with only the integer.": 42,
    "Compute 55 + 14. Answer with only the integer.": 69,
    "Compute 81 + 9. Answer with only the integer.": 90,
    "Compute 100 + 23. Answer with only the integer.": 123,
}


def _verify_math(prompt: str, response: str) -> bool:
    expected = _PROMPT_EXPECTATIONS.get(prompt)
    match = re.fullmatch(r"\s*(\d+)\s*", response)
    return expected is not None and match is not None and int(match.group(1)) == expected


def run(
    model_path: str,
    *,
    runtime_id: str,
    experiment_id: str = "DONOR-RUNTIME-001",
    device_name: str = "auto",
    max_new_tokens: int = 8,
    trust_remote_code: bool = False,
    output: str | Path | None = None,
    records_output: str | Path | None = None,
) -> dict:
    device = choose_device(device_name)
    spec = DonorRuntimeSpec(
        model_path=model_path,
        runtime_id=runtime_id,
        device=str(device),
        max_new_tokens=max_new_tokens,
        trust_remote_code=trust_remote_code,
        allow_model_load=True,
    )
    donor = LocalTransformersDonor(spec)
    prompts = list(_PROMPT_EXPECTATIONS)
    records = donor.response_records(
        prompts,
        donor_id=f"resident-{Path(model_path).name}",
        verifier_id="external-integer-addition-v1",
        verifier=_verify_math,
        lineage_key=f"{runtime_id}:math-probe-v1",
    )
    if records_output:
        write_response_records(records_output, records)
    accepted = sum(record.accepted for record in records)
    interpretation = (
        "MEASURED RUNTIME PROBE: one explicitly selected resident open-weight model was queried locally with bounded greedy generation; accepted records are verifier-gated and are not automatically distilled or promoted."
        if accepted
        else
        "MEASURED NEGATIVE RUNTIME PROBE: the explicitly selected resident open-weight model was queryable with bounded greedy generation, but zero responses passed the external verifier; no donor evidence was eligible for distillation or promotion."
    )
    result = {
        "schema": "remora-v0-resident-donor-query-result",
        "mode": "EXPLICIT_LOCAL_TRANSFORMERS_DONOR_QUERY",
        "model_path": str(Path(model_path).expanduser().resolve()),
        "runtime_id": runtime_id,
        "experiment_id": experiment_id,
        "config_repairs": list(donor.config_repairs),
        "device": str(device),
        "prompt_count": len(records),
        "accepted_count": accepted,
        "records_path": str(records_output) if records_output else None,
        "responses": [
            {
                "record_id": record.record_id,
                "prompt": record.prompt,
                "response": record.response,
                "accepted": record.accepted,
            }
            for record in records
        ],
        "verifier": "external-integer-addition-v1",
        "promotion": {
            "state": "CANDIDATE_RESPONSE_CAPTURED",
            "promoted": False,
            "decision_authority": "external Remora experiment harness",
        },
        "interpretation": interpretation,
        "runtime": runtime_context(device),
    }
    if output:
        write_json(output, result)
    record_experiment(
        ROOT,
        experiment_id,
        "A resident open-weight model can provide bounded, provenance-rich responses to Remora without being silently loaded by inspection or allowed to promote a candidate.",
        "Explicitly load the user-selected local Transformers model with local_files_only, bounded greedy generation, and an external integer-addition verifier; write donor-response-v1 records.",
        "The runtime loads only after explicit opt-in, respects the generation bound, records hashes/runtime lineage, and leaves promotion false; verifier-passing responses are reusable evidence.",
        "A model loads without explicit opt-in, network files are fetched, generation exceeds the bound, records lack runtime/hash lineage, or the donor/candidate self-promotes.",
        "python -m experiments.resident_donor_query --model-path <local-model> --runtime-id <id> --allow-model-load",
        0,
        {
            "model_path": result["model_path"],
            "runtime_id": runtime_id,
            "prompt_count": len(records),
            "accepted_count": accepted,
            "config_repairs": list(donor.config_repairs),
            "records_path": result["records_path"],
        },
        result["interpretation"],
        "Use accepted records only in a separately declared response-distillation experiment; collect hidden states through donor-activation-v1 only when the runtime exposes an explicit layer hook.",
        hardware=runtime_context(device),
    )
    if accepted == 0:
        record_failure(
            ROOT,
            experiment_id,
            {"model_path": result["model_path"], "runtime_id": runtime_id},
            {"probe": "strict_integer_addition", "prompt_count": len(records), "max_new_tokens": max_new_tokens},
            0,
            "zero of the strict integer-addition probe responses passed the external verifier",
            "The resident model is executable through the bounded runtime boundary, but the selected prompt/verifier pair did not yield reusable behavior; this may reflect prompt format, model family, or runtime compatibility.",
            "Retest with the donor's documented chat template and a separately validated task before distillation; do not accept unverified text as training evidence.",
            runtime={"device": str(device), "accepted_count": accepted, "runtime_id": runtime_id},
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--experiment-id", default="DONOR-RUNTIME-001")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-model-load", action="store_true", help="required safety acknowledgment")
    parser.add_argument("--output", default=str(ROOT / "results" / "resident-donor-query.json"))
    parser.add_argument("--records-output", default=str(ROOT / "results" / "resident-donor-response-records.jsonl"))
    args = parser.parse_args()
    if not args.allow_model_load:
        raise SystemExit("refusing resident model load: pass --allow-model-load explicitly")
    result = run(
        args.model_path,
        runtime_id=args.runtime_id,
        device_name=args.device,
        max_new_tokens=args.max_new_tokens,
        trust_remote_code=args.trust_remote_code,
        output=args.output,
        records_output=args.records_output,
        experiment_id=args.experiment_id,
    )
    print(json.dumps({
        "interpretation": result["interpretation"],
        "model_path": result["model_path"],
        "accepted_count": result["accepted_count"],
        "prompt_count": result["prompt_count"],
        "promotion": result["promotion"],
    }, indent=2))


if __name__ == "__main__":
    main()
