#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
receipt="$repo_root/repro/flash-next/deployment-2026-09-18/sanitized_receipt.json"
printf 'lane=flash-next/deployment-2026-09-18\n'
printf 'status=EXCLUDED_WEIGHTS_DATA\n'
printf 'receipt=%s\n' "$receipt"
printf 'model_payload_published=false\n'
printf 'runtime_branch_published=false\n'
printf 'deployment_coherent=true\n'
printf 'canaries_correct=12/12\n'
printf 'context_tokens=262144\n'
printf 'prefetch_decode_tps=3.86,4.07\n'
printf 'quality_retention=NOT_MEASURED\n'
printf 'final_v2=NOT_ASSERTED\n'
printf 'reason=measurements are published as sanitized values; reproduction inputs are excluded from this repository\n'
