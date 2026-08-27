#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
receipt="$repo_root/repro/flash-next/full-model/sanitized_receipt.json"
printf 'lane=flash-next/full-model\n'
printf 'status=BLOCKED_PROVENANCE\n'
printf 'receipt=%s\n' "$receipt"
printf 'first_token_gate=NOT_CLEARED\n'
printf 'generation_tokens_per_s=NOT_ASSERTED\n'
printf 'reason=full-model first-token and generation evidence remains incomplete\n'
