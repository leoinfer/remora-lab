#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
receipt="$repo_root/repro/qwen27b/historical-baseline/sanitized_receipt.json"
printf 'lane=qwen27b/historical-baseline\n'
printf 'status=UNRECOVERABLE_HISTORICAL_RESULT\n'
printf 'receipt=%s\n' "$receipt"
printf 'throughput_tokens_per_s=NOT_ASSERTED\n'
printf 'reason=exact public model, executor, prompt, timing receipt, and correctness gate were not cleared\n'
