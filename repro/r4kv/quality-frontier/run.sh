#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
receipt="$repo_root/repro/r4kv/quality-frontier/sanitized_receipt.json"
printf 'lane=r4kv/quality-frontier\n'
printf 'status=BLOCKED_PROVENANCE\n'
printf 'receipt=%s\n' "$receipt"
printf 'quality_metric=NOT_ASSERTED\n'
printf 'reason=storage KAT is not model quality and no cleared model-quality receipt is public\n'
