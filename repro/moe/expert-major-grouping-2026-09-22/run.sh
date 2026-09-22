#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
receipt="$repo_root/repro/moe/expert-major-grouping-2026-09-22/sanitized_receipt.json"
printf 'lane=repro/moe/expert-major-grouping-2026-09-22\n'
printf 'status=EXCLUDED_WEIGHTS_DATA\n'
printf 'receipt=%s\n' "$receipt"
printf 'tmac_per_s_before=0.185\n'
printf 'tmac_per_s_after=6.115\n'
printf 'speedup=33.0\n'
printf 'same_kernel_parity=32/32_bit_exact\n'
printf 'deployed_configuration_gate=REVERT\n'
printf 'amdahl_ceiling=1.652\n'
printf 'cpu_moe_prefill_regression=-0.851\n'
printf 'weights_published=false\n'
printf 'reason=expert slabs and the instrumenting runtime are excluded from this repository\n'
