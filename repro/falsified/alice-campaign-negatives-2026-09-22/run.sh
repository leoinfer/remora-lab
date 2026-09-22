#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
receipt="$repo_root/repro/falsified/alice-campaign-negatives-2026-09-22/sanitized_receipt.json"
printf 'lane=repro/falsified/alice-campaign-negatives-2026-09-22\n'
printf 'status=EXCLUDED_WEIGHTS_DATA\n'
printf 'receipt=%s\n' "$receipt"
printf 'f2_verdict=NULL\n'
printf 'cpu_moe_prefill_regression=-0.851\n'
printf 'coarse_double_staging_failure_mib=49326.56\n'
printf 'alice_moe_block_verdict=REVERT\n'
printf 'target_100k_raw_decode=CLOSED\n'
printf 'structured_sparsity=CLOSED\n'
printf 'laya_agreement=0.384_vs_baseline_0.849\n'
printf 'weights_published=false\n'
printf 'reason=these are measured rejections; weights and the executing runtime trees are excluded from this repository\n'
