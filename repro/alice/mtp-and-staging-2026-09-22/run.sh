#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
receipt="$repo_root/repro/alice/mtp-and-staging-2026-09-22/sanitized_receipt.json"
printf 'lane=repro/alice/mtp-and-staging-2026-09-22\n'
printf 'status=EXCLUDED_WEIGHTS_DATA\n'
printf 'receipt=%s\n' "$receipt"
printf 'kat_failures_before_fix=276\n'
printf 'kat_checks_total=682\n'
printf 'kat_failures_after_fix=0\n'
printf 'greedy_parity_K=0,2,3,4\n'
printf 'overlap_probe_separate_stream_exposed_ms=0.00-0.60\n'
printf 'f2_control_hot_median_pp_tps=515.858\n'
printf 'f2_candidate_hot_median_pp_tps=507.085\n'
printf 'f2_verdict=NULL\n'
printf 'sync_drain_share_of_prefill_pass=0.002\n'
printf 'weights_published=false\n'
printf 'reason=model weights and the executing runtime tree are excluded from this repository\n'
