#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
receipt="$repo_root/repro/alice/artifact-and-residency-2026-09-22/sanitized_receipt.json"
printf 'lane=repro/alice/artifact-and-residency-2026-09-22\n'
printf 'status=EXCLUDED_WEIGHTS_DATA\n'
printf 'receipt=%s\n' "$receipt"
printf 'total_params_llamacpp_accounting=79635961472\n'
printf 'blocks=48\n'
printf 'experts_per_layer=512\n'
printf 'experts_per_token=10\n'
printf 'artifact_bytes=39913721760\n'
printf 'effective_bpw=4.003\n'
printf 'expert_share_of_artifact=0.9341\n'
printf 'decode_tps_cold_mmap=4.49\n'
printf 'decode_tps_arena_12gib_pinned=15.58\n'
printf 'arena_hit_rate=0.95308\n'
printf 'arena_destructive_parity_max_abs_diff=0.0\n'
printf 'weights_published=false\n'
printf 'reason=model weights and the executing runtime tree are excluded from this repository\n'
