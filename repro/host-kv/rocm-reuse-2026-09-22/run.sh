#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
receipt="$repo_root/repro/host-kv/rocm-reuse-2026-09-22/sanitized_receipt.json"
printf 'lane=repro/host-kv/rocm-reuse-2026-09-22\n'
printf 'status=EXCLUDED_WEIGHTS_DATA\n'
printf 'receipt=%s\n' "$receipt"
printf 'pinned_k1_physical_gbps=13.997\n'
printf 'pinned_k96_logical_gbps=244.944\n'
printf 'pageable_registered_k96_logical_gbps=246.89\n'
printf 'parity_rel_rms=4.511e-06..4.955e-06\n'
printf 'parity_max_abs=2.3935e-07\n'
printf 'registered_host_memory=correctness_prerequisite\n'
printf 'logical_is_not_physical=true\n'
printf 'weights_published=false\n'
printf 'reason=kernel source, run logs, and the executing runtime tree are excluded from this repository\n'
