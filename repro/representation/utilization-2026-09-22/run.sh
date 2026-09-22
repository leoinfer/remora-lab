#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
receipt="$repo_root/repro/representation/utilization-2026-09-22/sanitized_receipt.json"
printf 'lane=repro/representation/utilization-2026-09-22\n'
printf 'status=EXCLUDED_WEIGHTS_DATA\n'
printf 'receipt=%s\n' "$receipt"
printf 'unpack_int8_resident_gbps=297.6\n'
printf 'unpack_packed4_gbps=229.7\n'
printf 'unpack_w4x_symmetric_gbps=188.8\n'
printf 'pc4=PROJECTION_ONLY\n'
printf 'gsq_rco_synthesis=NOT_IMPLEMENTED\n'
printf 'flash_target=MODELED\n'
printf 'quality_is_not_weight_mse=true\n'
printf 'weights_published=false\n'
printf 'reason=weights, activation captures, and the panel scripts are excluded from this repository\n'
