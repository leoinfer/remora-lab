#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
receipt="$repo_root/repro/alice/backend-speed-and-prefill-2026-09-22/sanitized_receipt.json"
printf 'lane=repro/alice/backend-speed-and-prefill-2026-09-22\n'
printf 'status=EXCLUDED_WEIGHTS_DATA\n'
printf 'receipt=%s\n' "$receipt"
printf 'vulkan_decode_tps_historical_config_of_record=18.430\n'
printf 'vulkan_decode_tps_independent_reanchor=18.360\n'
printf 'hip_raw_k0_decode_tps=15.374/15.422\n'
printf 'prefill_pp2048_ub4096_hot_tps=662.7437\n'
printf 'prefill_pp2048_ub4096_cold_tps=91.345\n'
printf 'prefill_pp2048_ub512_hot_band_tps=486.9-533.0\n'
printf 'backend_policy=ROCm_HIP_PRIMARY_VULKAN_PARITY_ORACLE\n'
printf 'weights_published=false\n'
printf 'reason=model weights and the executing runtime trees are excluded from this repository\n'
