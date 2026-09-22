#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
receipt="$repo_root/repro/qwen27b/rocm-ladder-2026-09-22/sanitized_receipt.json"
printf 'lane=repro/qwen27b/rocm-ladder-2026-09-22\n'
printf 'status=EXCLUDED_WEIGHTS_DATA\n'
printf 'receipt=%s\n' "$receipt"
printf 'k0_raw_tps_quiet=17.35\n'
printf 'k1_accepted_tps_quiet=29.19\n'
printf 'k2_accepted_tps_quiet=34.25\n'
printf 'k4_accepted_tps_quiet=33.90\n'
printf 'historical_vulkan_k2_accepted_tps=42.11\n'
printf 'wide_m_effective_gbps_m1_to_m5=197,184,164,129\n'
printf 'iq4_nl_kv=NO_HIP_FA_KERNEL_SILENT_CPU_FALLBACK\n'
printf 'weights_published=false\n'
printf 'reason=model weights and the executing runtime tree are excluded from this repository\n'
