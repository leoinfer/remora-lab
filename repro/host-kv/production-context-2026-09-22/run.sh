#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
receipt="$repo_root/repro/host-kv/production-context-2026-09-22/sanitized_receipt.json"
printf 'lane=repro/host-kv/production-context-2026-09-22\n'
printf 'status=EXCLUDED_WEIGHTS_DATA\n'
printf 'receipt=%s\n' "$receipt"
printf 'ram_kv_ctx262144_decode_tps=8.4-8.7\n'
printf 'ram_kv_ctx262144_prefill_tps=306-335\n'
printf 'vram_kv_ctx114688_decode_tps=18.53\n'
printf 'pinned_host_kv_delta=+0%\n'
printf 'async_staging_delta=0%\n'
printf 'vulkan_control_decode_tps=5.38\n'
printf 'reuse_kernel_integrated=false\n'
printf 'weights_published=false\n'
printf 'reason=model weights and the executing runtime tree are excluded from this repository\n'
