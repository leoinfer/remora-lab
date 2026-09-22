#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
receipt="$repo_root/repro/ngram/action-memory-2026-09-22/sanitized_receipt.json"
printf 'lane=repro/ngram/action-memory-2026-09-22\n'
printf 'status=EXCLUDED_WEIGHTS_DATA\n'
printf 'receipt=%s\n' "$receipt"
printf 'admissible_clean_multiplier=1.771\n'
printf 'headline_5_10x=RETRACTED\n'
printf 'hit_ratio=5.0995\n'
printf 'miss_ratio=0.9981\n'
printf 'panel_solved=20/20\n'
printf 'packed_reduction=-0.721\n'
printf 'state_reuse=NOT_CLAIMABLE_NO_IDENTITY_GATE\n'
printf 'weights_published=false\n'
printf 'reason=model weights, the executing runtime, and the panel prompts are excluded from this repository\n'
