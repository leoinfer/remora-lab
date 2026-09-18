#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
receipt="$repo_root/repro/flash-next/representation-panel/sanitized_receipt.json"
printf 'lane=flash-next/representation-panel\n'
printf 'status=EXCLUDED_WEIGHTS_DATA\n'
printf 'receipt=%s\n' "$receipt"
printf 'weights_published=false\n'
printf 'raw_t1_weight_rel_rms=0.77599\n'
printf 'ls_t1_weight_rel_rms=0.60888\n'
printf 'two_bit_codebook_gain_over_donor=0.47656\n'
printf 'dense_t2_gain_over_donor=0.20783\n'
printf 'island_weight_rel_rms=0.00322\n'
printf 'teacher_kl=NOT_CAPTURED\n'
printf 'quality_retention=NOT_MEASURED\n'
printf 'reason=representation panels are bounded fidelity measurements; weights and captures are excluded from this repository\n'
