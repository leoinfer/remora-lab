#!/usr/bin/env bash
set -euo pipefail

# Rust-only public validation for the recovered R4X width evidence lane.
# Historical throughput is data in sanitized_receipt.json; it is not rerun by
# this script because the original foreign runtime is intentionally excluded.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

cargo run --quiet --release -p repro-harness -- r4x-d32a

echo "R4X_WIDTH_EVIDENCE=RECOVERED_RECEIPT_ONLY"
echo "R4X_WIDTH_RECEIPT=repro/r4x/width-sweep/sanitized_receipt.json"
echo "R4X_WIDTH_METRIC=kernel_prefill_rows_per_second_not_generation_tokens_per_second"
