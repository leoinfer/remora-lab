# Reproducibility status

This document is a live status page, not a claim that the entire research
corpus is closed. The machine-readable source of truth is
[`repro_manifest.json`](repro_manifest.json); important results without a
declared disposition are required to remain zero.

| Lane | Status | Public command | Evidence | Boundary |
| --- | --- | --- | --- | --- |
| R4X D32A geometry | `FULLY_REPRODUCIBLE` | `./repro/r4x/width-sweep/run_width_sweep.sh` | Rust known-answer output | Synthetic/model-free geometry only |
| R4X full-model width sweep | `HISTORICAL_RECONSTRUCTION_AVAILABLE` | `./repro/r4x/width-sweep/run_width_sweep.sh` | [`sanitized_receipt.json`](repro/r4x/width-sweep/sanitized_receipt.json) | Exact historical throughput awaits Rust-only model executor |
| R4X W4096 width | `NOT_RUN` | — | No authoritative receipt found | Predictions/preregistration are not measurements |
| R4X ubatch=4096 series | `MALFORMED` | — | Malformed receipt prefix retained | W2048 aborted during Vulkan submission |
| R4KV storage/page KAT | `FULLY_REPRODUCIBLE` | `./repro/r4kv/storage/run.sh` | Rust receipt | Codec/profile/page correctness only; no model-quality frontier |
| R4KV model-quality frontier | `BLOCKED_PROVENANCE` | `./repro/r4kv/quality-frontier/run.sh` | [bounded receipt](repro/r4kv/quality-frontier/sanitized_receipt.json) | No cleared model-quality receipt; storage KAT remains separate |
| Effective-context accounting | `FULLY_REPRODUCIBLE` | `./repro/context/effective-context/run.sh` | Rust receipt | Addressable representation and shortcut probe; not dense 10M attention |
| MTP acceptance accounting | `FULLY_REPRODUCIBLE` | `./repro/mtp/accounting/run.sh` | Rust receipt | Synthetic acceptance bookkeeping; not neural MTP throughput |
| N-gram replay accounting | `FULLY_REPRODUCIBLE` | `./repro/ngram/accounting/run.sh` | Rust receipt | Synthetic replay; throughput intentionally not measured |
| Qwen historical decode baseline | `UNRECOVERABLE_HISTORICAL_RESULT` | `./repro/qwen27b/historical-baseline/run.sh` | [bounded receipt](repro/qwen27b/historical-baseline/sanitized_receipt.json) | Historical 20.6/33.8 labels are not asserted as public throughput |
| Flash-Next full-model generation | `BLOCKED_PROVENANCE` | `./repro/flash-next/full-model/run.sh` | [bounded receipt](repro/flash-next/full-model/sanitized_receipt.json) | First-token and generation gates remain incomplete |
| SWMMAC falsifier | `FALSIFIED_REPRODUCIBLE` | `./repro/swmmac/falsifier/run.sh` | Rust receipt | Accumulator known-answer falsifier; no TOPS claim |

The broader Qwen and Flash-Next full-model lanes remain historical, blocked, or
active as described in the technical artifact index. A prose summary is not
promoted to a reproducibility result without an executable disposition and a
receipt.
