# Reproducibility status

This document is a live status page, not a claim that the entire research
corpus is closed. The machine-readable source of truth is
[`repro_manifest.json`](repro_manifest.json); important results without a
declared disposition are required to remain zero.

| Lane | Status | Public command | Evidence | Boundary |
| --- | --- | --- | --- | --- |
| R4X D32A geometry | `FULLY_REPRODUCIBLE` | `./repro/r4x/width-sweep/run_width_sweep.sh` | Rust known-answer output | Synthetic/model-free geometry only |
| R4X full-model logical-prefill-row sweep | `HISTORICAL_RECONSTRUCTION_AVAILABLE` | `./repro/r4x/width-sweep/run_width_sweep.sh` | [`sanitized_receipt.json`](repro/r4x/width-sweep/sanitized_receipt.json) | Exact historical throughput awaits Rust-only model executor |
| R4X `llama-bench -p 4096` logical prefill-row point | `NOT_RUN` | — | No authoritative receipt found | Predictions/preregistration are not measurements |
| R4X ubatch=4096 series | `MALFORMED` | — | Malformed receipt prefix retained | W2048 aborted during Vulkan submission |
| R4KV storage/page KAT | `FULLY_REPRODUCIBLE` | `./repro/r4kv/storage/run.sh` | Rust receipt | Codec/profile/page correctness only; no model-quality frontier |
| R4KV model-quality frontier | `BLOCKED_PROVENANCE` | `./repro/r4kv/quality-frontier/run.sh` | [bounded receipt](repro/r4kv/quality-frontier/sanitized_receipt.json) | No cleared model-quality receipt; storage KAT remains separate |
| Effective-context accounting | `FULLY_REPRODUCIBLE` | `./repro/context/effective-context/run.sh` | Rust receipt | Addressable representation and shortcut probe; not dense 10M attention |
| MTP acceptance accounting | `FULLY_REPRODUCIBLE` | `./repro/mtp/accounting/run.sh` | Rust receipt | Synthetic acceptance bookkeeping; not neural MTP throughput |
| N-gram replay accounting | `FULLY_REPRODUCIBLE` | `./repro/ngram/accounting/run.sh` | Rust receipt | Synthetic replay; throughput intentionally not measured |
| Qwen historical decode baseline | `UNRECOVERABLE_HISTORICAL_RESULT` | `./repro/qwen27b/historical-baseline/run.sh` | [bounded receipt](repro/qwen27b/historical-baseline/sanitized_receipt.json) | Historical 20.6/33.8 labels are not asserted as public throughput |
| Flash-Next native full-model generation (HAR/R4F) | `BLOCKED_PROVENANCE` | `./repro/flash-next/full-model/run.sh` | [bounded receipt](repro/flash-next/full-model/sanitized_receipt.json) | Native first-token and generation gates remain incomplete; the deployment lane generates on a separate external research runtime |
| Flash-Next deployment record, 2026-09-18 | `EXCLUDED_WEIGHTS_DATA` | `./repro/flash-next/deployment-2026-09-18/run.sh` | [sanitized receipt](repro/flash-next/deployment-2026-09-18/sanitized_receipt.json) | Coherence, canaries, sustained completions, context pass, matched prefetch pair, route distribution, V2 hot region; model payload and runtime branch excluded |
| Flash-Next representation panel, 2026-09-18 | `EXCLUDED_WEIGHTS_DATA` | `./repro/flash-next/representation-panel/run.sh` | [sanitized receipt](repro/flash-next/representation-panel/sanitized_receipt.json) | Tensor- and activation-level fidelity only; no capability or quality result |
| SWMMAC falsifier | `FALSIFIED_REPRODUCIBLE` | `./repro/swmmac/falsifier/run.sh` | Rust receipt | Accumulator known-answer falsifier; no TOPS claim |
| Alice artifact and residency, 2026-09-22 | EXCLUDED_WEIGHTS_DATA | ./repro/alice/artifact-and-residency-2026-09-22/run.sh | [sanitized receipt](repro/alice/artifact-and-residency-2026-09-22/sanitized_receipt.json) | Identity, composition, expert split and host arena values; no rerunnable benchmark and no quality result |
| Alice backend speed and prefill, 2026-09-22 | EXCLUDED_WEIGHTS_DATA | ./repro/alice/backend-speed-and-prefill-2026-09-22/run.sh | [sanitized receipt](repro/alice/backend-speed-and-prefill-2026-09-22/sanitized_receipt.json) | Backend-labelled decode plus a state-sensitive prefill hot run; raw, accepted and prefill rates never blended |
| Alice MTP and staging, 2026-09-22 | EXCLUDED_WEIGHTS_DATA | ./repro/alice/mtp-and-staging-2026-09-22/run.sh | [sanitized receipt](repro/alice/mtp-and-staging-2026-09-22/sanitized_receipt.json) | MTP rollback correctness parity plus a measured null in the overlap programme; no multiplier, no speedup |
| Alice workhorse packaging, 2026-09-22 | EXCLUDED_WEIGHTS_DATA | ./repro/alice/workhorse-packaging-2026-09-22/run.sh | [sanitized receipt](repro/alice/workhorse-packaging-2026-09-22/sanitized_receipt.json) | Packaging and capability boundary only; the checkpoint is a base model |
| Expert-major grouping, 2026-09-22 | EXCLUDED_WEIGHTS_DATA | ./repro/moe/expert-major-grouping-2026-09-22/run.sh | [sanitized receipt](repro/moe/expert-major-grouping-2026-09-22/sanitized_receipt.json) | Single-layer kernel throughput and parity; not a full-model multiplier |
| Host-KV ROCm reuse, 2026-09-22 | EXCLUDED_WEIGHTS_DATA | ./repro/host-kv/rocm-reuse-2026-09-22/run.sh | [sanitized receipt](repro/host-kv/rocm-reuse-2026-09-22/sanitized_receipt.json) | Kernel-level physical and logical bandwidth plus parity; logical is not physical, and no end-to-end t/s |
| Host-KV production context, 2026-09-22 | EXCLUDED_WEIGHTS_DATA | ./repro/host-kv/production-context-2026-09-22/run.sh | [sanitized receipt](repro/host-kv/production-context-2026-09-22/sanitized_receipt.json) | Engine-level decode and prefill at two context shapes; no long-context quality claim |
| Qwen3.8-27B ROCm ladder, 2026-09-22 | EXCLUDED_WEIGHTS_DATA | ./repro/qwen27b/rocm-ladder-2026-09-22/run.sh | [sanitized receipt](repro/qwen27b/rocm-ladder-2026-09-22/sanitized_receipt.json) | Raw, accepted and prefill rates as separate columns; the historical 42.11 belongs to another artifact and backend |
| SSD action memory, 2026-09-22 | EXCLUDED_WEIGHTS_DATA | ./repro/ngram/action-memory-2026-09-22/run.sh | [sanitized receipt](repro/ngram/action-memory-2026-09-22/sanitized_receipt.json) | Task-level wall-clock A/B on a frozen panel; the 5.10x headline is retracted and the token/forward figures are a proxy |
| Alice-campaign negatives, 2026-09-22 | EXCLUDED_WEIGHTS_DATA | ./repro/falsified/alice-campaign-negatives-2026-09-22/run.sh | [sanitized receipt](repro/falsified/alice-campaign-negatives-2026-09-22/sanitized_receipt.json) | Measured rejections and retractions; instrument readings, not tolerance bands |
| Representation utilization, 2026-09-22 | EXCLUDED_WEIGHTS_DATA | ./repro/representation/utilization-2026-09-22/run.sh | [sanitized receipt](repro/representation/utilization-2026-09-22/sanitized_receipt.json) | One measured unpack ordering plus explicitly modelled projections; no measured compression result |

The broader Qwen and Flash-Next full-model lanes remain historical, blocked, or
active as described in the technical artifact index. A prose summary is not
promoted to a reproducibility result without an executable disposition and a
receipt.
