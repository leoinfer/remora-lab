# Public reproduction lanes

Each lane contains its own README, manifest, expected outcome, runnable
command, and receipt or explicit limitation. Current lanes are:

- [R4X logical-prefill-row sweep](r4x/width-sweep/) — historical full-model
  receipt plus a Rust-only D32A known-answer validation path.
- [R4KV storage](r4kv/storage/) — Rust codec, profile, page, and fail-closed
  known-answer receipt; no model-quality claim.
- [R4KV quality frontier](r4kv/quality-frontier/) — explicit blocked
  provenance disposition; no model-quality number is asserted.
- [Effective-context accounting](context/effective-context/) — exact address
  recovery and shortcut-failure fixture; no dense 10M attention claim.
- [MTP accounting](mtp/accounting/) — deterministic acceptance bookkeeping;
  no neural throughput claim.
- [N-gram accounting](ngram/accounting/) — deterministic token replay; no
  neural decode throughput claim.
- [Qwen historical baseline](qwen27b/historical-baseline/) — explicit
  unrecoverable receipt disposition for the historical 20.6/33.8 labels.
- [Flash-Next full model](flash-next/full-model/) — explicit blocked
  first-token/generation disposition.
- [SWMMAC falsifier](swmmac/falsifier/) — independent-accumulator known-answer
  gate for the invalidated sparse-throughput claim.
- [Alice artifact and residency](alice/artifact-and-residency-2026-09-22/) —
  identity, artifact composition, expert split, and host arena values.
- [Alice backend speed and prefill](alice/backend-speed-and-prefill-2026-09-22/) —
  backend-labelled decode records plus the state-sensitive prefill hot run.
- [Alice MTP and staging](alice/mtp-and-staging-2026-09-22/) — MTP rollback
  correctness and the measured null in the overlap programme.
- [Alice workhorse packaging](alice/workhorse-packaging-2026-09-22/) —
  packaging and capability boundary of a base checkpoint.
- [Host-KV ROCm reuse](host-kv/rocm-reuse-2026-09-22/) — physical versus
  logical host-KV bandwidth, parity, and the registration prerequisite.
- [Host-KV production context](host-kv/production-context-2026-09-22/) —
  engine-level huge-context behaviour with its null controls.
- [Qwen3.8-27B ROCm ladder](qwen27b/rocm-ladder-2026-09-22/) — raw, accepted,
  and prefill rates as separate columns, plus the `iq4_nl` KV defect.
- [Expert-major grouping](moe/expert-major-grouping-2026-09-22/) — single-layer
  grouped-kernel throughput, parity, and the Amdahl bound.
- [SSD action memory](ngram/action-memory-2026-09-22/) — frozen-panel A/B, the
  retracted headline, and the admissible clean multiplier.
- [Alice-campaign negatives](falsified/alice-campaign-negatives-2026-09-22/) —
  measured rejections and retractions.
- [Representation utilization](representation/utilization-2026-09-22/) — the
  measured unpack ordering plus explicitly modelled projections.

Run [`setup/verify-environment.sh`](setup/verify-environment.sh) before lanes
that require a specific hardware phenotype. No lane may make Python, C++,
llama.cpp, GGML, CMake, or a foreign inference backend a HAR runtime
dependency.
