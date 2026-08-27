# Public reproduction lanes

Each lane contains its own README, manifest, expected outcome, runnable
command, and receipt or explicit limitation. Current lanes are:

- [R4X width sweep](r4x/width-sweep/) — historical full-model receipt plus a
  Rust-only D32A known-answer validation path.
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

Run [`setup/verify-environment.sh`](setup/verify-environment.sh) before lanes
that require a specific hardware phenotype. No lane may make Python, C++,
llama.cpp, GGML, CMake, or a foreign inference backend a HAR runtime
dependency.
