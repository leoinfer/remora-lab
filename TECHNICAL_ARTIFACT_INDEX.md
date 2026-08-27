# Technical artifact index

This index is deliberately separate from `RESEARCH_IDEA_INDEX.md` and the
publication coverage matrix. It points to actual public implementations,
experiment lanes, receipts, reproduction commands, and explicit evidence
boundaries. Search terms such as `W512`, `33.8`, `K6V4`, and `1166592` are
included here so a reviewer can reach the relevant disposition without relying
on prose discovery.

The machine-readable source is [`technical_artifact_index.json`](technical_artifact_index.json).

## Direct evidence lanes

| Artifact | Status | Implementation | Reproduction | Receipt | Metric boundary |
| --- | --- | --- | --- | --- | --- |
| HAR Rust runtime and release gates | `FULLY_REPRODUCIBLE` | [`har/`](har/) | [`DEVELOPMENT.md`](DEVELOPMENT.md) | [`PUBLIC_HAR_RELEASE_AUDIT.json`](PUBLIC_HAR_RELEASE_AUDIT.json) | Bounded Rust tests, policy, linked-object, exec, and Vulkan smoke evidence |
| R4X D32A clean-room KAT | `FULLY_REPRODUCIBLE` | [`formats/r4x/FORMAT.md`](formats/r4x/FORMAT.md), [`tools/repro-harness/src/main.rs`](tools/repro-harness/src/main.rs) | [`repro/r4x/width-sweep/run_width_sweep.sh`](repro/r4x/width-sweep/run_width_sweep.sh) | [`sanitized_receipt.json`](repro/r4x/width-sweep/sanitized_receipt.json) | Geometry/vector KAT, not full-model parity |
| R4X logical-prefill-row sweep | `HISTORICAL_RECONSTRUCTION_AVAILABLE` | [`historical_command.sh`](research/archival/r4x/width-sweep/historical_command.sh) | [`repro/r4x/width-sweep/`](repro/r4x/width-sweep/) | [`sanitized_receipt.json`](repro/r4x/width-sweep/sanitized_receipt.json) | Logical prefill diagnostic rows/s, not generation tokens/s |
| R4KV storage/page KAT | `FULLY_REPRODUCIBLE` | [`har/crates/r4kv/`](har/crates/r4kv/) | [`repro/r4kv/storage/run.sh`](repro/r4kv/storage/run.sh) | [`sanitized_receipt.json`](repro/r4kv/storage/sanitized_receipt.json) | Codec/page/accounting, not model quality |
| Effective-context accounting | `FULLY_REPRODUCIBLE` | [`har/crates/har-contextfold/`](har/crates/har-contextfold/) | [`repro/context/effective-context/run.sh`](repro/context/effective-context/run.sh) | [`sanitized_receipt.json`](repro/context/effective-context/sanitized_receipt.json) | Addressable representation and shortcut probe, not dense 10M attention |
| MTP acceptance accounting | `FULLY_REPRODUCIBLE` | [`har/crates/har-execution/src/speculation.rs`](har/crates/har-execution/src/speculation.rs) | [`repro/mtp/accounting/run.sh`](repro/mtp/accounting/run.sh) | [`sanitized_receipt.json`](repro/mtp/accounting/sanitized_receipt.json) | Synthetic acceptance bookkeeping, not neural throughput |
| N-gram replay accounting | `FULLY_REPRODUCIBLE` | [`tools/repro-harness/src/main.rs`](tools/repro-harness/src/main.rs) | [`repro/ngram/accounting/run.sh`](repro/ngram/accounting/run.sh) | [`sanitized_receipt.json`](repro/ngram/accounting/sanitized_receipt.json) | Synthetic replay; throughput intentionally null |
| SWMMAC falsifier | `FALSIFIED_REPRODUCIBLE` | [`research/falsified/GFX1200_SPARSE_MATRIX_ANOMALY.md`](research/falsified/GFX1200_SPARSE_MATRIX_ANOMALY.md) | [`repro/swmmac/falsifier/run.sh`](repro/swmmac/falsifier/run.sh) | [`sanitized_receipt.json`](repro/swmmac/falsifier/sanitized_receipt.json) | Independent committed-work KAT, not TOPS |
| REMORA metabolism Rust control | `FULLY_REPRODUCIBLE` | [`har-metabolism`](har/crates/har-metabolism/) and runtime bridges | [`invariants.rs`](har/crates/har-metabolism/tests/invariants.rs) | none | Bounded deterministic accounting/invariants, not full-model throughput or energy |

## Explicitly bounded historical or open artifacts

| Search term/family | Public disposition | What is available | What is not claimed |
| --- | --- | --- | --- |
| `33.8`, `20.6`, Qwen decode | `UNRECOVERABLE_HISTORICAL_RESULT` | [bounded disposition lane](repro/qwen27b/historical-baseline/) and historical caution | Exact public rerun without the original model/runtime/receipt |
| `K6V4`, R4KV quality frontier | `BLOCKED_PROVENANCE` | [blocked quality lane](repro/r4kv/quality-frontier/) plus Rust storage/profile KAT | Perplexity, attention-quality, or model-parity frontier |
| R4F / Flash-Next | `BLOCKED_PROVENANCE` | [blocked full-model lane](repro/flash-next/full-model/) plus format notes and current campaign status | Full-model first-token or generation readiness |
| R4X-H/R4X-S/XP-S | `PROPOSED_NO_RESULT_YET` | Family specifications and research notes | Interoperability or measured performance |
| MoE, ExpertPack, residency | `PROPOSED_NO_RESULT_YET` | Public Rust bounded structures and design records | Full-model residency/quality frontier |
| HERMES, REMORA, RSSO, PHASE | `PROPOSED_NO_RESULT_YET` | Idea atlas, formal notes, and open-problem records | An idea record being mistaken for an implementation result |

Every claim in [`claims.json`](claims.json) has an entry in the JSON
`claim_coverage` array. The Rust `repro-audit` checks this index, the claim
mapping, and the lane manifests. An explicit blocked or unrecoverable status is
evidence bookkeeping; it is not a substitute for a missing result.

The three bounded lanes above are executable disposition checks. They exit
successfully so a clean release gate can verify that the gap is intentional;
their receipts contain null performance/quality fields and must not be read as
model results.
