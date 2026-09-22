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
| Flash-Next deployment record, 2026-09-18 | `EXCLUDED_WEIGHTS_DATA` | [`research/flash-next/CURRENT_RESEARCH_LOG.md`](research/flash-next/CURRENT_RESEARCH_LOG.md), [`PREFETCH_RESULT.md`](research/flash-next/PREFETCH_RESULT.md), [`research/moe-residency/`](research/moe-residency/) | [`repro/flash-next/deployment-2026-09-18/run.sh`](repro/flash-next/deployment-2026-09-18/run.sh) | [`sanitized_receipt.json`](repro/flash-next/deployment-2026-09-18/sanitized_receipt.json) | Coherence, canaries, sustained completions, context capacity, matched prefetch pair, route distribution, V2 hot region; model payload and runtime branch excluded |
| Flash-Next representation panel, 2026-09-18 | `EXCLUDED_WEIGHTS_DATA` | [`research/representation/README.md`](research/representation/README.md) | [`repro/flash-next/representation-panel/run.sh`](repro/flash-next/representation-panel/run.sh) | [`sanitized_receipt.json`](repro/flash-next/representation-panel/sanitized_receipt.json) | Tensor/activation fidelity on a 9-slice panel plus one off-budget island; not capability or quality evidence |
| Alice artifact, residency and host arena, 2026-09-22 | EXCLUDED_WEIGHTS_DATA | [`research/alice/README.md`](research/alice/README.md) | [`repro/alice/artifact-and-residency-2026-09-22/run.sh`](repro/alice/artifact-and-residency-2026-09-22/run.sh) | [`sanitized_receipt.json`](repro/alice/artifact-and-residency-2026-09-22/sanitized_receipt.json) | Identity, artifact composition, host/device expert split, arena frontier; weights and container excluded |
| Alice backend speed and prefill, 2026-09-22 | EXCLUDED_WEIGHTS_DATA | [`research/alice/README.md`](research/alice/README.md) | [`repro/alice/backend-speed-and-prefill-2026-09-22/run.sh`](repro/alice/backend-speed-and-prefill-2026-09-22/run.sh) | [`sanitized_receipt.json`](repro/alice/backend-speed-and-prefill-2026-09-22/sanitized_receipt.json) | Vulkan historical 18.430 with an 18.360 re-anchor, ROCm/HIP raw K0 15.374-15.422, prefill 662.7437 hot run |
| Alice MTP correctness and staging, 2026-09-22 | EXCLUDED_WEIGHTS_DATA | [`research/alice/README.md`](research/alice/README.md) | [`repro/alice/mtp-and-staging-2026-09-22/run.sh`](repro/alice/mtp-and-staging-2026-09-22/run.sh) | [`sanitized_receipt.json`](repro/alice/mtp-and-staging-2026-09-22/sanitized_receipt.json) | KAT 276/682 to 0/682, greedy parity at K=0/2/3/4, overlap probe vs production null |
| Alice workhorse packaging, 2026-09-22 | EXCLUDED_WEIGHTS_DATA | [`research/alice/README.md`](research/alice/README.md) | [`repro/alice/workhorse-packaging-2026-09-22/run.sh`](repro/alice/workhorse-packaging-2026-09-22/run.sh) | [`sanitized_receipt.json`](repro/alice/workhorse-packaging-2026-09-22/sanitized_receipt.json) | Local worker packaging on a base checkpoint with no chat template |
| Host-KV ROCm block reuse, 2026-09-22 | EXCLUDED_WEIGHTS_DATA | [`research/host-kv/README.md`](research/host-kv/README.md) | [`repro/host-kv/rocm-reuse-2026-09-22/run.sh`](repro/host-kv/rocm-reuse-2026-09-22/run.sh) | [`sanitized_receipt.json`](repro/host-kv/rocm-reuse-2026-09-22/sanitized_receipt.json) | Physical host read at the link roof, logical service to 244.944-246.89 GB/s, parity 4.5e-6, registration prerequisite |
| Host-KV production context, 2026-09-22 | EXCLUDED_WEIGHTS_DATA | [`research/host-kv/README.md`](research/host-kv/README.md) | [`repro/host-kv/production-context-2026-09-22/run.sh`](repro/host-kv/production-context-2026-09-22/run.sh) | [`sanitized_receipt.json`](repro/host-kv/production-context-2026-09-22/sanitized_receipt.json) | 262144 host-RAM KV vs 114688 VRAM KV, with pinning and async staging nulls |
| Qwen3.8-27B ROCm ladder, 2026-09-22 | EXCLUDED_WEIGHTS_DATA | [`research/qwen27b/README.md`](research/qwen27b/README.md) | [`repro/qwen27b/rocm-ladder-2026-09-22/run.sh`](repro/qwen27b/rocm-ladder-2026-09-22/run.sh) | [`sanitized_receipt.json`](repro/qwen27b/rocm-ladder-2026-09-22/sanitized_receipt.json) | Raw K0 17.35, accepted 29.19/34.25/33.90, iq4_nl KV defect, wide-M decay |
| Expert-major grouping, 2026-09-22 | EXCLUDED_WEIGHTS_DATA | [`research/alice/README.md`](research/alice/README.md) | [`repro/moe/expert-major-grouping-2026-09-22/run.sh`](repro/moe/expert-major-grouping-2026-09-22/run.sh) | [`sanitized_receipt.json`](repro/moe/expert-major-grouping-2026-09-22/sanitized_receipt.json) | 0.185 to 6.115 TMAC/s, bit-exact parity, Amdahl bound, retired CPU-MoE prefill |
| SSD action memory, 2026-09-22 | EXCLUDED_WEIGHTS_DATA | [`research/ssd-action-memory/README.md`](research/ssd-action-memory/README.md) | [`repro/ngram/action-memory-2026-09-22/run.sh`](repro/ngram/action-memory-2026-09-22/run.sh) | [`sanitized_receipt.json`](repro/ngram/action-memory-2026-09-22/sanitized_receipt.json) | Frozen-panel hit/miss A/B, 5.10x retraction, admissible 1.771x, packed-extent reduction |
| Alice-campaign negatives, 2026-09-22 | EXCLUDED_WEIGHTS_DATA | [`research/falsified/ALICE_CAMPAIGN_NEGATIVES.md`](research/falsified/ALICE_CAMPAIGN_NEGATIVES.md) | [`repro/falsified/alice-campaign-negatives-2026-09-22/run.sh`](repro/falsified/alice-campaign-negatives-2026-09-22/run.sh) | [`sanitized_receipt.json`](repro/falsified/alice-campaign-negatives-2026-09-22/sanitized_receipt.json) | Laya, 100k raw decode, structured sparsity, ALICE_MOE_BLOCK, F2, CPU-MoE, coarse double staging |
| Representation utilization, 2026-09-22 | EXCLUDED_WEIGHTS_DATA | [`research/representation/UTILIZATION.md`](research/representation/UTILIZATION.md) | [`repro/representation/utilization-2026-09-22/run.sh`](repro/representation/utilization-2026-09-22/run.sh) | [`sanitized_receipt.json`](repro/representation/utilization-2026-09-22/sanitized_receipt.json) | Unpack cost ordering measured; ALU estimates, pc4 and the Flash-Next target are modelled |

## Explicitly bounded historical or open artifacts

| Search term/family | Public disposition | What is available | What is not claimed |
| --- | --- | --- | --- |
| `33.8`, `20.6`, Qwen decode | `UNRECOVERABLE_HISTORICAL_RESULT` | [bounded disposition lane](repro/qwen27b/historical-baseline/) and historical caution | Exact public rerun without the original model/runtime/receipt |
| `K6V4`, R4KV quality frontier | `BLOCKED_PROVENANCE` | [blocked quality lane](repro/r4kv/quality-frontier/) plus Rust storage/profile KAT | Perplexity, attention-quality, or model-parity frontier |
| R4F / Flash-Next native lane | `BLOCKED_PROVENANCE` | [blocked native-lane disposition](repro/flash-next/full-model/) plus format notes and campaign status | Native full-model first-token or generation readiness. The model itself generates coherently through a separate external research runtime (see the 2026-09-18 deployment lane); that does not close this native gate. |
| Flash-Next deployment and representation, 2026-09-18 | `EXCLUDED_WEIGHTS_DATA` | Sanitized receipts for the [deployment record](repro/flash-next/deployment-2026-09-18/) and the [representation panel](repro/flash-next/representation-panel/) | Reproduction inputs (weights, container, executing runtime branch, captures, raw receipts) are excluded; no quality-retention, 384K, MTP, or resident-throughput result |
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
