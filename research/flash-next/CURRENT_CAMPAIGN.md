# Current Flash-Next campaign

**Status:** active bring-up; 48-layer GPU assembly is pending.

This is a sanitized campaign summary. It contains methodology and bounded
results only. Model weights, the large R4F container, raw receipts, private
machine identifiers, private paths, and dirty worktree contents are not part
of this repository. The current campaign is not a full-model generation
result and it has no throughput claim.

## What is verified

| Seam | Boundary and result | Public disposition |
| --- | --- | --- |
| R4F codec/container bring-up | Exact accounting for 1,658 tensor records; native reopen/checksum and five finite codec families; graph binding covers attention, recurrent, PLE, QSA, routed-expert, router, embedding, final-norm, shared-expert, and sampler resource families. | Method and status are public; campaign-specific adapter/container promotion is pending sanitization and provenance review. |
| Embedding and layer-0 prefix | Real embedding lookup rows pass finite f32 parity; the layer-0 prefix reaches a Q5F QKV input projection. | Bounded seam evidence only. |
| GDN/recurrent path | Attention hyperconnection normalization/down/up/nonlinear mixing, Q5F QKV, initial convolution/SiLU, recurrent GDN, gated normalization, Q5F output projection, and attention reinjection have bounded GPU probes. | Bounded seam evidence only; full graph remains pending. |
| QSA indexer, selector, and output projection | Real indexer projection/state probes pass at a layer-3 prefix boundary; the main QSA q/k/v/o path includes the output projection at that prefix boundary; a 2,052-token / 513-block selector fixture has zero mask mismatches at top-k 512. | Fixture and prefix evidence; fused full-model selector is not claimed. |
| Selected-KV attention | The actual prefix state binds selector output to selected-KV attention and replays consistently. | Bounded GPU binding evidence; not first-token generation. |
| PLE addressed reads and page-cache | Row transport measures 144 actual addressed-read targets. An explicit bounded 4-KiB LRU executes a 16-token / 256-row stream with 260 unique pages and exact decoded-row hashing, including replay accounting. | Bounded page-cache evidence; no full PLE table is resident or claimed. |
| Routed MoE and Q4F expert execution | A selected Q4F expert capsule matches CPU references with maximum recorded errors of `4.77e-7`, `3.58e-7`, and `5.96e-8`, with one ULP at a BF16 boundary. A ten-expert CPU oracle reproduces routes, inputs, per-expert and final hashes. | Capsule/oracle evidence; no full-model GPU generation. |
| Q8F router, top-10, and routed accumulation | The latest guarded campaign scoreboard records route, top-k, ten expert outputs, and weighted accumulation passing primary and replay checks; representative maximum errors are `1.49e-8` for routing and `4.47e-8` for accumulation, without the earlier device fault. | Multi-expert seam evidence; 48-layer composition remains pending. |
| CPU replay/oracle | CPU 3/3, multi-prompt 9/9, and long-16 16/16 parity/replay records exist for the bounded text executor and route hashes. | Research evidence, not a production-generation claim. |

## Precision and failure boundaries

The frozen R4F-MVP precision policy failed isolated QSA-only and PLE-only
tests at early layers. A combined all-active BF16 fallback passed the bounded
CPU parity set. This is a precision-isolation result, not evidence that the
fallback is a finished model policy.

Earlier GPU cooperative-matrix mismatch/device-fault behavior is retained as
negative evidence; the campaign recovered for later bounded probes. A stale
storage-admission receipt is not treated as current execution evidence. The
selected-expert path includes a one-ULP BF16 boundary, so “matches” means the
stated tolerance, not bitwise identity. The false multi-POPS sparse result and
its repeated-accumulator overcount remain documented in
[`research/falsified/GFX1200_SPARSE_MATRIX_ANOMALY.md`](../falsified/GFX1200_SPARSE_MATRIX_ANOMALY.md).

## The first-token gate

The CPU reference currently reaches the expected first-token sequence
`[80692, 58649, 220]` in the private campaign environment. The GPU path has
not yet produced the first correct oracle token. The remaining gate is
completion through final normalization, `lm_head`, sampler, and oracle-token
comparison. Until that gate passes, this campaign makes no full-model
generation, quality, latency, or throughput claim.

The public Rust boundaries that provide reusable adjacent contracts are
[`har/crates/har-model-package`](../../har/crates/har-model-package/),
[`har/crates/har-model-compiler`](../../har/crates/har-model-compiler/),
[`har/crates/har-storage`](../../har/crates/har-storage/),
[`har/crates/r4kv`](../../har/crates/r4kv/), and
[`har/crates/har-vulkan`](../../har/crates/har-vulkan/). Current
Flash-Next-specific adapter modules and shaders remain outside the candidate
until their cleanliness, license, provenance, and public-fixture status are
cleared.
