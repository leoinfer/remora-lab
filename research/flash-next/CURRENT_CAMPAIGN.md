# Current Flash-Next campaign

**Status:** active split-lane research. The model generates coherently end to
end through the deployment lane; the native HAR/R4F lane's full-model gate
remains open.

The detailed public-safe log is
[`CURRENT_RESEARCH_LOG.md`](CURRENT_RESEARCH_LOG.md), and the storage-path
mechanism result is [`PREFETCH_RESULT.md`](PREFETCH_RESULT.md).

This is a sanitized campaign summary: methodology and bounded results only.
Model weights, large containers, raw receipts, private machine identifiers,
private paths, and dirty worktree contents are not part of this repository.

## The two lanes

| Lane | Status | What it is |
| --- | --- | --- |
| Deployment / measurement | generating coherently | llama.cpp-derived research branches plus measurement tooling. Produces every current result. An external reference implementation used as a research platform; not part of HAR, and HAR does not depend on it. |
| Native HAR / R4F | gate open | The native-Rust runtime and its experimental Flash-Next container direction. Bounded seam evidence exists; native full-model generation is not closed. |

## Current deployment record (2026-09-18, `MEASURED`)

- Coherent end-to-end generation on the reference machine.
- 12/12 deployment canaries produced correct outputs (evaluated against expected
  answers; the harness has no automated grader).
- Two 128-token completions in one sustained run, both reaching the token budget.
- 262,144-token context passes with the Q8 KV configuration. Capacity and
  coherence only; no long-context quality claim.
- Scaffold memory profile: VRAM ~4.42–4.47 GiB, GTT ~0.109–0.133 GiB, RSS
  ~11.5–12.6 GiB, `MemAvailable` floor 19.40 GiB, host-tier expert bank
  accounted by the loader at 35,129.29 MiB (≈34.3 GiB) page-backed.
- The deployed bank is a **temporary Q2 scaffold / control**. The manifest
  labels it `TEMPORARY CONTROL / FALLBACK`; it is not the quality authority and
  is to be replaced region by region with BF16-derived representations.

## Route-aware prefetch (`MEASURED`, matched pair)

Same weights, same build, same prompts, same flags; only the prefetch setting
differs. Decode moved from 1.28/0.81 t/s to 3.86/4.07 t/s, wall clock from
325.4 s to 127.0 s, major faults from 3,931,682 to 13,714, and device reads from
151,721,033,728 B to 74,699,780,096 B. This is a storage and residency path
result, not a quantization gain; see [`PREFETCH_RESULT.md`](PREFETCH_RESULT.md)
for the caveat list.

## Route distribution and V2 hot region

The trace set covers 1,542 decode steps across 6 workload classes over 48
layers, 512 experts, top-10 routing: 740,160 routed selections across 24,576
`(layer, expert)` units. The hottest 42% of units (10,321) carry 90.2% of routed
access mass.

The V2 experiment rebuilds that region directly from BF16 — `gate` and `up` as
`q4_K`, `down` as `q4_0` (the K-quant block geometry cannot cover 640-wide
`down` rows) — with ancestry verified by byte-identical re-encode from the BF16
authority. Built footprint: 28,535,500,800 B for the hot 42%.

The remaining 58% of the population and the non-expert core still run on the Q2
scaffold. **This is a migration configuration, not final ancestry.**

Streaming that hot bank through the slow tier is `REJECTED` as an architecture:
the streamed V2 arms measured 0.73–1.29 t/s decode at 0.229–1.667 GB/token of
device-normalized reads, against a modeled 1,262 MB/token for the 42% mask.
High-route-mass expert weights have to become persistent hot/warm cache hits
rather than per-token streams; that residency integration is designed and
modeled but not yet deployed.

## Native-lane seam record (2026-08 / 2026-09, `EXPERIMENTAL`)

This table describes the R4F/native work and is retained as the native lane's
own evidence record. None of these seams claim full-model generation.

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

## Native-lane precision and failure boundaries

The frozen R4F-MVP precision policy failed isolated QSA-only and PLE-only tests
at early layers. A combined all-active BF16 fallback passed the bounded CPU
parity set. This is a precision-isolation result, not evidence that the fallback
is a finished model policy.

Earlier GPU cooperative-matrix mismatch/device-fault behaviour is retained as
negative evidence; the campaign recovered for later bounded probes. A stale
storage-admission receipt is not treated as current execution evidence. The
selected-expert path includes a one-ULP BF16 boundary, so "matches" means the
stated tolerance, not bitwise identity. The false multi-POPS sparse result and
its repeated-accumulator overcount remain documented in
[`research/falsified/GFX1200_SPARSE_MATRIX_ANOMALY.md`](../falsified/GFX1200_SPARSE_MATRIX_ANOMALY.md).

## The native first-token gate

The CPU reference reaches the expected first-token sequence `[80692, 58649, 220]`
in the private campaign environment. The native GPU path has not yet produced
the first correct oracle token. The remaining gate is completion through final
normalization, `lm_head`, sampler, and oracle-token comparison.

This gate applies to the native lane only. The deployment lane already produces
coherent generation; the two must not be quoted as if one implied the other.

## Public Rust boundaries

The public Rust boundaries that provide reusable adjacent contracts are
[`har/crates/har-model-package`](../../har/crates/har-model-package/),
[`har/crates/har-model-compiler`](../../har/crates/har-model-compiler/),
[`har/crates/har-storage`](../../har/crates/har-storage/),
[`har/crates/r4kv`](../../har/crates/r4kv/), and
[`har/crates/har-vulkan`](../../har/crates/har-vulkan/). Current
Flash-Next-specific adapter modules and shaders remain outside the candidate
until their cleanliness, license, provenance, and public-fixture status are
cleared.
