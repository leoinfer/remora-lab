# Qwen3.8 Flash research log

**Updated:** 2026-09-18 evidence freeze, extended 2026-09-22 with the Alice
campaign, the host-KV result, and the ROCm pivot.
**Scope:** public-safe research summary; no model payloads, raw receipts, private
paths, host identifiers, or unpublished identities are reproduced here.

This log keeps the current Flash-Next work visible without turning a bounded
experiment into a product or performance claim. The public status vocabulary is
explained in [`docs/methodology.md`](../../docs/methodology.md) and the claims
ledger remains authoritative in [`CLAIMS.md`](../../CLAIMS.md).

Dated records are append-only. The 2026-09-13 records below are preserved as
they were written; 2026-09-18 adds new measurements rather than rewriting them.

## How to read this log

| Label | Meaning in this log |
| --- | --- |
| `MEASURED` | A direct run or physical scan with a recorded scope. |
| `DERIVED` | Arithmetic from measured metadata or a cited public specification. |
| `MODELED` | Replay, cache, or traffic model; not a live device result. |
| `EXPERIMENTAL` | A bounded implementation or fixture exists, but coverage is incomplete. |
| `HYPOTHESIS` | A target or mechanism proposed for falsification, not an observed result. |
| `UNMEASURED` | The required run has not produced an authority. |
| `BLOCKED` | A required artifact, comparator, or runtime gate is unavailable. |
| `HISTORICAL` | Useful prior context whose original receipt is not in this tree. |
| `REJECTED` / `FALSIFIED` | A proposed interpretation failed its own acceptance rule. |

`VERIFIED`, `EXPERIMENTAL`, `HISTORICAL`, and `INVALIDATED` remain the claims
ledger statuses. A value can be measured in a narrow experiment and still be
unverified as a broader product claim.

## Runtime lanes: what executes the current experiments

Two lanes exist and must not be conflated, in either direction.

**Deployment and measurement lane (`EXPERIMENTAL`).** Most current model
experiments execute on llama.cpp-derived research branches and the surrounding
measurement tooling. That runtime produced every 2026-09-18 number in this log.
It is an external reference implementation used as a research platform. It is
not part of HAR, HAR does not depend on it, and its research branches are not
published in this repository.

**Native HAR / R4F lane (`EXPERIMENTAL`, gate not closed).** HAR is the
native-Rust runtime and control-plane project developed inside the same research
program, with its own scope: Rust host/runtime code, a Rust Vulkan resource and
dispatch layer, model/package and storage contracts, scheduling, residency
accounting, and correctness gates. Its native full-model Flash-Next generation
gate is not closed; the bounded disposition is
[`repro/flash-next/full-model/`](../../repro/flash-next/full-model/). HAR is
documented in [`har/README.md`](../../har/README.md) and
[`research/systems/HAR.md`](../systems/HAR.md).

Before 2026-09-18 this log described HAR as the runtime of record for the
Flash-Next campaign. That framing is corrected here rather than deleted: the
native lane is real work with real gates, but it is not the lane that currently
produces model results.

## Deployment status (`MEASURED`, 2026-09-18)

Qwen3.8 Flash-Next generates coherent text end-to-end on the reference RX 9060
XT machine through the deployment lane.

| Result | Boundary |
| --- | --- |
| Coherent end-to-end generation | Short-context deployment profile; coherent output, not a quality-parity result. |
| 12/12 deployment canaries produced correct outputs | The verdict is evaluation of printed outputs; the harness has no automated answer grader. `returncode: 0` is the machine-checked part. |
| Two 128-token completions in one sustained run | Both generation streams ran to their 128-token budget with no early stop. |
| 262,144-token context with Q8 KV passed | Capacity and coherence smoke only. No long-context quality claim. |

Memory profile of the deployment scaffold runs (peaks sampled every 10 s from
device and kernel counters):

| Run | Peak RSS | Peak VRAM | Peak GTT | `MemAvailable` floor |
| --- | --- | --- | --- | --- |
| 12 canaries | 11.55 GiB | 4.47 GiB | 0.133 GiB | 19.64 GiB |
| Sustained 2×128 tokens | 11.73 GiB | 4.42 GiB | 0.109 GiB | 20.59 GiB |
| 12 canaries, final | 12.56 GiB | 4.42 GiB | 0.129 GiB | 19.40 GiB |
| 262,144-context Q8 profile | 5.20 GiB | 10.50 GiB | 0.504 GiB | 20.37 GiB |

The 262,144-context profile is a different memory shape: context state dominates
VRAM and the resident weight footprint is small.

The host-tier expert bank is a file-mapped, page-backed region. The loader
accounts it as `CPU_Mapped model buffer size = 35129.29 MiB` (approximately
34.3 GiB); RSS counts only the resident pages of that mapping, which is why the
run RSS figures are far below the mapped size. A separate arithmetic figure of
32,400 MiB for the same bank exists in an earlier safety-stop record and has not
been reconciled with the loader's accounting; the published number here is the
loader's own line.

Two caveats travel with this deployment:

- swap was in use throughout the window (about 13 GiB), so "no OOM" is true and
  "no memory pressure" would be false;
- a full all-layer Q2 offload attempt on this host failed as a **residency**
  failure (VRAM and GTT both saturated, `MemAvailable` near 2 GiB), not as
  evidence about Q2 quality or Vulkan correctness.

## Route-aware prefetch (`MEASURED`, single matched pair)

Same weights, same build, same prompts, same flags; the only difference is the
route-aware prefetch setting. Full caveat list and mechanism description:
[`PREFETCH_RESULT.md`](PREFETCH_RESULT.md).

| Metric | Control | Route-aware prefetch |
| --- | --- | --- |
| Decode, long-explain | 1.28 t/s | 3.86 t/s |
| Decode, long-code | 0.81 t/s | 4.07 t/s |
| p50 token latency | 702 / 1189 ms | 224 / 233 ms |
| p95 token latency | 1634 / 2089 ms | 475 / 362 ms |
| Wall clock | 325.4 s | 127.0 s |
| NVMe bytes read | 151,721,033,728 B | 74,699,780,096 B |
| Major page faults | 3,931,682 | 13,714 |

This is a storage and residency path result on identical weights, not a
quantization gain. The two arms ran back-to-back, device counters are
machine-wide, and prefetch emission counters were never logged; the ratios are
an observed pair on the reference machine.

Corroborating but **not** part of the matched pair: a control-plus-prefetch run
on a harder 12-prompt panel, on a later binary build, measured 3.49–4.30 t/s
decode. It is recorded as directional support only.

Related storage-path measurements, which indicate headroom rather than an
automatic multiplier:

- native cold `pread` at the 460,800-byte slice size the scaffold serves:
  0.2103 GB/s at one reader, 1.1883 GB/s at eight readers;
- the runtime's effective expert-read traffic in the deployed scaffold is
  quoted as roughly 0.11–0.36 GB/s. That band is a restated estimate: the
  primary run artifact behind it was not recovered;
- the Python `mmap_fault` benchmark rows are GIL-bound floors and are not a
  like-for-like measurement of the runtime's page-fault path.

## Route distribution and the V2 experiment

### Route trace (`MEASURED`, `DERIVED`)

The trace set covers 6 workload classes × 257 decode steps = 1,542 steps, over
48 layers and 512 experts with top-10 routing: 740,160 routed selections across
24,576 `(layer, expert)` units, or 73,728 role slices.

| Quantity | Value |
| --- | --- |
| Hottest 42% of `(layer, expert)` units | 10,321 of 24,576 |
| Share of routed access mass they carry | 90.19590358841332% |
| Units needed for exactly 90% of mass | 10,241 (41.67% of the population) |
| Per-layer top-64 share | 36.8% – 70.0% (median 53.6%) |

The distribution is skewed but not extreme: roughly 42% of the population
carries about 90% of the traffic.

### V2 hot region (`EXPERIMENTAL`)

The V2 experiment rebuilds the high-value region **directly from BF16**:

```text
gate -> q4_K
up   -> q4_K
down -> q4_0
```

`down` takes `q4_0` rather than a K-quant because ggml K-quant blocks are 256
values wide and the `down` rows are 640 wide, so the K-quant block geometry does
not apply. All roles land at the same 4.5 bits per weight and the same 921,600
bytes per role slice, or 2,764,800 bytes per unit.

Ancestry is verified rather than asserted: sampled bank slices were re-encoded
from the BF16 authority with the same codec and compared byte-wise, and all
sampled slices matched. Measured fidelity of the served hot bank is weight
relRMS 0.0724 (gate), 0.0723 (up), 0.0868 (down) as means over all 48 layers,
against 0.44–0.72 for the scaffold and 0.775 for raw T1.

Built bank footprint for the 42% region: 28,535,500,800 B (28.54 GB) allocated,
out of a sparse full-geometry file of 67,947,724,800 B apparent size.

Remaining regions are **temporary**: the cold 58% of the population and the
entire non-expert core still run on the Q2 scaffold, and the scaffold is
labelled `TEMPORARY CONTROL / FALLBACK` in the deployment manifest. It is
explicitly not the quality authority and is to be replaced region by region once
a BF16-derived codec at that rate is admitted. The remaining donor-derived
region is therefore not final ancestry.

### Why streaming the hot bank is not the answer (`MEASURED`, `REJECTED` as architecture)

| Arm | Decode | Device-normalized reads |
| --- | --- | --- |
| Scaffold, no prefetch | 1.28 / 0.81 t/s | 465,401,944 B/token |
| Scaffold + prefetch | 3.86 / 4.07 t/s | 229,140,429 B/token |
| V2 hot bank + prefetch | 1.29 / 1.05 t/s | 701,223,540 B/token |
| V2 hot bank, 12-prompt panel | 0.937 t/s mean | 1,667,317,258 B/token |

Moving a larger, higher-fidelity region through the slow tier made decode
*slower*, not faster: the streamed V2 arms measured 0.73–1.29 t/s with
0.229–1.667 GB/token of device-normalized reads, against a modeled
1,262 MB/token of expert traffic for the 42% mask.

The conclusion is architectural, not representational: at these byte rates the
per-token device traffic is the wall. A related residency analysis states the
same thing as a cold-streaming ceiling of about 2.29 committed tokens/s and
notes that the only lever above 2× is raising the resident fraction of the
bank. A projected 7–9 committed tokens/s for a 10–16 GiB resident arena is
`MODELED`; the 80 and 250 tokens/s figures remain research targets.

Note on attribution: earlier internal notes attributed "1.34 GB/token" to the
streamed V2 configuration. That figure belongs to a different, earlier
base-closure decode cell, not to V2. The V2-specific values are the ones above.

### Intended execution architecture (`HYPOTHESIS` / `MODELED`)

```text
hot:  ready route-aligned expert slots in device-local memory
warm: bounded host-memory triplet cache with identity and integrity checks
cold: canonical expert payload in storage, fetched only through admission
```

plus an optional one-shot bypass staging path for a cold expert that does not
deserve VRAM promotion. The design goal is that normal-token execution becomes
mostly cache hits, so that storage is not the normal expert-compute path.

This is a target, not a deployed state. Route-aware prefetch is demonstrated to
help; persistent hot/warm caching is designed and modeled but its runtime
integration is not complete. Full-model resident throughput remains
`UNMEASURED`.

## Representation research (`MEASURED`, bounded)

The measured panels that changed the representation direction are recorded with
their metric definitions in [`research/representation/`](../representation/).
Summary:

| Representation | Bits/weight | Weight relRMS | Activation error |
| --- | --- | --- | --- |
| Raw ternary T1 | ~1.75 | ~0.776 | ~0.778 |
| LS-reencoded T1, same bytes | ~1.75 | ~0.609 | ~0.611 |
| Q2_0 donor scaffold | ~2.25 | ~0.459–0.488 | ~0.444 |
| T1 + dense ternary T2 | ~3.50 | ~0.438 on the measured slice | — |

Equal-added-byte correction panel (activation error removed, 358,400 B added
per role): dense ternary T2 ~20.78%, uniform 2-bit / 4-level residual codebook
~47.66%, rotated 2-bit codebook ~47.78% over the donor base; and over the T1
base, LS-scale re-encode ~62.9% versus dense ternary T2 ~46.0%.

A bounded Q8 residual-island test on layer 0 / expert 283 moved weight relRMS
from ~0.779 to ~0.0032 using 5,222,400 B of residual payload. That proves the
island mechanism works and that it is byte-inefficient: the island costs about
4.86× the equal-byte budget, so it is mechanism evidence, not a proposal.

None of this is capability evidence. There is no teacher-KL or end-to-end
quality result for these panels, and full BF16 capability retention is
`UNMEASURED`.

## Evidence corrections made during this publication pass

Publishing this material surfaced defects in earlier internal notes. They are
recorded here because silently fixing them would hide exactly the kind of error
this repository is supposed to catch.

| Earlier statement | Correction |
| --- | --- |
| Host-tier expert bank "35.13 GiB" | Units error: the loader line reads 35,129.29 **MiB**, which is ~34.3 GiB. Quote the loader line. |
| Donor Q2_0 weight relRMS "0.44" | 0.44 is the donor's *activation* error. Its weight relRMS is ~0.459 (27 slices) / ~0.488 (panel). |
| T1 per-role payload "1,433,600 B" in one receipt | `float32` byte-count unit error; the correct payload is 358,400 B. |
| "1.34 GB/token" attributed to the V2 streamed configuration | Belongs to an earlier base-closure decode cell; V2-specific values are 1,262 MB/token modeled and 0.229–1.667 GB/token measured. |
| V2 canaries "answers identical to control" | 12/12 correct, but only 9/12 byte-identical; three outputs differ in wording and remain correct. |
| 262,144-context memory line in "GiB" | Values were decimal GB under binary labels (for example 11.28 vs 10.50 GiB). |
| Expert-bank size | Two unreconciled figures exist (loader 35,129.29 MiB; arithmetic 32,400 MiB). This log quotes the loader line and flags the delta. |
| "739,200 expert selections" | Arithmetic slip; the trace contains 740,160 selections. |

## Model metadata and the arithmetic roof

The official [Qwen3.8-Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
and its configuration describe a 125B-parameter model with approximately 6B
activated parameters, a 512-expert MoE with 10 routed experts plus a shared
expert, 48 trunk layers, a 20M-entry n-gram vocabulary, and one native MTP
layer. These are model facts, not local throughput measurements.

A deliberately coarse arithmetic normalization is:

```text
6e9 active parameters/token × 2 operations/MAC ≈ 12e9 operations/token
                                      = 12 GOP/token
```

Using AMD's published [RX 9060 XT product figures](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9060xt.html),
that normalization gives:

```text
410e12 INT4 dense operations/s ÷ 12e9 operations/token ≈ 34,167 tokens/s
821e12 structured-sparse operations/s ÷ 12e9 operations/token ≈ 68,417 tokens/s
```

Both results are `DERIVED` arithmetic ceilings, not generation rates. The
second line additionally requires compatible structured-sparse weights,
metadata, dimensions, and a selected sparse instruction path; ordinary
quantized weights do not receive that multiplier. The
[dense GPU roofline provenance](../../docs/dense-roofline.md) keeps physical
bandwidth, compute-equivalent bandwidth, and useful token rate separate; the
older approximately 154 TB/s figure is a six-bit compute-equivalent
normalization, **not** physical VRAM bandwidth.

The **250 tokens/s** figure, and the 80 tokens/s figure alongside it, are
research targets: not benchmarks, not forecasts, not promises. Under the same
shorthand 250 tokens/s represents about 3 TOPS of useful arithmetic, roughly
0.73% of the published 410-TOPS dense headline. It is high enough to force an
end-to-end answer for data movement, expert residency, state transactions,
scheduling, and kernel efficiency while remaining far below the idealized
arithmetic roof.

MTP does not create bandwidth. Speculative future positions may expose more
reuse — more rows to group per expert, advance route information, better
prefetch lead time — and grouping may move execution closer to the matrix roof,
but none of that is demonstrated yet.

## Historical records (2026-09-13 and earlier)

These are preserved as written at the time. They refer to the MIX34/R4X line and
to the native lane, not to the Remora V2 hot-bank line above.

### MIX34 geometry authority (`MEASURED` + `DERIVED`, 2026-09-12)

- 49 blocks; 147 expert tensors; 123,312,537,600 expert values;
- 63,968,378,880 expert payload bytes; 59.5751953125 GiB;
- 4.15 realized bits per stored expert value;
- 192,675,840 selector blocks scanned, all with popcount 12.

The byte rate follows from
`8 × 63,968,378,880 / 123,312,537,600 = 4.15`. The 49-block authority is
distinct from the historical 48-block artifact, and MTP-unique non-expert
tensors are not attached to this extension. The selector count is a
metadata-geometry scan, not a throughput measurement.

### MIX34 quality boundary (`BLOCKED`)

A candidate perplexity value of 5.4587 exists for a 48-block physical
provenance artifact, but its matching reference run was deferred and the
corrected 49-block authority has no matching reference graph. This log does not
call that candidate a quality-parity, Q6/Q8/BF16-equivalent, or 49-block
quality result. A synthetic quality harness exists for deterministic wiring
checks; its random-weight floor does not establish language quality.

### MIX34 warm-cache model (`MODELED`, not deployed, 2026-09-13)

The warm-cache design treats one `(block, expert)` gate/up/down triplet as the
cache unit. The retained partial trace covers 31 of 48 layers, 28 token records
per layer, and 8,680 routed selections — it is not current live 48-layer
routing. On that trace a global LRU model reports 0.255529 GiB/token logical
payload traffic (0.256718 GiB/token after 4 KiB page rounding), and a 1 GiB
global-LRU row reaches a modeled 0.652880 hit rate. Larger rows change capacity
but do not turn a partial trace into live runtime evidence.

The separate 49-block selected-floor arithmetic is
`1,249,382,400 bytes/token = 1.163578 GiB/token` before reuse. These are
**logical bytes per token**, not storage bandwidth, and multiplying them by a
hoped-for tokens/s target is a scenario calculation, not a benchmark.

### Negative storage controls (`MEASURED`, regime-limited, 2026-09-12)

The historical 0.3495 tokens/s control and the 0.3718766 tokens/s control are
classified `OUT_OF_CORE_NVME_THRASH_REGIME`. They measure end-to-end latency
while storage and page faults dominate. They are not kernel throughput, not a
resident-design slowdown measurement, and not evidence against the resident
design.

### Native MTP reconstruction (`MEASURED` inventory, `EXPERIMENTAL` fixture, 2026-09-12)

The BF16 source contains 31 `mtp.*` tensors totalling 5,214,301,696 bytes: 24
per-layer tensors and 7 outer `nextn.*` tensors. The native reconstruction is
one full-attention layer with hyper-connections, the QSA indexer, routed/shared
MoE, shared embeddings/head, and a conditioned multi-stream hidden passed to
the next draft step — sequential speculation, not independent lightweight
heads.

| Scope | Result | Public status |
| --- | --- | --- |
| Synthetic graph/transaction fixture | Full-accept, first-reject, partial-accept, and repeated-mixed transaction scenarios pass within the fixture contract. | `EXPERIMENTAL`; fixture-only. |
| Synthetic paired timing | `MTP_NET_FIXTURE = 0.3275` from BASE 21.64 t/s and MTP 7.09 t/s, with 0 of 291 random-weight drafts naturally accepted. | `MEASURED` fixture result; not production economics. |
| Real trained source | 31 MTP tensors with shapes and bytes present in the source inventory. | `MEASURED` source fact. |
| Real trained artifact and quality | Production artifact/load, real acceptance, real generation quality, and real paired `MTP_NET`. | `UNMEASURED` / `BLOCKED`. |

The fixture value must not be substituted for a real trained acceptance rate or
MTP multiplier.

## Dated research timeline

| Date | Record | Evidence and boundary |
| --- | --- | --- |
| 2026-02 | Requested historical pointer | A February 2026 MIX34 warm-cache label was requested, but no recovered authority receipt carries that date. Retained as `UNVERIFIED`, not back-dated. |
| 2026-08-20 | Dense roof provenance | Archived reconstruction separating 410/821 TOPS, physical bandwidth, compute-equivalent bandwidth, and useful token rate. `HISTORICAL`. |
| 2026-08-26 | Flash source integrity | Public-source metadata and safetensors headers reconcile 1,658 tensors and the MTP class inventory. `MEASURED` source audit. |
| 2026-09-12 | MIX34 authority and negative control | 49-block geometry reconciled; 0.3495 and 0.3718766 t/s frozen as out-of-core NVMe-thrash controls. `MEASURED`, narrow scope. |
| 2026-09-12 | MTP fixture tranche | Native graph/state-transaction validation and fixture timing, recorded separately from real trained-weight readiness. `EXPERIMENTAL`. |
| 2026-09-13 | Warm-cache V1 design | Partial-trace global-LRU and page-rounded traffic modeled; runtime integration and live route capture pending. `MODELED`. |
| 2026-09-13 | Route traces captured | 6 workload classes × 257 steps × 48 layers, top-10 over 512 experts; 740,160 selections. `MEASURED`, captured via a diagnostic dev hook at 2.75× cost. |
| 2026-09-16 | Activation capture | 2,048-token calibration pass over all 48 layers at the routed-expert input, plus a 512-token holdout pass, using a CPU executor. Foundation for the representation panels. `MEASURED`. |
| 2026-09-18 | Representation panels | T1 / LS-T1 / T2 / Q2 / 2-bit-codebook comparison, equal-byte correction panel, and a Q8 residual island on one expert. `MEASURED`, tensor- and activation-level only. |
| 2026-09-18 | Deployment coherence | Coherent end-to-end generation; 12/12 canaries correct; two 128-token completions; 262,144-token Q8 KV pass. `MEASURED`, short-context evaluation. |
| 2026-09-18 | Route-aware prefetch pair | Matched control/prefetch pair on identical weights: 3–5× decode, 2.56× wall clock, ~287× fewer major faults. `MEASURED`, single pair with stated caveats. |
| 2026-09-18 | V2 hot region and streamed arms | BF16-derived q4_K/q4_K/q4_0 hot bank over the hottest 42% of units; streamed arms measured 0.73–1.29 t/s. `EXPERIMENTAL`; streaming `REJECTED` as architecture. |
| 2026-09-19 | Alice bring-up | Loader receipt at `2026-09-19T11:05:00Z`: a second model family, custom hybrid KDA linear-attention + MoE, 48 blocks, 512 routed experts top-10 plus one shared, 262144 context, 1,287/1,287 trunk tensors mapped. `MEASURED`, campaign start. |
| 2026-09-19 | Alice artifact | Mixed-quantization GGUF built: 39,913,721,760 B at 4.003 effective bpw, expert bank 34.7227 GiB (93.41 % of artifact bytes) at 3.862 bpw. `MEASURED`. |
| 2026-09-19 → 2026-09-21 | Alice host arena | Explicit arena over a packed `O_DIRECT` store: 4.49 t/s cold mmap → 13.99 (8 GiB) → 14.93 (12 GiB shipped) → 15.58 (pinned); 95.308 % hit; destructive parity `max|diff| = 0.0`. `MEASURED`. |
| 2026-09-22 | Alice MTP rollback root cause | Alice-local recurrent conv snapshot plane convention (`min(slot, n_seq_tokens)` instead of `n_seq_tokens - slot`), not shared infrastructure; KAT 276/682 failing → 0/682; greedy parity at K = 0/2/3/4. `MEASURED`. |
| 2026-09-22 | Alice prefill hot run | 662.7437 t/s at `pp2048`/ubatch 4096 as the hot request of a warm-repeat (cold 91.345, warm 245.614); ub512 band 486.9–533.0 t/s; ub2048 unusable as a control. `MEASURED`, state-sensitive, not a stable baseline. |
| 2026-09-22 | Backend pivot | ROCm/HIP adopted as the production and performance backend with Vulkan as parity oracle and mechanism donor; current clean HIP raw K0 15.374–15.422 t/s against the historical Vulkan 18.430 configuration of record. `MEASURED` + `HISTORICAL`. |
| 2026-09-22 | Host-KV reuse on ROCm | Physical host read held at the ~14 GB/s link roof with logical KV service to 244.944–246.89 GB/s; exact-attention parity rel_rms 4.5e-6, max_abs ≤ 2.4e-7; unregistered pageable host memory faults the GPU, so registration is a correctness prerequisite. `MEASURED`. |
| 2026-09-22 | Staging/overlap null | F2 copy stream measured 515.858 pp/s control against 507.085 pp/s candidate; the synchronization drains it targeted priced out at ~0.2 % of a prefill pass; coarse duplication failed allocation at 49,326.56 MiB. `MEASURED`, null. |
| 2026-09-22 | Expert-major grouping | Single-layer local GEMM 0.185 → 6.115 TMAC/s with 32/32 bit-exact same-kernel parity; deployed-configuration gate `REVERT`; CPU-MoE prefill measured −85.1 %. `MEASURED` locally, `INVALIDATED` for deployment. |
| 2026-09-22 | Qwen3.8-27B ROCm ladder | Raw K0 17.35 t/s; accepted 29.19 / 34.25 / 33.90 at K=1/2/4; `iq4_nl` KV has no HIP flash-attention kernel and falls back to the CPU silently before faulting. `MEASURED`. |
| 2026-09-22 | Representation utilization | Measured unpack ordering 297.6 → 229.7 → 188.8 GB/s at a real expert shape; GSQ and RCO adopted as prior art; the Flash-Next compression target is `MODELED` only. `MEASURED` + `MODELED`. |

## 2026-09-19 → 2026-09-22: second campaign and the backend pivot

The records above are summarized here and documented in full in
[`research/alice/`](../alice/), [`research/host-kv/`](../host-kv/),
[`research/qwen27b/`](../qwen27b/), [`research/ssd-action-memory/`](../ssd-action-memory/),
and [`research/representation/UTILIZATION.md`](../representation/UTILIZATION.md).

Three points about this window are easy to get wrong and are therefore stated
explicitly:

- **The Alice campaign is about three days old** (2026-09-19 → 2026-09-22), not
  months. The surrounding program is older; this campaign is not.
- **Alice throughput records are backend-labelled.** The 18.430 t/s figure is
  the historical *Vulkan* configuration of record; current clean ROCm/HIP raw
  K0 is 15.374–15.422 t/s. They are different backends and different source
  trees.
- **The 662.74 t/s prefill figure is a hot run**, the third request of a
  warm-repeat in one session. It is not a universally reproducible stable
  baseline, and the cross-session 71.105 t/s comparison behind the "9.3×"
  headline is inadmissible.

The window also produced a large set of rejections, which are recorded rather
than dropped:
[`research/falsified/ALICE_CAMPAIGN_NEGATIVES.md`](../falsified/ALICE_CAMPAIGN_NEGATIVES.md).

## Current status

| Area | Status | What is safe to say | What is not claimed |
| --- | --- | --- | --- |
| Deployment coherence | `MEASURED` | Coherent generation, 12/12 canaries correct, 128-token completions | Quality parity with BF16 |
| 262,144-token context | `MEASURED` | Capacity/coherence pass with Q8 KV | Long-context quality, or a 384K result |
| Deployment memory profile | `MEASURED` | Scaffold VRAM ~4.42–4.47 GiB, GTT ~0.11–0.13 GiB, RSS ~11.5–12.6 GiB | Resident full-model throughput |
| Route-aware prefetch | `MEASURED`, single matched pair | 3–5× decode, 2.56× wall clock, ~287× fewer major faults on identical weights | A quantization gain or a general multiplier |
| Route distribution | `MEASURED` | Hottest 42% of units carry 90.2% of routed mass | A stable invariant across prompts |
| V2 hot region | `EXPERIMENTAL` | BF16-derived, ancestry-verified, weight relRMS 0.072–0.087 | Final ancestry or final quality |
| Streamed V2 arms | `MEASURED`, `REJECTED` as architecture | 0.73–1.29 t/s at 0.23–1.67 GB/token | The final architecture |
| Representation panels | `MEASURED`, bounded | Base and equal-byte correction comparisons | Capability or BF16-retention evidence |
| Final V2 bank | `UNMEASURED` | Migration from BF16 in progress | Any V2 quality or sustained-speed number |
| Native MTP real artifact | `BLOCKED` | Source tensors and graph contract documented | Real acceptance, quality, or `MTP_NET` |
| 80 / 250 tokens/s | `HYPOTHESIS` | Falsifiable research targets | A benchmark, forecast, or promise |

## Near-term order

1. make the high-route-mass region a persistent hot/warm cache hit instead of a
   per-token stream, then re-measure non-thrashing resident throughput;
2. replace the remaining donor-derived regions with BF16-derived representations
   region by region, and re-run the deployment canaries and sustained tests on
   the resulting bank;
3. close the final V2 quality question with a matched reference rather than a
   local-fidelity panel;
4. extend the context ladder beyond 262,144 tokens and record a 384K
   daily-driver result;
5. finish the real MTP artifact/load gate and measure real acceptance and paired
   `MTP_NET` under matched base/MTP conditions;
6. evaluate the 80 and 250 tokens/s targets only after those lanes close.

Joint integration is intentionally last. HAR documentation and lineage remain
visible through the [implementation map](../implementation-map.md),
[MoE residency overview](../moe-residency/README.md),
[ExpertPack research note](../systems/ExpertPack.md), and the existing
[Flash campaign summary](CURRENT_CAMPAIGN.md).

## Public references

- [Qwen3.8-Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [AMD Radeon RX 9060 XT specifications](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9060xt.html)
- [AMD RDNA4 instruction-set reference](https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture)
- [`PUBLICATION_ALLOWLIST.md`](../../PUBLICATION_ALLOWLIST.md)
- [`PUBLICATION_DENYLIST.md`](../../PUBLICATION_DENYLIST.md)
