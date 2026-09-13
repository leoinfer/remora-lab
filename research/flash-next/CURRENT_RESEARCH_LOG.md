# Qwen3.8 Flash research log

**Updated:** 2026-09-13 evidence freeze
**Scope:** public-safe research summary; no model payloads, raw receipts, private paths,
host identifiers, branch names, or unpublished identities are reproduced here.

This log keeps the current Flash-Next work visible without turning a bounded
experiment into a product or performance claim. The public status vocabulary is
explained in [`docs/methodology.md`](../../docs/methodology.md) and the claims
ledger remains authoritative in [`CLAIMS.md`](../../CLAIMS.md).

## Scope and prioritization

HAR is the production runtime and control plane in this repository. Its public
scope is native Rust host/runtime code, Rust Vulkan resource and dispatch
boundaries, model/package and storage contracts, scheduling, residency, and
correctness gates. Research-only scripts and external reference runtimes may
produce evidence, but they are not HAR runtime dependencies or proof of HAR
coverage.

The current Qwen3.8 work is split into two independent lanes:

1. **Native MTP, primary:** recover the trained MTP block, build the native
   graph, prove state transactions, then measure placement, quality, and
   `MTP_NET` on the real artifact.
2. **MIX34/R4X residency, secondary:** freeze the corrected 49-block geometry,
   pair quality against a matching reference, integrate route-aware hot/warm/
   cold residency, and only then measure non-thrashing resident throughput.

The lanes must not share a speed, quality, or acceptance number until each has
an independent finish line. This is the current prioritization pivot: MTP gets
the first heavy validation slot; MIX34 remains useful low-memory analysis while
its live route and resident-throughput gates are pending.

For the adjacent Qwen3.8-27B line, the intended MoE architecture is explicit
expert residency, not CPU-offload execution: keep the dense core and ready
expert slots device-local, use bounded host memory as a refill/cache tier, and
keep canonical weights in the cold storage tier. HAR's residency contracts track
slot readiness, in-flight movement, and eviction separately. This is an
architecture and scheduling boundary, not a claim that a full expert bank is
currently resident or fast.

## Evidence vocabulary

Use the following labels in research notes and receipts; never promote a label
without its stated boundary:

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
| `REJECTED` | A proposed interpretation failed its own acceptance rule. |

`VERIFIED`, `EXPERIMENTAL`, `HISTORICAL`, and `INVALIDATED` remain the claims
ledger statuses. A value can be measured in a narrow experiment and still be
unverified as a broader product claim.

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
quantized weights do not receive that multiplier. The [dense GPU roofline provenance](../../docs/dense-roofline.md)
keeps physical bandwidth, compute-equivalent bandwidth, and useful token rate
separate; the older approximately 154 TB/s figure is a six-bit compute-
equivalent normalization, not physical VRAM bandwidth.

The **250 tokens/s** figure is therefore a research target, not a benchmark or
promise. Under the same shorthand it represents about 3 TOPS of useful
arithmetic, roughly 0.73% of the published 410-TOPS dense headline. It is high
enough to force an end-to-end answer for data movement, expert residency,
state transactions, scheduling, and kernel efficiency while remaining far below
the idealized arithmetic roof. The target is useful as a falsifiable systems
budget; it is not evidence that the current runtime approaches it.


## MIX34 warm-cache and residency record

### Geometry authority (`MEASURED` + `DERIVED`)

The corrected fullbank authority is a 49-block, trunk-only extension:

- 147 expert tensors;
- 123,312,537,600 expert values;
- 63,968,378,880 expert payload bytes;
- 59.5751953125 GiB of expert payload;
- 4.15 realized bits per stored expert value;
- 192,675,840 selector blocks scanned, all with popcount 12.

The byte rate follows from `8 × 63,968,378,880 / 123,312,537,600 = 4.15`.
The 49-block authority is distinct from the historical 48-block artifact. MTP-
unique non-expert tensors are not attached to this MIX34 extension.

The 192,675,840 count is a selector/metadata geometry scan, not a throughput
measurement. The fullbank byte count is a physical payload authority; neither
number establishes end-to-end generation speed.

### Quality boundary (`MEASURED` candidate only; `BLOCKED` paired verdict)

A candidate perplexity value of 5.4587 exists for a 48-block physical
provenance artifact. Its matching reference run was deferred, and the corrected
49-block authority has no matching 49-block reference graph. Therefore this
log does not call the candidate quality-parity result, a Q6/Q8/BF16-equivalent
result, or a 49-block quality result.

A separate synthetic quality harness is useful for deterministic wiring checks,
but its random-weight floor does not establish language quality. Quality remains
a gate, not a side effect of storage geometry.

### Warm-cache model (`MODELED`, not deployed)

The warm-cache design treats one `(block, expert)` gate/up/down triplet as the
cache unit. The retained partial trace covers 31 of 48 layers, 28 token records
per layer, and 8,680 routed selections; it is not current-live 48-layer routing.
On that trace, a global LRU model reports:

- `0.255529 GiB/token` logical payload traffic;
- `0.256718 GiB/token` after 4 KiB page rounding;
- a 1 GiB global-LRU row already reaches a modeled 0.652880 hit rate;
- larger rows change capacity but do not turn the partial trace into live
  runtime evidence.

The separate 49-block selected-floor arithmetic is
`1,249,382,400 bytes/token = 1.163578 GiB/token` before reuse. These values are
**logical bytes per token**, not storage bandwidth. A bandwidth claim requires
live route IDs, live cache counters, device I/O accounting, and a measured token
rate. Multiplying a modeled GiB/token value by a hoped-for tokens/s target is a
scenario calculation, not a benchmark.

The intended implementation remains explicit:

```text
hot:  ready route-aligned expert slots in device-local memory
warm: bounded host-memory triplet cache with identity and integrity checks
cold: canonical expert payload in storage, fetched only through admission
```

The warm-cache plan is `MODELED` and runtime integration is not complete.
Full-model resident throughput is `UNMEASURED`. No RAM purchase, full-bank
residency, or CPU-offload speed claim follows from the trace.

### Negative storage result (`MEASURED`, regime-limited)

The historical 0.3495 tokens/s control and the later 0.3718766 tokens/s control
are both classified as `OUT_OF_CORE_NVME_THRASH_REGIME`. They measure end-to-end
latency while storage and page faults dominate. They are not MIX34 kernel
throughput, not a 48x slowdown, and not evidence against the resident design.
The non-thrashing resident full-model rate remains `UNMEASURED`; an approximately
20.2 tokens/s resident-traffic number remains `MODELED` only.

## Native MTP reconstruction

### What survived (`MEASURED` source inventory; `EXPERIMENTAL` implementation)

The BF16 source contains 31 `mtp.*` tensors totaling 5,214,301,696 bytes
(4.8562 GiB): 24 per-layer tensors and 7 outer `nextn.*` tensors. The native
reconstruction is one full-attention Qwen4Exp layer with hyper-connections, the
QSA indexer, routed/shared MoE, shared embeddings/head, and a conditioned
multi-stream hidden passed to the next draft step. It is sequential speculation,
not a set of independent lightweight heads.

The state transaction must cover the MTP KV/indexer cache, trunk attention KV,
GDN recurrent state, PLE convolution history, QSA indices, positional/token
history, and MTP-private state. The transaction shape is:

```text
begin → speculate → verify leading matches → commit accepted prefix
                                      ↘ discard/recover rejected suffix
```

This preserves the HAR principle that storage presence, execution readiness,
and committed model state are separate facts.

### Synthetic fixture versus real trained weights

These evidence classes must remain separate:

| Scope | Result | Public status |
| --- | --- | --- |
| Synthetic graph/transaction fixture | Graph conformance and full-accept, first-reject, partial-accept, and repeated-mixed transaction scenarios pass within the fixture contract. | `EXPERIMENTAL`; fixture-only. |
| Synthetic paired timing | `MTP_NET_FIXTURE = 0.3275` from BASE 21.64 t/s and MTP 7.09 t/s, with 0 of 291 random-weight drafts naturally accepted. | `MEASURED` fixture result; not production economics. |
| Real trained source | 31 MTP tensors and their shapes/bytes are present in the source inventory. | `MEASURED` source fact. |
| Real trained artifact and quality | Production artifact/load, real acceptance, real generation quality, and real paired `MTP_NET` are not established by the fixture. | `UNMEASURED` / `BLOCKED`. |

The fixture value must not be substituted for a real trained acceptance rate or
MTP multiplier. Upstream n-gram examples and another architecture's MTP runs are
not Qwen3.8 evidence.

## Dated research timeline

| Date | Record | Evidence and boundary |
| --- | --- | --- |
| 2026-02 | Requested historical pointer | A February 2026 `MIX34` warm-cache label was requested, but no recovered authority receipt carries that date. It is retained as `UNVERIFIED`, not back-dated evidence. |
| 2026-08-20 | Dense roof provenance | The archived dense-roof reconstruction separates 410/821 TOPS, physical bandwidth, compute-equivalent bandwidth, and useful token rate. The approximately 154 TB/s value is derived six-bit normalization, not physical VRAM traffic. `HISTORICAL` context. |
| 2026-08-26 | Flash source integrity | Public-source metadata and safetensors headers reconcile 1,658 tensors and the MTP class inventory. Model weights remain outside this repository. `MEASURED` source audit. |
| 2026-09-12 | MIX34 authority and negative control | 49-block geometry was reconciled; 0.3495 and 0.3718766 tokens/s were frozen as out-of-core NVMe-thrash controls; resident full-model throughput stayed unmeasured. `MEASURED` with narrow scope. |
| 2026-09-12 | MTP fixture tranche | Native graph/state transaction validation and fixture timing were recorded separately from real trained-weight readiness. `EXPERIMENTAL` / fixture-only. |
| 2026-09-13 | Warm-cache V1 design | Partial-trace global-LRU and page-rounded traffic were modeled; 0.255529–0.256718 GiB/token is logical traffic, not live bandwidth. Runtime integration and live route capture remain pending. `MODELED` / `UNMEASURED`. |

The dated records above are sanitized aliases for the local research evidence
set. Raw receipts, private execution context, and model payloads are deliberately
not copied into the public candidate.

## Current status table

| Area | Status | What is safe to say | What is not claimed |
| --- | --- | --- | --- |
| HAR runtime scope | `EXPERIMENTAL` | Rust runtime/control and bounded residency contracts are public. | Full Flash-Next production coverage. |
| Flash metadata | `DERIVED` | The model card/config support the active-parameter and architecture arithmetic above. | A measured Flash tokens/s result. |
| MIX34 geometry | `MEASURED` + `DERIVED` | 49-block payload and selector geometry reconcile. | 49-block quality parity. |
| MIX34 warm cache | `MODELED` | Partial-trace LRU and logical traffic are explicit. | Deployed cache throughput or bandwidth. |
| MIX34 resident decode | `UNMEASURED` | A non-thrashing run is the next authority. | 20.2 or 250 tokens/s as measured. |
| Native MTP fixture | `EXPERIMENTAL` | Graph, transaction, and fixture timing seams are testable. | Production acceptance or `MTP_NET_REAL`. |
| Native MTP real artifact | `BLOCKED` | Source tensors and graph contract are documented. | Real-weight generation quality, acceptance, or speed. |
| 250 tokens/s | `HYPOTHESIS` / target | A falsifiable end-to-end systems target. | A result, forecast, or guarantee. |

## Automation motivation and near-term roadmap

Local researcher automation exists to make long hardware experiments reviewable:
freeze artifact identity, record command and resource provenance, keep lane
ownership explicit, preserve negative results, and stop a modeled value from
silently becoming a benchmark. Automation reduces coordination cost; it does
not replace correctness gates or make AI-assisted work evidence.

Near-term order:

1. finish the real MTP artifact/load gate from the 31-tensor source inventory;
2. run real-weight graph and state-transaction parity, including full, first-
   reject, partial, and repeated-mixed cases;
3. measure real MTP acceptance, quality, traffic, and `MTP_NET` under matched
   base/MTP conditions;
4. run a matched 48-block quality pair and resolve the corrected 49-block
   reference boundary for MIX34;
5. integrate live route-aware hot/warm/cold residency and prove a non-thrashing
   resident run with storage counters and correctness checks;
6. evaluate the 250-token target only after the independent evidence lanes close.

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
