# REMORA Lab

REMORA Lab is an open local-AI systems research repository about making large,
sparse models practical on constrained consumer hardware. It is a working
record of experiments, measurements, failures, and falsifiable targets, not a
product announcement: most numbers here are hardware-specific, several
directions were killed after measurement, and the current configuration is a
scaffold rather than a finished model.

The reference machine is one desktop: a 16 GB RDNA4 Radeon RX 9060 XT, 32 GB
of system RAM, and NVMe storage, on Linux with RADV/Vulkan. The current primary
target is **Qwen3.8 Flash-Next**, a 125B-parameter hybrid-attention MoE with
roughly 6B activated parameters and a 512-expert routing layer. It fits in
neither memory tier, which is the entire point of the project.

## Start here

Reader shortcut, in roughly the order a new visitor should ask:

| Question | Where to look |
| --- | --- |
| What currently works? | [Current verified results](#current-verified-results), [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md) |
| What changed after 2026-09-18? | [Since the last update](#since-the-last-update-2026-09-19--2026-09-22), [Alice campaign](research/alice/README.md), [host-KV](research/host-kv/README.md) |
| What is being tested right now? | [current Flash-Next log](research/flash-next/CURRENT_RESEARCH_LOG.md), [campaign summary](research/flash-next/CURRENT_CAMPAIGN.md) |
| What failed, and what was killed? | [falsified and bounded results](docs/research/falsified-results.md), [`research/falsified/`](research/falsified/) |
| Which results are real, and at what scope? | [`CLAIMS.md`](CLAIMS.md), [methodology and evidence labels](docs/methodology.md) |
| Where are the receipts? | [`TECHNICAL_ARTIFACT_INDEX.md`](TECHNICAL_ARTIFACT_INDEX.md), [`repro/`](repro/) |
| What are the moonshots? | [RDNA4 / bandwidth / MTP research](#rdna4--bandwidth--mtp-research), [`RESEARCH_IDEA_INDEX.md`](RESEARCH_IDEA_INDEX.md) |
| What machine produced these numbers? | [`HARDWARE_PROFILE.md`](HARDWARE_PROFILE.md) |
| How do the ideas map to code? | [`research/implementation-map.md`](research/implementation-map.md) |

## What is it?

REMORA Lab is an umbrella repository for local-AI systems research: model
representation, memory hierarchy, MoE expert residency, storage behaviour,
quantization, speculative decoding, GPU execution on RDNA4, context systems,
and the accounting that keeps those measurements honest.

Three names are worth separating before reading further:

- **REMORA** is the research program: precision and residency policy for sparse
  models on hardware that cannot hold them.
- **HAR** (`har/`) is an experimental native-Rust runtime and control-plane
  side lane developed inside the same program. It is public and it is
  maintained, but it is **not** the runtime that most current model experiments
  execute on. See [HAR side lane](#har-side-lane).
- **The research corpus** (`research/`, `docs/`, `repro/`) is the part that
  matters most right now: dated logs, measured panels, negative results, and
  reproduction lanes.

Everything else — HERMES, R4X, R4KV, R4F, ContextFold, Laguna, HAR-X,
ExpertPack, DSpark-MTP — is a named sub-project reachable from the
[research idea index](RESEARCH_IDEA_INDEX.md) and the
[implementation map](research/implementation-map.md). None of them are
prerequisites for understanding the current work.

## What am I trying to do right now?

Current target configuration:

```text
models     Qwen3.8 Flash-Next (125B total / ~6B active, 512-expert MoE)
           Alice (80B-A3B, 48 blocks, 512 experts top-10, 262144 context)
           Qwen3.8-27B (ROCm side campaign, MTP ladder)
gpu        RX 9060 XT 16 GB (gfx1200 / RDNA4)
backends   ROCm/HIP production, RADV/Vulkan as parity oracle and donor
host       32 GB system RAM, NVMe storage, Linux
question   how close to BF16 quality, and how fast, can this get?
```

Current themes:

- heterogeneous precision across expert roles and layers;
- MoE expert residency across VRAM, RAM, and NVMe, with an explicit host arena;
- route-aware prefetch and expert caching;
- utilization-first representation design: keep execution fast, then descend bits;
- storage-path measurement, page-cache behaviour, and fault accounting;
- MTP / speculative future-state research, including rollback correctness;
- grouped and RDNA4-native execution;
- effective (logical) bandwidth versus physical bandwidth — and block reuse that
  converts one into the other;
- context systems, long-context KV behaviour, and host-resident KV;
- falsifiable roofline targets.

The question is no longer "can it run?". It is how much quality can be kept
and how much speed can be recovered while the model lives on 16 GB of VRAM,
32 GB of RAM, and an NVMe device.

## Current verified results

**As of 2026-09-18.** These are measured on the reference machine, not
extrapolated. Scope and limits are attached to each line on purpose.

Deployment:

- Qwen3.8 Flash-Next generates coherent text end-to-end on the RX 9060 XT.
- 12 / 12 deployment canaries produced correct outputs on the coherent scaffold
  (the verdict is evaluation of printed outputs; the harness has no automated
  grader).
- Two 128-token completions ran to their token budget in one sustained run.
- 262,144-token context passed with the Q8 KV configuration (capacity and
  coherence smoke only).
- Scaffold VRAM approximately 4.42–4.47 GiB; GTT approximately 0.109–0.133 GiB.
- RSS approximately 11.5–12.6 GiB, with a 19.40 GiB `MemAvailable` floor in the
  recorded deployment.
- The host-tier expert bank is a file-mapped, page-backed region the loader
  accounts at 35,129.29 MiB (approximately 34.3 GiB); only its resident pages
  count toward RSS.

Mechanism result — same weights, same model, control versus route-aware
prefetch:

| Metric | Control | Route-aware prefetch |
| --- | --- | --- |
| Decode, long-explain | 1.28 t/s | 3.86 t/s |
| Decode, long-code | 0.81 t/s | 4.07 t/s |
| p50 token latency | ~702–1189 ms | ~224–233 ms |
| p95 token latency | ~1634–2089 ms | ~362–475 ms |
| Wall clock | 325.4 s | 127.0 s |
| NVMe bytes read | 151.7 GB | 74.7 GB |
| Major page faults | 3,931,682 | 13,714 |

Interpretation: roughly 3–5× decode improvement and a 2.56× shorter wall clock
on identical weights, with about 287× fewer major faults and about half the
device read volume. This is a **storage and residency path result**. It is not
a quantization improvement, it is one matched pair of runs on the reference
machine, and it is not evidence that the same multiplier is available to every
future configuration — the device counters are machine-wide, the two arms ran
back-to-back, and the full caveat list is part of the record:
[`research/flash-next/PREFETCH_RESULT.md`](research/flash-next/PREFETCH_RESULT.md).

Representation (tensor- and activation-level panels, expert-weight scope). Two
different metrics appear here and must not be merged; weight relRMS compares a
reconstructed tensor against the BF16 authority, activation error compares
forward-pass outputs on captured expert inputs:

| Representation | Bits/weight | Weight relRMS | Activation error |
| --- | --- | --- | --- |
| Raw ternary T1 | ~1.75 | ~0.776 | ~0.778 |
| LS-reencoded T1, same bytes | ~1.75 | ~0.609 | ~0.611 |
| T1 + dense ternary T2 | ~3.50 | ~0.438 on the measured slice | — |
| Q2_0 donor scaffold | ~2.25 | ~0.459–0.488 | ~0.444 |

- The LS re-encode is a large free gain: the same ternary container at the same
  byte cost drops weight relRMS from ~0.776 to ~0.609.
- Dense T1+T2 is not compelling: it spends roughly 3.5 bits per weight to reach
  a weight relRMS the Q2 reference nearly matches at ~2.25.
- On the equal-added-byte panel, a 2-bit residual codebook removed ~47.7% of
  activation error where dense ternary T2 removed ~20.8% — roughly 2.3× the
  correction for the same added bytes.

These are fidelity measurements, not capability measurements: the teacher-KL
probe for these candidates was never captured. Details, boundaries, and what is
*not* decided are in [`research/representation/`](research/representation/).

Routing and residency:

- The top ~42% of expert units carry ~90.2% of routed traffic on the current
  route trace.

What is **not** measured yet, and must not be assumed: full BF16 capability
retention, AutomationBench retention, Artificial Analysis retention, final V2
quality, final V2 sustained speed, the 384K daily-driver result, and real MTP
acceptance or speed multipliers.

The current deployment runs a **temporary Q2 scaffold/control**. To be explicit
about three different things that are easy to confuse:

```text
Q2 donor/scaffold   known-good runtime control; not a quality target
BF16                the quality authority
final V2 bank       being regenerated directly from BF16
```

## Since the last update (2026-09-19 → 2026-09-22)

The previous public refresh was the 2026-09-18 Flash deployment record. Since
then the program added a second model campaign, a backend pivot, and the
strongest mechanism result in the corpus so far. Short version, with the full
records linked:

**Alice campaign — active for about three days.** A second model family
(`AliceAI-Foundation-80B-A3B-Base`, a custom hybrid KDA linear-attention + MoE
model, 48 blocks, 512 routed experts with top-10 routing, 262144 context) was
brought up, quantized to a 39.91 GB expert-dominated artifact at 4.003 effective
bits per weight, and run behind an explicit 12 GiB host expert arena that moved
decode from 4.49 to 15.58 t/s. Its `K > 1` MTP rollback failure was traced to an
Alice-local recurrent conv snapshot plane convention — not shared
infrastructure — and fixed, taking a model-free known-answer test from 276 of
682 checks failing to zero, with greedy parity at `K = 0/2/3/4`. Its prefill
record includes a 662.74 t/s hot run that is explicitly **not** a stable
baseline. Everything is in [`research/alice/README.md`](research/alice/README.md).

**Backend pivot.** ROCm/HIP is now the production and performance backend, with
Vulkan kept as the parity oracle, debug path, and mechanism donor. New
optimizations target HIP first. This does not retroactively change the
historical Vulkan numbers: the 18.430 t/s Alice decode record stays labelled as
the historical Vulkan configuration of record.

**Host-KV block reuse — the strongest mechanism result.** Fetching a host KV
block once and serving many query rows from it converts a ~14 GB/s physical
host read path into up to ~245 GB/s of *logical* KV service at exact-attention
parity (rel_rms 4.5e-6), on both backends. Physical bandwidth never moved; the
reuse did. Registered host memory turned out to be a correctness prerequisite
for direct host-KV kernel access, not a tuning knob. Integration into the
production attention path is the named remaining work. See
[`research/host-kv/README.md`](research/host-kv/README.md).

**Utilization-first representation.** The representation objective is now
stated as maximum fidelity × minimum bytes × maximum execution speed, in that
priority order: utilization first, bits-per-weight second, quality as a hard
constraint. A measured unpack-cost ordering shows smaller formats can be
*slower*, GSQ and RCO are adopted as prior art, and the Flash-Next compression
target is explicitly `MODELED`. See
[`research/representation/UTILIZATION.md`](research/representation/UTILIZATION.md).

**Also new:** a Qwen3.8-27B ROCm side campaign with its own ladder and KV
defects ([`research/qwen27b/`](research/qwen27b/)), SSD action memory with a
retracted headline and an admissible clean multiplier
([`research/ssd-action-memory/`](research/ssd-action-memory/)), and a substantial
set of new negative results
([`research/falsified/ALICE_CAMPAIGN_NEGATIVES.md`](research/falsified/ALICE_CAMPAIGN_NEGATIVES.md)).

**Runtime side.** The findings above are half of this work; the other half is
the runtime that produced them. The consolidated runtime is now published as a
public fork of llama.cpp on the same GitHub account — repository
`leoinfer/llama.cpp`, branch `rdna4-rocm-2026-09-22` (currently its default
branch). It carries upstream master plus the `alice_ai` hybrid
linear-attention MoE architecture, the recurrent-snapshot rollback correctness
fix and its model-free known-answer test, the host expert tier and arena
plumbing, the readback/submission batching, and the MIX34 type, and it builds
ROCm/HIP and Vulkan from one tree. Verified here: `llama-cli`, `llama-server`,
`llama-bench`, and the snapshot KAT all build, and the KAT reports 682 checks /
0 failures. It is a research branch, not a proposed upstream change. This
repository keeps the measurements, the negative results, and the exact
configurations; the fork is where you run them.

## Why VRAM / RAM / NVMe tiering?

Flash-Next does not fit. The reconciled 49-block expert geometry is
63,968,378,880 expert payload bytes (59.58 GiB at 4.15 realized bits per stored
value), before truncation, KV cache, embeddings, or runtime state, and the
machine has 16 GB of VRAM and 32 GB of RAM. Something always has to be
elsewhere.

The naive answer — let the operating system page the weights from disk — is
measurable and bad. Two early end-to-end runs landed at 0.3495 and 0.3718766
tokens/s, both dominated by storage and page faults. Those numbers are retained
as `OUT_OF_CORE_NVME_THRASH_REGIME` controls: they are not kernel throughput,
not a resident-design slowdown measurement, and not evidence that a different
placement policy cannot work.

The current working thesis separates two axes that are usually collapsed:

```text
SENSITIVITY DECIDES PRECISION.
FREQUENCY / REUSE DECIDES RESIDENCY.
FUTURE-STATE INFORMATION DECIDES MOVEMENT.
```

A rare expert is not a low-quality expert. An expert that is cheap to store is
not automatically the right one to keep in the slow tier. The desired state
space is:

| State | Sensitive | Tolerant |
| --- | --- | --- |
| Hot | high fidelity + VRAM | compact + VRAM |
| Warm | high fidelity + RAM | compact + RAM |
| Cold | potentially high fidelity + NVMe | compact + NVMe |

The goal is **not** to maximise SSD usage. The goal is to let NVMe hold a large
*population* tail while the hardware serves a small *traffic* tail, freeing RAM
and VRAM for experts that are actually used.

## Current Qwen3.8 Flash-Next work

The dated, public-safe record is
[`research/flash-next/CURRENT_RESEARCH_LOG.md`](research/flash-next/CURRENT_RESEARCH_LOG.md).
It carries the per-date evidence boundary; this section is the summary.

**Model.** The official model card and configuration describe a 125B-parameter
hybrid-attention MoE with approximately 6B activated parameters, 512 experts
with 10 routed experts plus a shared expert, 48 trunk layers, an n-gram
vocabulary, and one native MTP layer. Those are model facts, not local
throughput results.

**Deployment.** The current coherent scaffold is a Q2 donor-derived bank
(approximately 2.25 bits per weight) combined with a host-tier expert region the
loader accounts at 35,129.29 MiB (approximately 34.3 GiB, page-backed). It
generates coherent text, produces correct outputs on its canary and sustained
suites, and holds a 262,144-token context in the Q8 KV configuration. It is a
control, not the finish line: BF16 remains the quality authority and the final
bank is being regenerated from BF16 rather than inherited from the donor.

**Runtime boundary.** Most current model experiments execute through
llama.cpp-derived research branches and the surrounding measurement tooling.
That runtime is an external reference implementation used as an experiment
platform; it is not part of HAR, and HAR does not depend on it. Conversely, the
HAR native path has not yet closed its own full-model generation gate — see
[`repro/flash-next/full-model/`](repro/flash-next/full-model/).

**Routing and the V2 experiment.** On the current trace, the top ~42% of expert
units account for ~90.2% of routed traffic. The V2 experiment rebuilds that
high-value region directly from BF16 — gate/up as Q4_K and down as Q4_0 — while
the remaining regions stay on the Q2 scaffold temporarily. This is a migration
configuration, and the remaining donor-derived region is **not** final
ancestry: the final bank replaces every permanent donor-derived region with a
BF16-derived representation.

**Why the streamed version is not the answer.** Streaming the ~42%
hot-population bank through the slow tier is bandwidth-bound. The streamed V2
arms measured 0.73–1.29 tokens/s decode at 0.229–1.667 GB/token of
device-normalized reads, against a modeled 1,262 MB/token for the 42% mask —
and the streamed arms were *slower* than the scaffold control. That is a useful
measurement, not a failure of the representation: it says expensive
high-route-mass expert weights have to become persistent hot/warm cache hits
instead of being read from NVMe on every token, and that this residency work is
the next major runtime step rather than an existing result.

**Intended execution architecture.** The target hierarchy is:

```text
HOT   GPU-resident / immediately executable experts
WARM  RAM-resident experts, fast promotion, host-visible fallback
COLD  genuinely rare NVMe-backed population tail
```

plus an optional one-shot bypass staging path for a cold expert that does not
deserve VRAM promotion. Normal-token execution should eventually be mostly
cache hits, and SSD should not be the normal expert-compute path. Route-aware
prefetch is already demonstrated to help dramatically
([`PREFETCH_RESULT.md`](research/flash-next/PREFETCH_RESULT.md)); persistent
hot/warm caching is the next major runtime step.

## Current representation research

Uniform ternary is **not** the current favoured answer. Measured bounded
panels changed that conclusion.

| Representation | Bits per weight | Measured local error |
| --- | --- | --- |
| Raw ternary T1 | ~1.75 | ~0.776 relRMS |
| LS-reencoded T1, same bytes | ~1.75 | ~0.609 relRMS |
| T1 + dense ternary T2 | ~3.5 | ~0.43 |
| Q2 scaffold reference | ~2.25 | ~0.44 |

Dense T1+T2 therefore spends roughly 1.55× the bytes of the Q2 reference for
comparable local error, which makes it uncompelling as a universal
representation.

Equal-added-byte correction panel, measured over the donor base — this is the
more interesting axis, because it asks what a fixed byte budget buys:

| Correction format | Activation-error removal over donor base |
| --- | --- |
| Dense ternary T2 | ~20.78% |
| Uniform 2-bit / 4-level residual codebook | ~47.66% |
| Rotated 2-bit codebook | ~47.78% |

The 2-bit residual codebook delivered roughly 2.3× the correction gain of dense
ternary T2 for the same added bytes, and rotation contributed almost nothing
beyond that. Measured over T1 instead, an LS-scale re-encode at identical bytes
removed ~62.9% of error where published dense ternary T2 removed ~46.0%.

Interpretation: correction sidecars are useful, but T2 is only one candidate
correction format, and ternary should not be assumed to be the universal base.

**High-precision island boundary.** A bounded Q8 residual-island test on
layer 0 / expert 283 moved T1 local error from ~0.779 to ~0.0032 using roughly
5.2 MB of total residual payload (~1.74 MB per expert-role). That proves the
additive island mechanism can recover near-BF16 local fidelity. It does **not**
prove the representation is efficient: T1 + a Q8 island is byte-inefficient
next to simply choosing a stronger base codec, so it is published as evidence
that optional high-fidelity corrective payloads work — not as the final
representation.

Current representation thesis:

```text
BF16 authority
   -> heterogeneous base representation
      + optional additive correction
```

Candidate palette: LS-T1, Q2-class, Q3-class, Q4-class and higher precision
where justified, a 2-bit residual codebook, T2 only where it measurably wins,
and selective high-precision islands when they pass value-per-byte tests.
Details: [`research/representation/`](research/representation/).

## RDNA4 / bandwidth / MTP research

Three quantities that get confused constantly, and are kept separate here:

```text
PHYSICAL BANDWIDTH              actual VRAM / PCIe / RAM / NVMe movement
LOGICAL / COMPUTE-EQUIVALENT    a normalization for how many logical weight
BANDWIDTH                       uses matrix hardware can service under reuse
OUTPUT TOKENS/S                 actual end-to-end generation
```

They are **not** interchangeable. The approximately 154 TB/s figure that
appears in this repository is a six-bit compute-equivalent normalization
derived from vendor INT4 throughput figures; it is not physical VRAM
bandwidth, and no device here moves 154 TB/s. The derivation is in
[`docs/dense-roofline.md`](docs/dense-roofline.md).

**MTP / future-state research.** Speculative future positions are a live
research interest: they may provide more rows to group per expert, useful
advance route information, better prefetch lead time, more expert-weight reuse,
and therefore higher effective (logical) bandwidth. None of that is
demonstrated yet. MTP does not create bandwidth; grouping and reuse may move
execution closer to the matrix roof. The 80 t/s and 250 t/s figures are
explicit moonshot / falsifiable research targets — not benchmarks, not
forecasts, not promises.

## What failed?

Negative results are kept, not tidied away.

| Result | Disposition |
| --- | --- |
| Proposed gfx1200 SWMMAC multi-POPS result | **Falsified.** Repeated-accumulator behaviour overcounted committed work. The method survived; the headline did not. |
| Naive out-of-core NVMe execution (0.3495 / 0.3718766 t/s) | **Rejected as an architecture.** Retained as `OUT_OF_CORE_NVME_THRASH_REGIME` controls. |
| "Maximise SSD usage" as a residency strategy | **Rejected.** Streaming high-route-mass expert weights per token is bandwidth-bound: the streamed V2 arms ran 0.73–1.29 t/s at 0.229–1.667 GB/token of device-normalized reads, slower than the scaffold control. |
| Dense ternary T1+T2 as a universal representation | **Not compelling.** It spends ~3.5 bits per weight to reach a weight relRMS the Q2 reference nearly matches at ~2.25. |
| T1 + Q8 residual island as the final representation | **Byte-inefficient.** Kept as mechanism evidence, not as a proposal. |
| Faster-than-baseline claims | **Not claimed.** Historical comparable paths sometimes trailed llama.cpp by several tokens per second; the project makes no speedup claim without a public receipt. |
| Effective "10M context" | **Hypothesis only.** Not a run of dense attention over ten million positions. |
| FreeToken-inspired copy-stream overlap (F2) | **Null.** 515.858 pp/s control vs 507.085 pp/s candidate at ubatch 512; the synchronization drains it targeted price out at ~0.2% of a prefill pass. |
| CPU-MoE prefill path | **Retired.** −85.1% (581.3 → 87.1 t/s at `pp4096`). |
| Coarse double staging (`n_copies = 2`) | **Memory-infeasible.** Allocation failure at 49,326.56 MiB (~48.2 GiB). |
| `ALICE_MOE_BLOCK` | **Reverted.** +6.2%/+17.8% adjacent pairs sit inside a 1.64× control spread; the motivating 2.7× projection is retired. |
| 100k raw decode under ordinary execution | **Closed by physics.** 676.9 op-TOPS required against a 590–630 band, and 18.0–20.25 GiB needed against an 11.38 GiB budget. |
| Structured sparsity as that lever | **Closed.** 563–637 credited op-TOPS against 821, and the representation loses 1.56× to eligible dense 2-bit. |
| Laya / System-1 decision sidecar | **Negative.** 38.4% agreement against an 84.9% constant baseline; model and dependencies removed. |
| SSD action-memory 5.10× headline | **Retracted as contaminated.** Admissible clean multiplier: 1.771×. |
| Copying the Vulkan reuse kernel to ROCm to recover decode speed | **Killed.** At full context the host-KV path is already faster on ROCm than on Vulkan (8.4–8.7 vs 5.38 t/s); single-stream decode has no rows to reuse. |

The full record: [`docs/research/falsified-results.md`](docs/research/falsified-results.md)
and [`research/falsified/`](research/falsified/).

## Repo map / where to look

| Path | What is in it |
| --- | --- |
| [`research/flash-next/`](research/flash-next/) | Current Flash-Next campaign, dated research log, prefetch result |
| [`research/alice/`](research/alice/) | Alice campaign: custom hybrid linear-attention MoE, MTP correctness, backend records, prefill, arena |
| [`research/host-kv/`](research/host-kv/) | Host-KV block reuse, huge-context behaviour, registered-host-memory prerequisite |
| [`research/qwen27b/`](research/qwen27b/) | Qwen3.8-27B ROCm side campaign: ladder, KV defects, wide-M economics |
| [`research/ssd-action-memory/`](research/ssd-action-memory/) | Symbolic action memory on the cold tier, with its retraction record |
| [`research/representation/`](research/representation/) | Representation research: bases, corrections, islands, palette |
| [`research/falsified/`](research/falsified/) | Negative knowledge, counterexamples, failure ledgers |
| [`research/ideas/`](research/ideas/), [`research/systems/`](research/systems/) | Idea atlas and named-system notes (HERMES, REMORA, R4X, R4KV, R4F, ContextFold, ExpertPack, …) |
| [`research/moe-residency/`](research/moe-residency/) | Expert residency, hot/warm/cold tiers, route-aware placement |
| [`research/mtp-speculation/`](research/mtp-speculation/) | MTP and speculative decoding research |
| [`research/experiments/`](research/experiments/) | E001–E096 experiment cards |
| [`docs/`](docs/) | Methodology, architecture notes, roofline provenance, hardware support |
| [`repro/`](repro/) | Executable, model-free reproduction lanes and bounded dispositions |
| [`formats/`](formats/) | R4X, R4KV, R4F format documentation |
| [`har/`](har/) | The native-Rust runtime side lane (see below) |
| [`benchmarks/local-bench/`](benchmarks/local-bench/) | MIT-licensed Rust estimator; research-only, not a HAR dependency |

## HAR side lane

**HAR (Hardware-Aware Runtime)** is an experimental native-Rust runtime and
control-plane project developed inside the same research program. It is public,
it builds, and it contains real work: Rust-only runtime design, a Rust Vulkan
resource and dispatch layer, storage and model-package contracts, scheduling
and state transactions, residency accounting, and R4KV/REMORA mechanisms.

Two honest boundaries:

- Most current model experiments execute on llama.cpp-derived research
  branches and external measurement tooling. That is a separate research
  platform, not part of HAR, and HAR does not depend on it.
- HAR has not yet closed its own full-model generation gate. Its bounded
  reproduction lane stays at
  [`repro/flash-next/full-model/`](repro/flash-next/full-model/).

So HAR should be read as a long-running runtime research track, not as the
thing a new reader must understand before learning what REMORA Lab currently
does. Its scope, release gates, and evidence remain documented in
[`har/README.md`](har/README.md), [`PUBLIC_HAR_RELEASE_AUDIT.md`](PUBLIC_HAR_RELEASE_AUDIT.md),
and [`research/systems/HAR.md`](research/systems/HAR.md).

## Methodology and evidence labels

Every result carries an evidence class, and no label is promoted without its
boundary:

| Label | Meaning |
| --- | --- |
| `MEASURED` | A direct run or physical scan with a recorded scope. |
| `DERIVED` | Arithmetic from measured metadata or a cited public specification. |
| `MODELED` | Replay, cache, or traffic model; not a live device result. |
| `EXPERIMENTAL` | A bounded implementation or fixture exists; coverage is incomplete. |
| `HYPOTHESIS` | A target or mechanism proposed for falsification, not an observation. |
| `UNMEASURED` | The required run has not produced an authority. |
| `BLOCKED` | A required artifact, comparator, or runtime gate is unavailable. |
| `HISTORICAL` | Useful prior context whose original receipt is not in this tree. |
| `REJECTED` / `FALSIFIED` | A proposed interpretation failed its own acceptance rule. |

Old modeled numbers are never silently promoted into measured claims, raw
receipts and model payloads stay out of the tree, and a negative result is
research output when it exposes a real mechanism. The governing rule:

**Do not reject a moonshot because it sounds impossible. Do not accept it
because it sounds exciting. Try to kill it.**

## AI-assisted development

This project is heavily AI-assisted, and the repository does not present its
implementation as hand-written. The human role is primarily choosing research
questions, generating and selecting hypotheses, setting experimental
direction, evaluating results, deciding what to falsify, operating the
hardware, and making final research decisions. AI agents provide much of the
implementation and experimental labour.

That is a statement about how the work gets done, not a claim about its value.
The evidence is expected to stand on measurements, provenance, and
reproduction — not on who or what wrote the code.

## Hardware profile

Research here is deliberately hardware-specific rather than portable. The
reference GPU is a Sapphire NITRO+ Radeon RX 9060 XT OC 16 GB (RDNA 4 /
`gfx1200`) on RADV, with the host configuration recorded as a phenotype:
factory specifications, live configuration, idle samples, bounded workload
samples, and historical observations are labelled separately in
[`HARDWARE_PROFILE.md`](HARDWARE_PROFILE.md) and
[`hardware_profile.json`](hardware_profile.json).

Do not assume a result reported here reproduces on NVIDIA, Intel, another AMD
architecture, or even another RX 9060 XT without retuning. A cross-hardware
result is a separate experiment with its own device identity, driver probe,
memory state, and correctness evidence.

## Build / reproduce / audit

From the repository root:

```text
cargo build --workspace --release
cargo test --workspace --locked
cargo run -p publication-audit -- .
cargo run --locked -p repro-audit -- .
```

Model-free reproduction lanes under [`repro/`](repro/) define their own
commands and receipts; [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) and
[`REPRODUCIBILITY_STATUS.md`](REPRODUCIBILITY_STATUS.md) describe the status
vocabulary and the current lane dispositions.

For HAR-specific checks, run the Rust-only source gate and the native runtime
trace described in [har/README.md](har/README.md):

```text
rustc tools/check_rust_only_runtime.rs -o /tmp/har-rust-only-runtime
/tmp/har-rust-only-runtime har
```

The trace requires a caller-supplied model fixture and a machine with the
required Vulkan driver; this repository contains no model payload. Read
[`PROVENANCE.md`](PROVENANCE.md), [`docs/ACKNOWLEDGEMENTS.md`](docs/ACKNOWLEDGEMENTS.md),
and [`THIRD_PARTY.md`](THIRD_PARTY.md) before redistributing derived work.

The closure crosswalk for everything published here is
[`PUBLICATION_COVERAGE_MATRIX.md`](PUBLICATION_COVERAGE_MATRIX.md), with the
artifact-level map in [`TECHNICAL_ARTIFACT_INDEX.md`](TECHNICAL_ARTIFACT_INDEX.md)
and the corpus state in [`RESEARCH_CORPUS_STATUS.md`](RESEARCH_CORPUS_STATUS.md).
