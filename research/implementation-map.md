# Implementation map

This map is the navigation contract for the public-safe research tree. Read
each line from left to right:

`IDEA → SPEC → IMPLEMENTATION → EXPERIMENTS → EVIDENCE → FAILURE CASES → ROADMAP`

An implementation link means that code in this candidate exists and is
covered by the stated boundary. It does not mean that a model, a quality
target, or a performance target has been validated. A research-only item is
kept as a method or question when the implementation source is not cleared.

## Core and format crosswalk

| IDEA | SPEC | IMPLEMENTATION | EXPERIMENTS | EVIDENCE | FAILURE CASES | ROADMAP |
| --- | --- | --- | --- | --- | --- | --- |
| HAR native runtime | [`research/systems/HAR.md`](systems/HAR.md) | `har/crates/har-runtime/src/policy.rs`, `har/crates/har-serve/src/`, `har/crates/har-vulkan/src/lib.rs` | `research/experiments/`, workspace tests | [`CLAIMS.md`](../CLAIMS.md), [`PUBLIC_HAR_RELEASE_AUDIT.md`](../PUBLIC_HAR_RELEASE_AUDIT.md) | fallback rejection and bounded-kernel limits | complete caller-supplied model fixtures and production serving gates |
| R4X | [`formats/r4x/FORMAT.md`](../formats/r4x/FORMAT.md) | bounded geometry/parser representation in `formats/r4x`; package-slice support in `har/crates/har-package-slice/` | `research/r4x/`, `research/experiments/` | `CLAIMS.md` C-004 and format tests | no ecosystem compatibility or full model claim | independent decoder, public vectors, quality and end-to-end tests |
| R4X-D / R4X-H / R4X-S / R4X-D32A | [`research/systems/R4X.md`](systems/R4X.md), [`formats/r4x/FORMAT.md`](../formats/r4x/FORMAT.md) | D32A geometry is public; wider/aggressive tracks remain research-only | `research/r4x/`, falsified-result cards | bounded parser tests and documented accounting | regional quality, packing, and execution results are not interchangeable | clear each variant's provenance, vectors, quality floor, and kernel coverage |
| XP-S | [`research/systems/R4X.md`](systems/R4X.md) | no production implementation claimed; research map only | R4X experiment cards | methodology and negative knowledge | model-specific and quality evidence is omitted | publish a clean-room variant specification only after review |
| regional precision / storage-vs-execution precision | [`research/systems/R4X.md`](systems/R4X.md) | policy concepts are represented in research docs; no hidden runtime fallback | quantization and precision experiment cards | claims are experimental | storage savings can disappear after dequantization, launch, or residency cost | add held-out quality and exposed-byte measurements |
| QAT / 2:4 sparsity / mask learning / kernel-width research | [`research/r4x/README.md`](r4x/README.md) | no production quality or sparsity backend is claimed | `research/falsified/`, `research/experiments/` | methodology and falsified anomaly note | masks, training artifacts, and kernel receipts are not bundled | publish fixtures and known-answer kernels with provenance |
| R4KV variants, HOT-WARM-COLD, restore, materialization, GPU decoder, dominated candidates, quality | [`formats/r4kv/FORMAT.md`](../formats/r4kv/FORMAT.md), [`research/systems/R4KV.md`](systems/R4KV.md) | `har/crates/r4kv/src/{quant.rs,page.rs,tier.rs,dma.rs,capture.rs,profiles.rs}` | `research/experiments/`, `research/falsified/` | codec tests, profile arithmetic, claims ledger | codec correctness is not model-quality or long-context proof | independent decoder vectors, restore/replay fixtures, and quality gates |
| R4F formats / tensor precision / expert representation / PLE / execution-storage | [`formats/r4f/FORMAT.md`](../formats/r4f/FORMAT.md), [`research/systems/R4F.md`](systems/R4F.md) | public package/compiler/storage boundaries only; current R4F-specific adapter work is pending sanitization | [`research/flash-next/CURRENT_CAMPAIGN.md`](flash-next/CURRENT_CAMPAIGN.md) | bounded campaign seams and explicit gate state | no stable byte format and no full-model generation | promote only clean Rust modules with provenance and public fixtures |

## Research systems and control planes

| IDEA | SPEC | IMPLEMENTATION | EXPERIMENTS | EVIDENCE | FAILURE CASES | ROADMAP |
| --- | --- | --- | --- | --- | --- | --- |
| HERMES / REMORA | [`research/systems/HERMES.md`](systems/HERMES.md), [`research/systems/REMORA.md`](systems/REMORA.md) | `har/crates/har-residency/`, `har/crates/har-metabolism/`, `har/crates/har-lang-compiler/` | HERMES atlas, REMORA manifest, `research/experiments/` | idea index and static research cards | gates remain explicit for live model evidence | formalize interfaces and add replayable public traces |
| OP-01..OP-12 / C-01..C-10 / F0..F15 | [`research/open-problems/`, `research/conjectures/`, `research/theory/checkers/`](.) | checker and contract descriptions; no claim that every checker is live production code | numbered cards and formalization queue | machine-readable index and source register | unresolved proofs, missing ranking, and blocked live traces remain marked | close one checker with a public fixture at a time |
| ContextFold / ContextPack / effective-context / 10M / retrieval | [`research/systems/ContextFold.md`](systems/ContextFold.md), [`research/effective-context/README.md`](effective-context/README.md) | `har/crates/har-contextfold/src/{lib.rs,orchestrator.rs,policy.rs,store.rs}` | effective-context cards and experiment queue | accounting hypotheses and policy tests | effective context is not dense attention at 10M tokens; external numeric codec remains a boundary | public codec vectors, retrieval evaluation, and full memory accounting |
| benchmark weaknesses / exact base / residual KV / key-first / value-late | [`research/systems/ContextFold.md`](systems/ContextFold.md) | policy/store abstractions only | context and falsification cards | methodology notes | no raw model receipts; exactness is not assumed | adversarial retrieval and reconstruction tests |
| Delta-Certified skipping / Context-RSSO / UARC / causal shared-prefix | [`research/conjectures/`, `research/systems/RSSO.md`](.) | `har/crates/har-contextfold/` and `har/crates/har-certificates/` provide bounded contracts | conjecture and checker cards | theorem boundaries are labeled | conjecture is not proof; live traces remain pending | certificate-bearing replay and dependency closure fixtures |
| Reclaim / Refrigerator / Salvage / Waste Ledger / Tiered Reserve | [`research/systems/REMORA.md`](systems/REMORA.md) | `har/crates/har-metabolism/src/{reclaim.rs,reserve.rs,salvage.rs,ledger.rs}` | REMORA cards and static ledgers | Rust unit tests and accounting identities | no claim of autonomous production control | connect policies to bounded replay traces |
| Reserve Mobilization / Moving Maintenance Setpoint / Safe Surplus / TBEH / shadow pricing | [`research/systems/PFM.md`](systems/PFM.md), [`research/systems/REMORA.md`](systems/REMORA.md) | `har/crates/har-metabolism/src/{setpoint.rs,surplus.rs}` and `har/crates/har-execution/src/speculation.rs` | formalization and experiment cards | contracts and static checks | live authority and optimality remain unproven | held-out replay, resource-constrained checker, and policy audit |
| speculative decoding / MTP / elastic horizon / accepted-token roofline | [`research/mtp-speculation/README.md`](mtp-speculation/README.md), [`research/systems/DSpark-MTP.md`](systems/DSpark-MTP.md) | `har/crates/har-execution/src/speculation.rs`, `har/crates/har-residency/src/mtp.rs`, `har/crates/har-decode-control/src/language.rs` | MTP cards and E-series | acceptance/resource contracts | no universal speedup claim; accepted-token traces are required | public trace schema and caller-supplied oracle replay |
| ngram / RSSO / expert-major / REMORA-Spark / acceptance-weighted quantization | [`research/systems/RSSO.md`](systems/RSSO.md), [`research/systems/ExpertPack.md`](systems/ExpertPack.md) | `har/crates/har-residency/` and `har/crates/har-serve/src/moe.rs` for bounded contracts | speculation and residency cards | static scheduler and routing tests | quality and acceptance data are model-dependent | held-out acceptance, quality, and exposed-byte evidence |
| MoE / residency / EER / expert capsules / predictive residency / PFM-A | [`research/systems/ExpertPack.md`](systems/ExpertPack.md), [`research/moe-residency/README.md`](moe-residency/README.md) | `har/crates/har-residency/src/{manager.rs,expert_lru.rs,page_store.rs,scheduler.rs}`, `har/crates/har-serve/src/moe.rs` | residency experiment cards | Rust tests and synthetic routing | synthetic paths are not full-model generation | real caller-supplied expert traces with quality checks |
| HOT-WARM-COLD experts / dual-tri path / repacking / regime detection / resource-complementarity scheduling | [`research/moe-residency/README.md`](moe-residency/README.md), [`research/systems/PFM.md`](systems/PFM.md) | residency and metabolism contracts above | HERMES/REMORA cards and E-series | static resource accounting | predictive and regime claims are not live proof | replayable placement, repacking, and regime-transition fixtures |

## Hardware, benchmark, and failure crosswalk

| IDEA | SPEC | IMPLEMENTATION | EXPERIMENTS | EVIDENCE | FAILURE CASES | ROADMAP |
| --- | --- | --- | --- | --- | --- | --- |
| gfx1200 / RDNA4 / SWMMAC / known-answer | [`HARDWARE_PROFILE.md`](../HARDWARE_PROFILE.md), [`research/falsified/GFX1200_SPARSE_MATRIX_ANOMALY.md`](falsified/GFX1200_SPARSE_MATRIX_ANOMALY.md) | `har/crates/har-vulkan/`, `har/shaders/` | GPU smoke and known-answer cards | bounded Rust/Vulkan evidence | cooperative-matrix/device-fault evidence is retained as a limitation | expand public known-answer vectors and fault recovery tests |
| 2:4 / false multi-POPS / quarter-result / accumulator / all falsified kernel work | [`research/falsified/GFX1200_SPARSE_MATRIX_ANOMALY.md`](falsified/GFX1200_SPARSE_MATRIX_ANOMALY.md), [`research/falsified/README.md`](falsified/README.md) | no accepted production kernel claim | falsification ledger and experiment cards | repeated-accumulator overcount is documented | apparent throughput was invalidated; methodology, not the number, survives | publish independent known-answer harness and committed-work accounting |
| Laguna / HAR-X / hardware phenotype compilation / benchmark / moonshot / falsification | [`research/laguna/README.md`](laguna/README.md), [`research/har-x/README.md`](har-x/README.md), [`HARDWARE_PROFILE.md`](../HARDWARE_PROFILE.md) | `har/crates/har-model-compiler/`, `benchmarks/local-bench/`, `har/crates/har-vulkan/` where applicable | benchmark and phenotype cards | claims ledger, hardware profile, falsified results | license, model, and raw receipt boundaries remain explicit | clean-room compilation examples, public benchmark fixtures, and adversarial replication |

## Current Flash-Next boundary

The current campaign is represented by
[`research/flash-next/CURRENT_CAMPAIGN.md`](flash-next/CURRENT_CAMPAIGN.md).
It records R4F codec/container bring-up, embedding, GDN/recurrent state, QSA
indexing and selected-KV attention, PLE page caching, routed MoE, Q8F routing,
Q4F expert capsules, routed accumulation, and CPU replay/oracle evidence. The
source worktree that produced these seams is dirty and is therefore not
silently promoted into this candidate. The first-token oracle remains the
gate; full-model generation and throughput are not claimed.

## Rust path rule

`har/` is the only production runtime boundary. Research notes and the
reviewed Rust benchmark may describe other systems, but Python, C++,
llama.cpp, GGML, CMake-built components, subprocess helpers, C ABI execution,
and foreign inference backends are not runtime dependencies. GPU shader/SPIR-V
material and Vulkan/OS driver libraries remain the only non-Rust execution
boundaries called out by the release audit.
