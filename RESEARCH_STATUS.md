# Research status

**Updated:** 2026-09-18. This page is the current-state summary. Dated results
remain in the [Flash-Next research log](research/flash-next/CURRENT_RESEARCH_LOG.md),
and claim-level wording is governed by [CLAIMS.md](CLAIMS.md).

## Current primary target: Qwen3.8 Flash-Next

| Area | Status | What is supported | What is not claimed |
| --- | --- | --- | --- |
| Deployment coherence | `MEASURED` | Coherent end-to-end generation on the reference RX 9060 XT; 12/12 deployment canaries passed; two sustained 128-token generation tests passed | Final model quality, or quality parity with BF16 |
| Long context | `MEASURED` | 262,144-token context with the Q8 KV configuration | A 384K daily-driver result |
| Deployment memory profile | `MEASURED` | Scaffold VRAM ~4.42–4.47 GiB, GTT ~0.11–0.13 GiB, RSS ~11.6–12.6 GiB, ~35.13 GiB page-backed host-tier expert bank | Resident full-model throughput |
| Route-aware prefetch | `MEASURED`, single matched pair | 3.02×/5.02× decode, 2.56× shorter wall clock, ~287× fewer major faults, ~2.03× less device read on identical weights | A quantization gain, or a guaranteed multiplier for other configurations |
| Route distribution | `MEASURED` | Top ~42% of expert units carry ~90.2% of routed traffic on the current trace | A stable invariant across prompts or model revisions |
| V2 BF16-derived hot region | `EXPERIMENTAL` | High-value region rebuilt from BF16 (gate/up Q4_K, down Q4_0); remainder temporarily on the Q2 scaffold | Final ancestry, or a performance result |
| Streamed V2 configuration | `MEASURED`, rejected as architecture | ~1.34 GB/token at ~0.8–1.0 t/s | The final architecture |
| Representation panels | `MEASURED`, bounded | T1, LS-reencoded T1, T1+T2, Q2, and 2-bit residual codebook comparisons on a measured panel | Full BF16 capability retention |
| High-precision islands | `MEASURED`, bounded | T1 + Q8 island recovered local error from ~0.779 to ~0.0032 on one expert | An efficient final representation |
| Final V2 bank | `UNMEASURED` | Migration in progress from BF16 | Any V2 quality or speed number |
| MTP real acceptance | `UNMEASURED` / `BLOCKED` | 31-tensor source inventory and bounded graph/transaction fixture | Real acceptance, speed multiplier, or production integration |
| 80 / 250 tokens/s | `HYPOTHESIS` | Explicit falsifiable research targets | A benchmark, forecast, or promise |

The current deployment runs a **temporary Q2 scaffold/control**. The Q2 donor is
a known-good runtime control, BF16 is the quality authority, and the final V2
bank is being regenerated directly from BF16.

## Runtime boundary

| Area | Status | What is supported | What is not claimed |
| --- | --- | --- | --- |
| Research runtime (llama.cpp-derived branches) | `EXPERIMENTAL` | Executes the current model experiments and measurement tooling | That it is part of HAR, or that HAR depends on it |
| HAR native runtime and control plane | `EXPERIMENTAL` | Rust host/runtime code, Rust Vulkan resource and dispatch layer, storage/package contracts, scheduling, residency accounting | Full-model Flash-Next generation (native gate not closed) |
| HAR Rust-only release gates | `VERIFIED FOR THIS CANDIDATE` | Policy gates, dependency metadata, Rust-only source gate | Production serving coverage |

HAR is a long-running native-Rust runtime research track inside this program,
not the runtime most current model experiments execute on. See
[README.md](README.md#har-side-lane) and
[`research/systems/HAR.md`](research/systems/HAR.md).

## Research formats and systems

| Area | Status | What is supported | What is not claimed |
| --- | --- | --- | --- |
| R4X | research format | D32A geometry and parser checks | Stable ecosystem compatibility |
| R4F | bring-up | architecture notes and admission vocabulary | Complete native Flash-Next implementation |
| R4KV | research codec | Rust profiles, pages, capture validation | Frozen public wire compatibility |
| Effective context | hypothesis/experiment | accounting models and explicit caveats | Dense attention at 10M tokens |
| MTP/speculation | experimental | scheduler and acceptance contracts | Universal speedup |
| Expert residency | experimental | Rust page/residency accounting | Full-model GPU residency proof |
| Laguna/HAR-X | historical input | sanitized inventory and open questions | Source redistribution before license review |

Performance claims are tracked separately in [CLAIMS.md](CLAIMS.md). A result
without a public receipt remains historical, experimental, or unmeasured — and
a result measured in one configuration is never promoted into a broader
capability claim.
