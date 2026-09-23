# Research status

**Updated:** 2026-09-22. This page is the current-state summary. Dated results
remain in the [Flash-Next research log](research/flash-next/CURRENT_RESEARCH_LOG.md),
and claim-level wording is governed by [CLAIMS.md](CLAIMS.md). The Alice,
host-KV, and Qwen3.8-27B records live in [`research/alice/`](research/alice/),
[`research/host-kv/`](research/host-kv/), and [`research/qwen27b/`](research/qwen27b/).

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

## New since 2026-09-18: Alice, host-KV, and the ROCm pivot

| Area | Status | What is supported | What is not claimed |
| --- | --- | --- | --- |
| Alice campaign | `MEASURED`, campaign ~3 days old | A second model family brought up end to end: custom hybrid KDA linear-attention + MoE, 48 blocks, 512 routed experts top-10, 262144 context, 79.64 B counted parameters | Any quality result; no calibration-derived quality basis or capability gate is recorded |
| Alice artifact | `MEASURED` | 39,913,721,760 B at 4.003 effective bpw; expert bank 34.7227 GiB (93.41% of bytes) at 3.862 bpw | Final representation, or a quality claim |
| Alice host expert arena | `MEASURED` | Explicit 12 GiB arena over a packed `O_DIRECT` store: 4.49 → 13.99 → 14.93 → 15.58 t/s across configurations, 95.308% hit, destructive parity `max|diff| = 0.0` | Generalization to other traces or configurations |
| Alice MTP correctness | `MEASURED` | KAT 276/682 failing → 0/682 after fixing an Alice-local conv snapshot plane convention; greedy parity at K = 0/2/3/4 | Any MTP multiplier; the K=1 accepted figure is historical and not reproducible |
| Alice backend records | `HISTORICAL` (Vulkan) + `MEASURED` (HIP) | Historical Vulkan config of record 18.430 t/s with an 18.360 re-anchor; current clean ROCm/HIP raw K0 15.374–15.422 t/s | That 18.430 is current HIP performance |
| Alice prefill | `HISTORICAL`, state-sensitive | 662.7437 t/s hot run at `pp2048`/ubatch 4096, with 245.614 warm and 91.345 cold in the same arm; ub512 band 486.9–533.0 t/s | A stable baseline, or the cross-session 9.3× ratio |
| Backend policy | directive | ROCm/HIP as production and performance backend; Vulkan as parity oracle, debug path, and mechanism donor | A retroactive change to historical Vulkan numbers |
| Host-KV block reuse | `MEASURED` | Physical host read at the ~14 GB/s link roof with logical KV service to 244.944–246.89 GB/s; exact-attention parity rel_rms 4.5e-6, max_abs ≤ 2.4e-7 | Physical PCIe bandwidth, llama.cpp integration, or end-to-end tokens/s |
| Registered host memory | `MEASURED` (prerequisite) | `hipHostRegister`/`hipHostMalloc` is required for direct host-KV kernel reads; unregistered pageable memory faults the GPU | A tuning recommendation — it is a correctness prerequisite |
| Production huge context | `HISTORICAL` | 262144 with host-RAM KV at 8.4–8.7 t/s versus 114688 with VRAM KV at 18.53 t/s; pinning and async staging both null | Long-context quality, or a 384K result |
| Qwen3.8-27B ROCm ladder | `MEASURED` / `HISTORICAL` | Raw K0 17.35 t/s; accepted 29.19 / 34.25 / 33.90 at K=1/2/4; K=2 is the HIP optimum | Merging the historical 42.11 (different artifact, backend, session) with 34.25 |
| KV format coverage on HIP | `MEASURED` (defect) | `q4_0` KV works on the HIP flash-attention path; `iq4_nl` KV has no HIP FA kernel and silently falls back to the CPU (~3.9× slower prefill) before faulting | That KV format choice is free |
| Expert-major grouping | `EXPERIMENTAL` | Single-layer local GEMM 0.185 → 6.115 TMAC/s with bit-exact same-kernel parity; route-materialisation rewrite removing 19.3%/29.5% of the step | A full-model multiplier; Amdahl ceiling ~1.65× and the deployed gate returned `REVERT` |
| CPU-MoE prefill | `INVALIDATED` | Measured −85.1% and retired | Revisiting without a fundamentally different mechanism |
| SSD action memory | `EXPERIMENTAL` | Frozen-panel hit stratum 84.508 → 16.574 s with 20/20 solved, miss stratum 0.9981×, packed extents cutting physical read bytes 72.1% | The retracted 5.10× headline; state reuse has no identity gate |
| Representation utilization | `MEASURED` + `MODELED` | Measured unpack ordering 297.6 → 229.7 → 188.8 GB/s at a real expert shape; modelled per-weight ALU costs; GSQ/RCO prior art | Any measured `pc4`, GSQ, RCO, or combined result; the Flash-Next compression target is modelled |
| New negative results | `INVALIDATED` / `RETRACTED` | F2 null, CPU-MoE regression, coarse double-staging OOM, `ALICE_MOE_BLOCK` revert, 100k raw decode closed, structured sparsity closed, Laya sidecar negative | That a rejection of one implementation closes the mechanism |

## Runtime boundary

| Area | Status | What is supported | What is not claimed |
| --- | --- | --- | --- |
| Research runtime (llama.cpp-derived branches) | `EXPERIMENTAL` | Executes the current model experiments and measurement tooling | That it is part of HAR, or that HAR depends on it |
| HAR native runtime and control plane | `EXPERIMENTAL` | Rust host/runtime code, Rust Vulkan resource and dispatch layer, storage/package contracts, scheduling, residency accounting | Full-model Flash-Next generation (native gate not closed) |
| HAR Rust-only release gates | `VERIFIED FOR THIS CANDIDATE` | Policy gates, dependency metadata, Rust-only source gate | Production serving coverage |
| Consolidated RDNA4/ROCm runtime fork | published | Public fork of llama.cpp (`leoinfer/llama.cpp`, branch `rdna4-rocm-2026-09-22`): `alice_ai` architecture, the recurrent-snapshot correctness fix with its model-free KAT, host expert tier and arena, readback batching, MIX34; builds ROCm/HIP and Vulkan from one tree | That it is a proposed upstream change, or that its defaults are production-tuned |

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
