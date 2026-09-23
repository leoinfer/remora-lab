# Qwen3.8-27B ROCm side campaign

Alice is the main project; this is the public/reference campaign that runs the
same ROCm/HIP work on a model whose geometry is small enough to reason about
exactly. It is where the backend defects, the MTP economics, and the wide-M
verification bandwidth surface were first measured.

**Reporting rule for this file:** raw `K0` decode, MTP-**accepted** decode, and
prefill are never blended into a single tokens/s figure. Every row states its
quality class, its KV type, and whether it is a quiet or loaded session.

## Artifact and geometry

| Field | Value | Class |
| --- | --- | --- |
| Artifact | `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` | `MEASURED` |
| File bytes | 12,120,016,960 B (3.549 bpw) | `MEASURED` |
| Parameters | 27,320,697,856 | `MEASURED` |
| Weights per `K0` token | 11,353,307,136 B (closed tensor-offset ledger) | `DERIVED` |
| Layers | 64 = 16 full-attention + 48 linear-attention | `MEASURED` |
| Attention | 24 Q heads / 4 KV heads (GQA 6), head_dim 256 | `MEASURED` |
| KV cost per token | 65,536 B f16 / 34,816 B `q8_0` / 18,432 B `q4_0` | `DERIVED` |
| Max context | 262144 | `MEASURED` |

## The HIP ladder (`MEASURED`, quiet session)

Single sequence, `np 1`, greedy (`temp 0`, `top_k 1`), 18-token canary prompt,
context 32768 with `q4_0` KV, all layers offloaded, GTT peak < 1 GiB.

| Stage | Raw `K0` t/s | Accepted t/s | Acceptance | Prefill pp512 / pp2048 |
| --- | --- | --- | --- | --- |
| ROCm/HIP `K0` | **17.35** | — | — | 484.9 / 576.4 |
| ROCm/HIP `K1` | — | 29.19 | 1.0000 | 467.2 / 542.6 |
| ROCm/HIP `K2` | — | **34.25** | 0.9286 | 445.7 / 542.0 |
| ROCm/HIP `K4` | — | 33.90 | 0.7500 | — |

Acceptance is exact speculative decoding with the model's own MTP block,
verified by the target: acceptance is reported, not assumed. **`K = 2` is the
measured HIP optimum** — `K = 4` costs more verification for slightly less
committed throughput, and the acceptance rate already tells the story (0.929 →
0.750).

The same configuration under ambient load moves to `K0` 16.41, `K2` 26.03
(acceptance 0.8229), `K4` 30.10 (0.6454). That ~32 % spread is the machine, not
the model; only same-session comparisons are treated as authoritative here.

## The historical `42 t/s` is a different artifact, backend, and session

There is a committed historical receipt of **42.11 t/s accepted at `K = 2`**.
It belongs to a different artifact (`UD-Q3_K_XL`, 13.1 GB), a different
backend (Vulkan), a different build, and context 4096. A ladder model on that
path predicts 42.45 t/s (`MODELED`), and a third artifact in the same campaign
measured 35.82 / 42.77 / 40.06.

**Do not merge 34.25 and 42.11 into one "current performance" claim.** They are
different rows on the same reference machine, and the 34.25 row is the one that
belongs to the GSQ-RCO artifact this campaign is about.

## Wider verification is not free

MTP economics degrade with verifier width, and the reason is measured rather
than assumed: the effective bandwidth of the wide-M pass falls from **197 →
184 → 164 → 129 GB/s** as M goes 1 → 5. At `K = 2` the round needs ~283 GB/s
effective through the round, which is 78 % of what this card demonstrably
delivers (354 GB/s demonstrated, 364–374 GB/s capability class); nothing
measured exceeds 154 GB/s at the widths `K >= 11` would require. So the
accepted-token targets are limited by wide-M GEMM efficiency, not by the memory
system.

| Target | Status |
| --- | --- |
| ≥ 50 raw t/s | **closed by bandwidth** — 11.353 GB/token ÷ 50 t/s = 227 GB/s at *perfect* efficiency |
| ≥ 60 accepted t/s | open, not bandwidth-closed; blocked by wide-M efficiency |
| ≥ 100 accepted t/s | blocked by wide-M efficiency (~280 GB/s needed at M≈11) |
| ≥ 100 raw t/s | physically closed — would need ≤ 3.2 GB/token, i.e. sub-bit-per-weight over 27.3 B parameters |

## KV quantization on HIP

| KV type | Result | Class |
| --- | --- | --- |
| `q4_0` | works on the HIP flash-attention path (`fattn-mma-f16`); all ladder rows above use it | `MEASURED` |
| `iq4_nl` | **no HIP flash-attention kernel exists for it**; attention silently falls back to the CPU — 457-token prefill 3717.5 ms (122.9 t/s) against 942.4 ms for `q4_0`, i.e. **~3.9× slower prefill** — and the cross-backend bridge then faulted on the GPU | `MEASURED` (defect) |

The failure mode is the important part: the fallback is silent. A KV type
without a matching HIP FA kernel does not error out, it quietly moves attention
to the CPU and degrades the run before it faults.

## Roofline anchors used by this campaign

| Anchor | Value | Boundary |
| --- | --- | --- |
| Demonstrated by this machine | 354 GB/s (97 % of the 364 figure) | `MEASURED` — a Vulkan draft step streams 1.225 GB in 3.43 ms inside a real graph |
| Owner-reported card capability | 364–374 GB/s | `REPORTED` — supplied as the hardware's capability, **not** measured here |
| Other measured anchors in the corpus | 364.4 and 317.8 GB/s | `MEASURED`, conditions-dependent |
| `K0` achieved on ROCm (quiet) | 197 GB/s (54 % of the 364 figure) | the actionable gap |

A widely repeated "355 GB/s" figure has **no receipt** in the corpus and is not
used here. Nothing on this page treats the capability figure as a measurement,
and no physical roof is derived from the DPM clock marker, which never selects
its top level under load yet still delivers 354 GB/s.

The 54 % vs 97 % difference is the campaign's headline deficit and it is
isolated to the ROCm `K0` path, not to the memory system.

## Backend status

The HIP path exists, is correctness-clean at the canary level (output
byte-identical to the Vulkan path), and measured three ROCm-specific defects
with root causes. There is **no ROCm production start command** for this model
family yet: production runs Vulkan, and HIP remains the development and
measurement backend — the inverse of the policy adopted for Alice, where HIP is
primary.

## Receipts and boundaries

Receipts behind this file are campaign-local (bring-up report, frontier table,
per-arm JSON receipts, provenance record, harness scripts, plus the recovered
historical Vulkan receipts) and are not part of this repository. They are
summarized with exact values and recorded as a bounded disposition lane in
[`repro/qwen27b/rocm-ladder-2026-09-22/`](../../repro/qwen27b/rocm-ladder-2026-09-22/).
Model weights and the executing runtime trees are excluded from publication.
