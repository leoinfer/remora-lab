# Falsified and bounded results

This record prevents attractive but unsupported numbers from becoming folklore.

## SWMMAC multi-POPS

The proposed multi-POPS result is invalidated. The acceptance contract did not
establish a valid end-to-end measurement, and the accounting was not strong
enough to support the headline. Stronger known-answer tests found that
repeated accumulator behavior caused the benchmark to overcount useful
committed work. The detailed record is the
[`gfx1200 sparse-matrix anomaly`](../../research/falsified/GFX1200_SPARSE_MATRIX_ANOMALY.md).
The failed result remains useful because it exposed instruction-level behavior
and improved the benchmark methodology.

## Faster-than-baseline claims

Several local experiments optimized a narrow kernel or a small resident slice.
That does not establish faster generation. Historical observations include
paths that trailed llama.cpp by several tokens per second. The public project
therefore makes no speedup claim until a new receipt contains the model
identity, prompt, warm-up policy, token count, hardware context, command,
binary identity, and raw output.

## Effective context

The “10M” target is an effective-context hypothesis based on storage,
compression, retrieval, and quality budgeting. It is not a run of dense
attention over ten million positions. Any future claim must report the active
representation, exact quality gate, recovery behavior, and the fraction of
context that was actually attended.

## Flash-Next / R4F

The bring-up work established enough interfaces to continue investigation but
not enough to claim full-model generation. Missing evidence includes complete
weight coverage, numerically checked recurrent state transitions, recovery
after rejected speculative work, and a public end-to-end receipt.

## MTP and expert residency

Acceptance telemetry and expert-read accounting are useful instrumentation,
not speedup proofs. A lower read count can coexist with worse latency,
contention, or quality. Future benchmarks must report all of those dimensions.

## Alice campaign, 2026-09-19 → 2026-09-22

The second model campaign produced a large set of measured rejections, all
retained:

| Direction | What was measured | Disposition |
| --- | --- | --- |
| FreeToken-inspired copy-stream overlap | 515.858 pp/s control vs 507.085 pp/s candidate at ubatch 512; the synchronization drains it targeted priced out at ~0.2% of a prefill pass | **Null** |
| CPU-MoE prefill path | 581.3 → 87.1 t/s at `pp4096` | **−85.1%, retired** |
| Coarse double staging (`n_copies = 2`) | Allocation failure at 49,326.56 MiB (~48.2 GiB) | **Memory-infeasible** |
| `ALICE_MOE_BLOCK` | +6.2%/+17.8% adjacent pairs inside a 1.64× control spread | **Reverted**; the 2.7× projection is retired |
| 100k raw decode, ordinary execution | 676.9 op-TOPS required against a 590–630 band; 18.0–20.25 GiB against an 11.38 GiB budget | **Closed** |
| Structured sparsity as that lever | 563–637 credited op-TOPS against 821; loses 1.56× to eligible dense 2-bit | **Closed** |
| Laya / System-1 decision sidecar | 38.4% agreement against an 84.9% constant baseline | **Negative**; sidecar removed |
| SSD action-memory 5.10× headline | Contaminated by a documented serving-path degradation in the same window | **Retracted**; admissible clean multiplier 1.771× |

Two further items are corrections rather than rejections: an early projection
that a fully host-resident 262K `q8_0` KV cache is "not viable at any MTP
width" was **retracted** after a 16× layer overcount was found, and an apparent
41 GB/s host read above the raw link rate was **falsified** as overlapping
KV regions caused by incorrect head/batch strides.

Full record:
[`research/falsified/ALICE_CAMPAIGN_NEGATIVES.md`](../../research/falsified/ALICE_CAMPAIGN_NEGATIVES.md).
