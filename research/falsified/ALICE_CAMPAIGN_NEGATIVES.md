# Alice-campaign negative results (2026-09-19 → 2026-09-22)

Negative knowledge from the Alice campaign and its adjacent lanes. Everything
here is a measured rejection, a retraction, or a closed direction. Each row
names what was believed, what was measured, and what the direction is now.

This file is separate from the older negative records already published in
[`README.md`](README.md), [`V4_NEGATIVE_KNOWLEDGE.md`](V4_NEGATIVE_KNOWLEDGE.md),
[`counterexamples/`](counterexamples/), and
[`GFX1200_SPARSE_MATRIX_ANOMALY.md`](GFX1200_SPARSE_MATRIX_ANOMALY.md), which
cover the V4/WARM/counterexample families and the sparse-throughput anomaly.

## Closed by measurement

| Direction | What was measured | Verdict |
| --- | --- | --- |
| Copy-stream overlap (F2) | Controlled interleaved A/B/A/B/A/B at `ub512`: control hot median 515.858 pp/s vs F2 507.085 pp/s | `NULL` — a 1.7 % deficit, not a win |
| Synchronization drains as the limiter | Per-MoE-layer id readbacks plus device-wide drains instrumented | `REFUTED` — ~0.2 % of a prefill pass |
| CPU-MoE prefill path | Forced prefill MoE onto the CPU: 581.3 → 87.1 t/s at `pp4096` | `REGRESSION` −85.1 %, retired |
| Compact MoE v2 prefill | Prefill ratio 0.164× | `REVERT` |
| Naive coarse double staging | `sched->n_copies = 2` with all copies marked output | `OOM` at 49,326.56 MiB (~48.2 GiB) |
| Targeted second staging slot (F1b) | Second slot restricted to host-MoE expert staging tensors | `FAILED TO LOAD` — two allocator failures, then a segfault before readiness; only control arms produced numbers |
| `ALICE_MOE_BLOCK` | Arms measured, adjacent pairs +6.2 % / +17.8 % against a 1.64× control spread | `REVERT`; the 2.7× projection is retired |
| 100k raw decode under ordinary execution | Physical proof: 676.9 op-TOPS required against a 590–630 band; 18.0–20.25 GiB needed against an 11.38 GiB budget | `CLOSED` |
| Structured sparsity as the 100k lever | 1.90–2.10× on the same logical GEMM, 563–637 credited op-TOPS against 821 | `CLOSED` — 1.06–1.20× short, and the representation loses 1.56× to eligible dense 2-bit |
| Wider MTP verifier | Wide-M effective bandwidth 197 → 184 → 164 → 129 GB/s as M goes 1 → 5; `K=2` costs 1.94× a forward for 2.55× span, `K=4` costs 3.55× for 2.79× | `DIMINISHING` — economics are not free |
| Laya / System-1 decision sidecar | 73 decisions: 38.4 % agreement with the lead against an 84.9 % constant baseline; lead-better 10, Laya-better 1; 1.8–3.0 s per decision; 2.1 GiB RSS | `NEGATIVE` — sidecar removed, receipt retained |

## Why the overlap programme failed, in mechanism terms

Three findings, all instrumented, explain the null and bound any retry:

1. **Synchronization is not the cost.** The per-layer id readbacks and
   device/backend drains that the overlap design was built to hide price out at
   ~0.2 % of a prefill pass. There is almost nothing to overlap against.
2. **Staging destinations alias.** Adjacent host-MoE staging destinations are
   observed to alias because the allocator sees a single logical copy. The next
   layer can therefore overwrite bytes still being consumed, which makes naive
   overlap unsafe rather than merely unhelpful.
3. **Duplication does not fit.** The general fix duplicates every split input
   and OOMs at ~48 GiB on a 32 GiB host. The targeted fix — a second slot for
   host-MoE expert staging tensors only — is the right shape but did not load.

Any future attempt needs the event ordering made explicit (consumer finishes
slot A → event → only then may the copy stream reuse A) and must demonstrate a
candidate arm that both loads and beats its control.

## Why `ALICE_MOE_BLOCK` is retired

The arms were measured rather than reasoned about, and the control spread
(1.64×) is larger than the apparent effect (adjacent pairs of +6.2 % and
+17.8 %). Routing parity held across 384 layer-steps, and several taps were
voided. The 2.7× projection that motivated the lane is **retired**: a
projection is not a measurement, and this one never produced a reproducible arm.

## Retractions inside the campaign

Several claims were withdrawn by their own authors once instruments disagreed:

- "The arena serves the wrong slices" — **retracted**; the arena's destructive
  parity test measures `max|diff| = 0.0`.
- The "15.58 settled rate" — **not reproducible** as a settled figure; the
  arena frontier is 4.49 → 13.99 → 14.93 → 15.58 t/s across configurations, and
  a re-run today gives a materially lower page-cache-dependent result.
- A "phantom CPU-backend overhead", "5.25 submissions per copy", and
  affinity-syscall costs — all **falsified** by instrumentation.
- A dispatch attribution of 91.09 % — **self-retracted** to 26 %.

## Laya sidecar, in full

The Laya/System-1 decision sidecar is worth preserving precisely because it
looked reasonable and failed decisively:

```text
agreement with the lead        38.4%
constant-baseline agreement    84.9%
settled disagreements          lead better 10, sidecar better 1
cost                          1.8-3.0 s per decision, 2.1 GiB RSS
gate                           unreachable by construction
outcome                       model, checkpoint, and dependencies removed
```

A sidecar that agrees with the lead less often than a constant baseline cannot
add value regardless of how it is gated. The model, checkpoint, and dependency
tree were deliberately removed; only this receipt is kept so the path is not
retried blindly.

## How to read this file

A negative result here is not a claim that the underlying mechanism is
impossible — it is a claim about a specific implementation, configuration, and
measurement window. The retained value is the instrument reading and the named
prerequisite for any retry. Where a direction was closed by *physics* (the 100k
raw-decode target, sub-bit-per-weight representations) that is stated; where it
was closed by *this implementation* the distinction is preserved.
