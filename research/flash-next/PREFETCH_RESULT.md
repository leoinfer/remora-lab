# Route-aware prefetch: matched control versus prefetch pair

**Evidence class:** `MEASURED` — one matched pair of runs, 2026-09-18.
**Scope:** Qwen3.8 Flash-Next on the reference RX 9060 XT machine, Q2_0 donor
scaffold, host-tier expert region enabled, 128 decode tokens per prompt,
two prompts.
**Not in scope:** quality, quantization gain, kernel throughput, or any claim
about a resident (non-storage-bound) configuration.

This is currently the strongest clean mechanism-level result in the Flash-Next
work, so it is recorded separately from the campaign log. It is also a result
with real caveats; read the caveat list before quoting any number.

## What was compared

Both arms are the same work:

- same model file and identical first-MiB digest;
- same binary (identical build timestamp and path);
- same prompts: one long-explain prompt (34 prompt tokens) and one long-code
  prompt (36 prompt tokens), 128 decode tokens each;
- same flags (`-n 128 -ngl 99 -c 2048`, 6 threads) and same environment;
- same host-tier expert configuration.

The **only** difference is the route-aware prefetch setting: three environment
variables that enable prefetching and name the scaffold file and offset table.
Everything else in the two receipts is byte-identical.

## Result

| Metric | Control | Route-aware prefetch |
| --- | --- | --- |
| Decode, long-explain | 1.28 t/s | 3.86 t/s |
| Decode, long-code | 0.81 t/s | 4.07 t/s |
| p50 token latency, long-explain | 702 ms | 224 ms |
| p50 token latency, long-code | 1189 ms | 233 ms |
| p95 token latency, long-explain | 1634 ms | 475 ms |
| p95 token latency, long-code | 2089 ms | 362 ms |
| Wall clock, whole run | 325.4 s | 127.0 s |
| NVMe bytes read during the run | 151,721,033,728 B | 74,699,780,096 B |
| Major page faults | 3,931,682 | 13,714 |
| Minor page faults | 561,636 | 967,122 |

Derived ratios over those measured rows: decode 3.02× (long-explain) and 5.02×
(long-code); p50 3.13× and 5.10× lower; p95 3.44× and 5.77× lower; wall clock
2.56× shorter; NVMe read volume 2.03× lower; major page faults 286.7× lower.

Consolidated means across the two prompts are 1.045 t/s (control) versus
3.965 t/s (prefetch), a 3.79× ratio. A consolidated normalized view reports
465,401,944 B/token (control) versus 229,140,429 B/token (prefetch) using
machine-wide NVMe counters divided by total tokens, and 13,783.2 versus 3,008.7
page faults per token.

## Interpretation

On identical weights, route-aware prefetch moved this configuration from
roughly 0.8–1.3 tokens/s to roughly 3.9–4.1 tokens/s, cut the wall clock by
about 2.56×, cut major faults by about 287×, and roughly halved device read
volume.

This is a **storage and residency path result**. It is not a quantization
improvement: no weight bytes changed between the two arms. It says that in a
storage-bound scaffold, the cost of demand-faulted expert reads dominates
decode, and that scanning for the pages a route will need before they are
demanded removes most of that cost.

## Mechanism, stated precisely

The implementation issues `posix_fadvise(..., POSIX_FADV_WILLNEED)` over the
**previous token's route identities**, per (layer, role, expert) triple, using
the scaffold slice size that the current Q2_0 representation actually serves.
It relies on route persistence between adjacent tokens to supply the lookahead:
the pages it asks for are the pages the next demand fault would have taken
anyway, so the demand fault becomes a minor fault instead of a synchronous
major fault.

It is therefore **not** a learned future-route predictor, and it does not need
prediction accuracy to be useful — it is warming the page cache for a set of
pages that has a high chance of being reused. The published pair uses the
default per-layer schedule; a deeper per-token burst mode exists in the
implementation but was not enabled for these runs.

## Caveats — read before quoting

1. **Single matched pair, and the arms ran back-to-back.** The prefetch arm
   started within a second of the control arm finishing, so it may have begun
   with a warmer page cache. A counter-balanced repeat was not found in the
   evidence set. Treat the ratios as an observed pair on the reference machine,
   not as a controlled double-blind measurement.
2. **Device read volume is a machine-wide counter.** NVMe bytes come from
   system-wide disk statistics summed across NVMe devices and differenced over
   the run window, so unrelated I/O during a run is attributed to that run.
   The 2.03× reduction is consistent with the run behavior but is not a
   process-attributed I/O accounting.
3. **Page faults are lifecycle-cumulative, polled.** The fault figures are
   absolute counters sampled during the run rather than start/end deltas, so
   they are indicative, not exact per-window deltas.
4. **Prefetch emission counters were not logged.** The implementation tracks
   issued slices and bytes, but those counters never reached the run receipts
   in this configuration. Do not quote "prefetched N GB"; that number does not
   exist in the evidence.
5. **An earlier, different configuration found route-based prediction
   net-negative.** Historical work in this repository measured previous-token
   route prediction at roughly 0.32 precision/recall with 68–85% wasted
   prefetch bytes, and called prediction-sourced prefetch net-negative *while
   the feed was the bottleneck*. That is a different regime: it measured
   prediction accuracy against a different target, whereas this pair measures
   page-cache population for pages the demand path would have read anyway.
   Both results are retained; the scope of each must be stated with it.
6. **This is not a resident-throughput result.** Both arms are storage-bound
   configurations. The result says nothing about decode speed once the working
   set is actually resident in VRAM or RAM.
7. **The source comment and the measurement disagree about byte traffic.** The
   prefetch code asserts that total device traffic is unchanged because the
   same pages are read, only earlier. The measured pair shows device read
   volume *falling* by about half. Those two statements cannot both be literal;
   the plausible reconciliation (synchronous demand reads amplifying traffic, or
   machine-wide counter attribution) is not established here. This record
   publishes the measurement and keeps the disagreement visible rather than
   resolving it by preference.

## Related measurement: storage-path headroom

Separately from the pair above, a native cold-read benchmark at the 460,800-byte
slice size the current Q2_0 scaffold actually serves measured:

| Reader mode | Concurrency 1 | Concurrency 8 |
| --- | --- | --- |
| Cold `pread` | 0.2103 GB/s | 1.1883 GB/s |
| Cold `O_DIRECT` | 0.2781 GB/s | 1.2123 GB/s |

The Python `mmap_fault` rows from the same benchmark are **GIL-bound floors**:
the touch loop copies each block inside Python while holding the interpreter
lock, so its concurrency scaling is a lower bound and not a like-for-like
measurement of the C++ runtime's page-fault path. They must not be used as a
comparison against the runtime.

The observed runtime's effective expert-read traffic in the deployed scaffold
is quoted as roughly 0.11–0.36 GB/s. That band is a **restated estimate**: the
primary run artifact behind it was not recovered, and every occurrence in the
evidence set points back to the deployment note or a benchmark framing remark.
It is published as a stated band, not as a fresh measurement.

Taken together these indicate **storage-path headroom** — the device can serve
this slice size faster than the current runtime consumes it. They do **not**
imply that the 3–5× decode ratio above is automatically available to every
future configuration, and they must not be multiplied together with it as if
they were independent confirmed gains.

## Reproducing this

The lane disposition is at
[`repro/flash-next/deployment-2026-09-18/`](../../repro/flash-next/deployment-2026-09-18/),
with a sanitized receipt containing the measured rows and their provenance
labels. The model payload and the research runtime branch are not part of this
repository, so the lane is a bounded disposition rather than a runnable public
benchmark.

The runtime that produced these numbers is a llama.cpp-derived research
worktree. It is an experiment platform, not part of HAR, and HAR does not
depend on it.
