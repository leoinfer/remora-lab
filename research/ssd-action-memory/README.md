# SSD action memory

The slow tier is not only a place to put weights. It can also hold **symbolic
action memory**: recorded tool-action extents that let a worker replay a known
procedure instead of generating it token by token. This file records what that
mechanism actually measured, and one headline it did not survive.

## The mechanism

```text
observe a task that resolves through a known action sequence
  -> store the action extent (packed) on the cold tier
  -> on a matching task, fetch and replay the extent
  -> skip the neural tokens and forward passes that would regenerate it
```

The path is real and measurable. What follows separates three different
measurements that have been quoted interchangeably.

## Isolated timing-proxy A/B (`MEASURED`, not model inference)

A deterministic proxy replay measured **30 → 8 neural tokens** and **12 → 4
forward passes** for a task served from action memory. This is a
timing-proxy bookkeeping result: it counts the work the path removes. It is not
model inference and must not be reported as a wall-clock speedup.

## Real-model panel A/B (`MEASURED`)

A frozen 22-task panel run against the real model, one A/B per task (baseline
row, then enabled row), on a frozen server configuration, produced 20 usable
baseline tasks:

| Stratum | n | Baseline mean wall | Enabled mean wall | Ratio | Solved |
| --- | --- | --- | --- | --- | --- |
| Memory **hit** | 12 | 84.508 s | 16.574 s | 5.0995 | 12/12 |
| Memory **miss** | 8 | 127.401 s | 127.639 s | 0.9981 | 8/8 |

Accuracy was preserved in the tested cases (20/20 tasks solved on both arms),
and the miss path is ~1.00× — i.e. a miss costs essentially nothing, which is
the property that makes the path safe to enable unconditionally.

## The headline retraction: 5.10× is not admissible

A subsequent audit of this measurement **rejected the 5.1× headline as
contaminated**. The panel ran inside a window that also contains a documented
serving-path degradation, and the tokens/forwards figures quoted alongside the
headline come from the timing proxy above rather than from model inference.
The audit's admissible clean multiplier for the same mechanism is **1.771×**.

The honest summary is therefore:

```text
mechanism: real, and correctness is preserved at the task level
clean admissible multiplier: 1.771x
5.10x panel headline: RETRACTED (contaminated measurement window)
30 -> 8 tokens / 12 -> 4 forwards: a timing proxy, not model inference
```

A later clean single-task A/B supports the mechanism without the contamination:
**22.158 s → 12.511 s**, with 1 forward pass and 64 tokens removed and 649 SSD
bytes read.

## Packed action extents

Storage cost is where this path becomes interesting on a cold tier, because the
extents are highly compressible:

| Measurement | Value |
| --- | --- |
| Instrumented physical read bytes, packed V2 vs V1 | 62.25 MB → **17.40 MB** (−72.1 %) |
| Action-extent compression trial | 565 B JSON → 73 B (zlib, 0.129) |

So the same action memory occupies roughly an eighth of the physical I/O once
the extents are packed, which is what makes replay cheap enough to win on a
storage-bound configuration.

## Reuse classes and the gate that is missing

The audit priced reuse classes rather than assuming one multiplier: the
workload-conditioned class measures **1.354×**, and the per-class figures differ
enough that a single global number would be misleading.

One class is explicitly **not claimable today**: *state reuse* — reusing
recurrent/conv state across turns — has no identity gate in the runtime, so
there is currently no way to prove that a reused state belongs to the context
it is being reused for. That is a correctness prerequisite, not a tuning item,
and it is recorded as an open gate rather than a result.

## What is not claimed

- No multiplier applies outside the frozen panel and the measured classes.
- No accuracy claim beyond the tested tasks (20/20 on the panel; not a
  general quality result).
- No neural-decode throughput claim from the timing proxy.
- No state-reuse number until the identity gate exists.

Receipts behind this file are campaign-local JSON and prose artifacts and are
not part of this repository; they are summarized with exact values and recorded
as a bounded disposition lane in
[`repro/ngram/action-memory-2026-09-22/`](../../repro/ngram/action-memory-2026-09-22/).
