# MoE residency research

Where should each expert live, and at what precision, on a machine that cannot
hold the model? This family covers expert placement policy, route-aware
prefetch, cache tiers, and the difference between fewer reads and faster
generation.

## Three independent decisions

Most MoE-offload discussion collapses into one knob. The working position here
is that there are three, and they are decided by different evidence:

```text
SENSITIVITY decides precision.
FREQUENCY / REUSE decides residency.
FUTURE-STATE INFORMATION decides movement.
```

A rare expert is not a low-quality expert. An expert that is cheap to store is
not automatically the right one to leave in the slow tier. The target state
space is deliberately two-dimensional:

| State | Sensitive | Tolerant |
| --- | --- | --- |
| Hot | high fidelity + VRAM | compact + VRAM |
| Warm | high fidelity + RAM | compact + RAM |
| Cold | potentially high fidelity + NVMe | compact + NVMe |

The goal is **not** to maximise SSD usage. The goal is to let NVMe hold a large
*population* tail while the hardware serves a small *traffic* tail, which frees
VRAM and RAM for experts that are actually used.

## Route distribution (`MEASURED`)

The trace set covers 1,542 decode steps across 6 workload classes over 48
layers, 512 experts, top-10 routing: 740,160 routed selections across 24,576
`(layer, expert)` units.

| Quantity | Value |
| --- | --- |
| Hottest 42% of units | 10,321 of 24,576 |
| Routed access mass they carry | 90.2% |
| Units needed for exactly 90% of mass | 10,241 (41.67%) |
| Per-layer top-64 share | 36.8% – 70.0% (median 53.6%) |

Skewed, but not extreme: the useful framing is a large population tail carrying
little traffic, not a handful of hot experts.

## Route-aware prefetch (`MEASURED`)

Prefetch issues `POSIX_FADVISE(WILLNEED)` over the **previous token's route
identities** per `(layer, role, expert)` triple, so the pages a demand fault
would have taken become minor faults instead of synchronous major faults. On a
matched control/prefetch pair over identical weights, decode moved from
1.28/0.81 t/s to 3.86/4.07 t/s, wall clock from 325.4 s to 127.0 s, and major
faults from 3,931,682 to 13,714.

Full caveat list: [`research/flash-next/PREFETCH_RESULT.md`](../flash-next/PREFETCH_RESULT.md).
The important scoping point is that this is a storage-bound result. An earlier
configuration found prediction-sourced prefetch net-negative while the feed was
the bottleneck; both results are retained with their regimes.

## Persistent caching is the intended architecture (`HYPOTHESIS` / `MODELED`)

```text
hot:  ready route-aligned expert slots in device-local memory
warm: bounded host-memory triplet cache with identity and integrity checks
cold: canonical expert payload in storage, fetched only through admission
```

plus an optional one-shot bypass staging path for a cold expert that does not
deserve VRAM promotion. Normal-token execution should eventually be mostly cache
hits, so that storage is not the normal expert-compute path.

Evidence for why this matters comes from a rejection: streaming the BF16-derived
V2 hot bank through the slow tier measured 0.73–1.29 t/s decode at
0.229–1.667 GB/token of device-normalized reads, against a modeled
1,262 MB/token for the 42% mask. Moving a larger, higher-fidelity region through
storage made decode slower, not faster.

A related residency analysis expresses the same constraint as a cold-streaming
ceiling of about 2.29 committed tokens/s and concludes that the only lever above
2× is raising the resident fraction of the bank. A projected 7–9 committed
tokens/s for a 10–16 GiB resident arena is `MODELED`; resident full-model
throughput remains `UNMEASURED`, and the deployed runtime integration of the
cache tiers is not complete.

## Precision coupling

Residency decisions interact with representation decisions. The measured
representation panels show that a 2-bit residual codebook removed roughly 2.3×
more activation error than a dense ternary residual plane for the same added
bytes, and that LS-scale re-encoding of the base is a large free gain at
identical size. That makes "which bytes are worth keeping" a measurable question
rather than a preference. See [`research/representation/`](../representation/).

## Boundaries

- Route distribution is measured on one trace set; it is not claimed as a
  prompt-independent invariant.
- The intended cache hierarchy is designed and modeled; it is not a deployed
  result.
- Historical read-count observations are not presented as throughput results.
- Fewer device reads can coexist with worse latency; every future benchmark must
  report both, plus quality.
