# Utilization-first representation research

The representation family has a second, newer thread beyond the fidelity panels
in [`README.md`](README.md). It starts from a different objective:

```text
maximum fidelity  x  minimum bytes  x  maximum execution speed on this GPU
```

and from an explicit ordering:

```text
utilization first
bpw second
quality is a hard constraint
```

The point of the ordering is that the usual framing — "find the smallest
bits-per-weight that survives a quality gate" — is the wrong optimisation for a
machine whose bottleneck is execution efficiency, not capacity. First make the
representation something the GPU can execute at high utilization; then descend
bits-per-weight **without surrendering that utilization**.

**Evidence class:** this file is a design-and-measurement record. The
per-weight ALU estimates and the projections below are `MODELED`; the unpack
probe is `MEASURED`; the Flash-Next byte budget is `MEASURED`/`DERIVED`. Nothing
here is a measured end-to-end result.

## Smaller formats can be slower

The batch-1 decode path runs MMVQ, where different quant formats have very
different decode and metadata costs. A four-arm probe at a real expert shape
(`gate_up`, K=2048, N=1024, M=128) holding the *output* fixed and varying only
the representation measured:

| Arm | Representation | Effective throughput |
| --- | --- | --- |
| A | int8-resident | **297.6 GB/s** |
| B | packed 4-bit, 4.5 bpw | **229.7 GB/s** |
| C | 4.25 bpw symmetric | **188.8 GB/s** |
| D | 4.25 bpw affine | 169.3 GB/s |

Effective throughput falls as the representation gets harder to unpack, even
though the byte count goes **down**. That is the whole argument for
utilization-first: bytes saved in storage can be paid back with interest in
unpack cost.

Per-weight ALU cost estimates from the same investigation, with their sources
recorded in the design document, are:

| Format | Estimated ALU ops/weight |
| --- | --- |
| `Q6_K` | ~2.75 |
| `Q4_K` | ~2.9 |
| `IQ3_S` | ~3.3 |
| `Q3_K` | ~4.4–5.7 |

These are `MODELED` per-weight estimates, not measurements. Their value is
ordinal: they predict that `Q3_K` is the expensive end of the palette and
`Q4_K`/`Q6_K` the cheap end, which is consistent with the unpack probe above.

## The `pc4` direction

A candidate simple per-channel 4-bit representation is interesting for the
opposite reason to a learned codebook: it has a **simple nibble layout, minimal
metadata, and low decode complexity**, so it should be cheap to execute, while
the *offline* quantization step that chooses the scales can be as sophisticated
as we like.

Its performance is a **projection only**. No measured `pc4` throughput exists,
and it must not be quoted as one.

## Prior art: GSQ and RCO

Two recent papers are adopted as major prior art, and both are relevant for a
different reason:

| Work | What it contributes to this program |
| --- | --- |
| **GSQ** — <https://huggingface.co/papers/2604.18556> | Improve the discrete quantized *values and scales* offline while keeping a relatively simple scalar execution format. This attacks fidelity without changing what the GPU has to unpack. |
| **RCO** — <https://huggingface.co/papers/2605.00649> | Allocate precision globally across tensors and layers under a total budget, instead of forcing a uniform bit width. |

A source-level audit of both (recorded in the campaign corpus) found GSQ to be
a Gumbel-Softmax scalar-grid method at group-128 with the container unchanged,
and RCO to be an exact-budget multiple-choice knapsack allocation. The audit's
most useful finding is that the advantage **decomposes with allocation
dominating representation** — which matches this program's own earlier result
that an exact-budget allocator's choice of *which tensor* matters more than the
choice of codec.

Intended synthesis, **not implemented**:

```text
GSQ-style offline value/scale optimisation
  + RCO-style global bit allocation
  + a low-overhead, native-to-this-GPU execution layout
```

Nothing in this repository claims that synthesis has been built, and no
combined result exists.

## Flash-Next: what GSQ/RCO-style compression would have to mean

**Section class: `MODELED` / future work. No measured compressed result
exists.**

The measured starting point is a byte budget, not an estimate:

| Component | Size | Class |
| --- | --- | --- |
| BF16 body payload | 359,999,963,128 B (≈360.0 GB decimal / 335.28 GiB) | `MEASURED` |
| Expert `gate_up` | 153.125 GiB | `MEASURED` (byte breakdown) |
| n-gram / PLE table | 95.429 GiB | `MEASURED` (byte breakdown) |
| Expert `down` | 76.562 GiB | `MEASURED` (byte breakdown) |
| Neural budget available on this machine | 31.74 GiB | `DERIVED` |
| Cheapest legal compact format evaluated | 25.12 GiB | `DERIVED` |

An earlier "~357 GB" figure circulates for this model; the byte breakdown above
is the authoritative accounting, and a later independent reading of the pass
volume measured **~400 GB**. All three numbers describe the same artifact at
different scopes and are labelled rather than reconciled.

Applying the compression ratios observed on the 27B GSQ/RCO artifact suggests
the neural body could land somewhere in the **~40–56 GB** range depending on the
target bitrate. The honest conclusion from that arithmetic is negative for this
machine:

- a **~3.5 bpw class** neural body remains too large for 16 GB VRAM + 32 GB RAM
  once runtime overhead (KV, graph buffers, staging) is included;
- the practical stretch target is closer to **2.25–2.5 effective bpw** for the
  neural body if a ~36–40 GB package is the goal.

The research question is therefore not "how small can we get", but:

```text
move the quality-vs-bpw curve left
```

with the levers this program has identified: aggressive compression of routed
experts only, higher precision for router / recurrent / attention / MTP / shared
experts, expert- and tensor-specific precision, GSQ-style value optimisation,
RCO-style allocation, and layouts that stay fast on this GPU.

## Quality is not weight-MSE

A local fidelity metric is a proxy, and this program has already been burned by
treating one as capability. The gate list proposed for any future representation
work is:

```text
KL / logits
perplexity
task quality
router top-k behaviour
MTP acceptance
agent / coding behaviour
long-generation stability
```

That list is `PROPOSED`; the panels in [`README.md`](README.md) still carry an
explicit `BLOCKED` teacher-KL disposition, and no candidate in this family has
been through a capability gate.

## Preserved negative

A compact expert format family at 4.25–4.27 bpw was designed, implemented, and
gated — and the verdict is **documented negative**. An earlier attempt in the
2.0–2.25 bpw range was quality-fatal outright. The conclusion recorded in the
campaign corpus is that no further budget should be spent on compact expert
formats as a throughput lever without a different mechanism. It is retained
here so the direction is not re-derived from scratch.

## What is not claimed

- No measured `pc4`, no measured GSQ, no measured RCO, and no measured
  combination of them.
- No compressed Flash-Next result; the 40–56 GB and 2.25–2.5 bpw figures are
  projections from another model's ratios.
- No capability claim from any fidelity metric in this family.
- No end-to-end speedup from the unpack probe, which is a single-shape kernel
  measurement.
