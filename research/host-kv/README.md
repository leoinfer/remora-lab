# Host-KV and huge-context work

**The strongest recent mechanism result in this program:** a slow physical tier
can serve far more than its own bandwidth when every fetched byte is reused
enough times.

```text
physical host read bandwidth stays at the link roof
logical KV service scales with how many query rows reuse each fetched block
```

**Never read the logical figure as physical bandwidth.** A logical 244.94 GB/s
on a link that physically delivers ~14 GB/s is block reuse, not a faster wire.
Physical PCIe 4.0 x16 on this machine is nominally 31.5 GB/s raw, and the
measured host→device roof is ~12.8–14 GB/s.

## The mechanism

```text
fetch one host KV block once
  -> stage it (shared memory / LDS)
  -> serve K query rows from that single fetch
  -> accumulate with online/streaming softmax
```

Two independent implementations prove it. They agree on the shape and disagree
on the absolute numbers, which is itself informative.

### ROCm/HIP kernel (`MEASURED`)

A from-scratch HIP attention kernel reading a `hipHostRegister`/`hipHostMalloc`
host buffer. Per KV block of 16 tokens: stage raw `q8_0`, dequantize to f32 in
LDS, `__syncthreads`, then each warp serves `ceil(K/8)` resident query rows,
keeping that row's running max, sum, and 256-wide accumulator in registers
across all 4096 blocks. 32 workgroups, one per `(KV head, batch)`, disjoint by
construction. Geometry: `q8_0` KV, 65536 tokens, 4 KV heads, 8 batches, 32
slices, head_dim 256, 1,140,850,688 B physical per pass.

**Pinned arm** (`hipHostMalloc`, the authoritative on-disk receipt):

| K | Physical GB/s | Logical GB/s |
| --- | --- | --- |
| 1 | 13.997 | 13.997 |
| 2 | 14.142 | 28.285 |
| 4 | 14.002 | 56.006 |
| 8 | 14.010 | 112.081 |
| 16 | 12.929 | 206.867 |
| 32 | 5.770 | 184.649 |
| 64 | 3.781 | 242.002 |
| 96 | 2.552 | 244.944 |

**Registered-pageable arm** (the same run family, second arm):

| K | Physical GB/s | Logical GB/s |
| --- | --- | --- |
| 1 | 14.02 | 14.02 |
| 8 | 14.00 | 112.04 |
| 32 | 4.50 | 143.94 |
| 96 | 2.57 | 246.89 |

Two warnings attach to that second table. First, it is **not the pinned arm**:
the arms disagree at `K >= 32` (4.50 vs 5.770 physical), so they must not be
averaged or blended. Second, the JSON receipt that run was supposed to write
(`hostkv_rocm_reuse_v2.json`) is **absent from disk**; the only surviving
artifact is the run log that contains those four pairs. The pinned table above
comes from the authoritative on-disk receipt, and the repeat runs reproduce the
`K <= 8` plateau within ~2 %.

### Vulkan control (`MEASURED`)

The unmodified in-tree `flash_attn.comp` MMQ kernel — packed int8
`dotPacked4x8EXT` dots on `q8_0`, shared-memory staging, subgroup cooperation,
online softmax, tiled QK/V accumulation — with only the K/V buffers placed in
the host-visible heap instead of device-local. Physical read rate stayed **flat
at 12.6–12.9 GB/s all the way to K=96**, with logical bandwidth reaching
**1232.53 GB/s**. This is published as a contradiction with the HIP arm, not
resolved away: the Vulkan kernel holds the physical roof because it pays for
the reuse with packed integer dots, while the HIP prototype pays with f32
arithmetic.

### Why the HIP arm breaks above K=16 (`MEASURED`, not guessed)

The staging-only kernel — identical loads, no attention math — takes **81.0–82.0
ms at every K from 1 to 96**. The full kernel tracks it exactly to `K=8`, then
adds pure arithmetic: +6.8 ms (K=16), +116.6 ms (K=32), +365.5 ms (K=96). That
is the f32 path's ~36 warp-instructions per `(token, row)` for 512 MACs. The
named next engineering item is therefore a **packed int8/f16 dot path in the
ROCm attention kernel** — the difference between "reuse helps to K=8–16" and
"reuse helps to K=96". The lane's own need is already covered: MTP needs
`K = 4..8`, which is inside the link-bound band.

## Numerical parity

| Path | rel_rms | max_abs | Class |
| --- | --- | --- | --- |
| HIP host-KV vs VRAM fp32 reference | 4.511e-6 … 4.955e-6 (K=1…96) | ≤ 2.3935e-7 at every K | `MEASURED` |
| Vulkan partitioned online-softmax vs CPU single-pass over the dequantized cache | 1e-6 (L=1024) … 3e-6 (L=16384) | 0.0 in every cell | `MEASURED` |

The HIP comparison deliberately uses a reference with a different loop order
and a different softmax formulation, so the residual is operation order, not
an algorithmic shortcut. For scale, the `q8_0` data's own quantization rel_rms
is 4.1e-3 — the attention path is three orders of magnitude tighter than the
storage format it reads.

## Registered host memory is a correctness prerequisite

An unregistered pageable host buffer **is not kernel-readable**. The verbatim
failure is:

```text
Memory access fault by GPU node-1 (Agent handle: 0x...)
  on address 0x...  Reason: Page not present or supervisor privilege.
```

Plain `malloc` reports `type = 0, gpu_accessible = 0`; `hipHostRegister` reports
`type = 1, gpu_accessible = 1` and the kernel reads it. Registered pageable then
runs at 12.12–13.96 GB/s physical, i.e. 88–90 % of pinned. So
`hipHostRegister` / `hipHostMalloc` is not a tuning knob on this path — it is
what makes direct host-KV kernel access legal at all.

## Production context results (`MEASURED`)

Same model family, `q8_0` KV, ROCm/HIP, `t = 8` (16 threads on this 8-core part
is **worse**: 4.7–5.7 t/s):

| Configuration | Context | Decode | Prefill |
| --- | --- | --- | --- |
| KV in VRAM | 114688 | **18.53 t/s** | 532–548 t/s |
| KV in VRAM | 98304 | 18.3–18.5 t/s | 512–548 t/s |
| KV in host RAM | **262144** | **8.4–8.7 t/s** | 306–335 t/s (ub512 best: 328–364) |

KV placement alone costs ~2.2× on decode at matched context, but the full
window cannot be device-resident: the model has 17 attention layers (16 trunk +
1 MTP) at 4 KV heads and 256-wide heads, i.e. 36.1 KB/token at `q8_0`, so
262144 tokens needs ~9.0 GiB — more than fits beside the weights.

Two plausible fixes were measured and are **negatives**: pinning the host KV
(`+0 %`, 8.75 vs 8.4–8.7) and async host staging (`0 %`, 8.39 vs 8.7, with the
flag proven 100 % engaged at `sync=0.281 MB / async=192.565 MB`). A Vulkan
control on the same artifact and flags at 262144 measured prefill 229.3 t/s and
decode **5.38 t/s** — on the host-KV path at full context, ROCm is *faster*
than Vulkan, which is the opposite of the VRAM-KV folklore. Together these kill
the "port the Vulkan reuse kernel and the speed comes back" plan: the reuse
kernel is a prefill/verification-rows mechanism, and single-stream decode at
`K = 1` has nothing to reuse.

## Retraction and correction record

- An early projection claimed a fully host-resident 262K `q8_0` cache was
  "not viable at any MTP width". That rested on a **16× layer overcount** and
  is **retracted**. Correct totals: 9,126,805,504 B (8.5 GiB) for the 16 trunk
  full-attention layers, 570,425,344 B (544.0 MiB) per full-attention layer,
  and 713.0 ms for one full 16-layer traversal at 12.8 GB/s.
- A prior apparent 41 GB/s "host read" from the Vulkan kernel — above the raw
  link rate — was **false**. Incorrect head/batch strides in the push constants
  made KV heads and batches overlap, so far fewer unique bytes were fetched
  than were counted. With corrected strides, hot and cold converge at
  ~12.97–12.98 GB/s at KV=131072. The earlier 84 GB/s "hot" reading at
  KV=16384 was L2 cache reuse on a 34 MiB working set.
- A "61 ms/token" comparison between the RAM-KV and VRAM-KV shapes was
  **withdrawn as a controlled A/B**: the two arms ran at different contexts
  (2.3× different KV volume) and different warm states.

## What is not proven

- No llama.cpp integration of the reuse mechanism on ROCm; no end-to-end
  tokens/s with reuse enabled.
- No real model tensors, no GQA head sharing (24 Q / 4 KV = 6× untouched), no
  page counter — "each byte is read once" is by construction plus rate
  agreement, not a hardware counter.
- The production 262144/114688 rows are engine-level measurements from a
  campaign ledger; the `w8mix` 8.4–8.7 t/s row in particular exists as prose in
  that ledger rather than as a machine-readable receipt.

## The named gate for integration

The scheduler only accepts the host buffer type for **integrated** devices, so
on a discrete card a host-buft KV cache is never used natively — it is staged
per operation anyway. That is why forcing the host-buffer class measured `+0 %`:
the KV landed in the right buffer class and the accept gate still refused it as
a native buffer. The integration objective is that gate, not the byte rate.

## Relationship to already-published work

This result measures a model that was already published as archival theory:
ContextFold/RSSO described a block-stationary cold tier and stated that no
novelty claim is warranted unless cold movement exceeds what existing batching
already provides. [`research/open-problems/OP-03.md`](../open-problems/OP-03.md)
recorded the analogous wavefront work as `BLOCKED`. The kernel-level reuse
proof above is the measurement that was missing; the engine-level integration
remains open.

Receipts behind this file are campaign-local artifacts (kernel source, run
logs, JSON receipts, prose deliverables) and are not part of this repository.
They are summarized with exact values and recorded as bounded disposition lanes
in [`repro/host-kv/`](../../repro/host-kv/).
