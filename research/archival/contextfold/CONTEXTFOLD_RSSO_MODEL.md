# Context-RSSO / block-stationary model

For a cold block `b`, let `F_b` be exposed movement plus decompression, `C_b`
attention compute for one query, and `O_b(J)` scheduling/accumulator overhead.

```
T_sequential,b = J F_b + J C_b
T_stationary,b = F_b + J C_b + O_b(J)
stationary wins iff (J-1)F_b > O_b(J)
```

The CPU simulator sweeps `J={1,2,3,4,6,8,16}`, block size, RAM/PCIe bandwidth,
and speculative acceptance.  It uses Qwen F16 geometry and prior project
bandwidth assumptions (RAM 6.9/12.6 GB/s, PCIe 6.4/12 GB/s), not live I/O.

## Current runtime audit

The source already constructs batches with multiple query rows and Flash
Attention kernels that loop over resident KV.  That is physical reuse for
resident cache.  It is incorrect to count that as a new RSSO gain.  The
simulator therefore reports both:

1. algebraic break-even against separately submitted queries; and
2. improvement over a baseline that already batches the rows.

The second count is zero in the default RSSO verdict because the modeled
stationary path has additional overhead and no new cold-transfer capability in
the existing source.  A future cold path could pass only if it demonstrates a
transfer avoided beyond current batching, with exact output/state parity.

## Accumulator memory

For `J` queries, each full-attention layer/head needs `m`, `l`, and an output
vector.  A conservative FP32 accumulator estimate is

```
J × attention_layers × query_heads × (head_dim + 2) × 4 bytes
```

plus one layer's staged block and metadata.  The memory calculator reports this
separately from full KV storage.  Recurrent state snapshots are additional.
