# Dense GPU roofline provenance

**Evidence status:** historical source context plus derived arithmetic.
**Scope:** AMD Radeon RX 9060 XT / RDNA4 (`gfx1200`) and a dense
approximately 27B-parameter Qwen comparison. This note is not a current
benchmark and does not describe Flash-Next end-to-end throughput.

The archived roofline work was recovered to keep three different quantities
separate:

1. physical memory bandwidth;
2. compute-equivalent parameter bandwidth; and
3. useful generated-token throughput.

The repository's general rule is in [`methodology.md`](methodology.md): a
narrow kernel, synthetic model, reduced byte count, or arithmetic normalization
must not be promoted to an end-to-end generation result.

## Historical arithmetic

AMD's public product figures list approximately 410 INT4 TOPS for dense work
and 821 INT4 TOPS for a compatible structured-sparse mode. Under the explicit
convention `1 MAC = 2 operations`:

```text
410e12 operations/s / 2 = 205e12 MAC/s
821e12 operations/s / 2 = 410.5e12 MAC/s
```

For the dense 410-TOPS line, a six-raw-bits-per-weight normalization gives:

```text
205e12 MAC/s × 6/8 bytes = 153.75e12 bytes/s ≈ 154 TB/s
```

That approximately 154 TB/s value is **compute-equivalent parameter bandwidth**.
It is not physical VRAM bandwidth. The literal four-bit normalization would be
102.5 TB/s. The corresponding structured-sparse four-bit normalization would
be 205.25 TB/s, but only for a compatible sparse representation and execution
path. Quantization alone does not make a model structured sparse.

For an illustrative dense 27B parameter model, one MAC per parameter
application gives:

```text
205e12 MAC/s / 27e9 parameters ≈ 7,593 logical parameter traversals/s
```

The resulting approximately 7,600 traversals/s is an arithmetic roof, not a
claim that a batch-1 decoder emits 7,600 tokens/s. A one-full-weight-stream
memory roof, cache reuse, target width, matrix utilization, dequantization,
state updates, and acceptance all remain separate terms.

## Provenance and disposition

The source reconstruction is retained as `HISTORICAL` context because its
original local receipt is not part of this public tree. The equations above are
`DERIVED` from the cited public hardware figures and stated counting
convention. No private path, host identifier, raw receipt, model payload, or
machine telemetry is required to audit the arithmetic.

- [AMD Radeon RX 9060 XT specifications](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9060xt.html)
- [AMD RDNA4 instruction-set reference](https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture)
- [Research methodology](methodology.md)
- [Falsified sparse-matrix result](../research/falsified/GFX1200_SPARSE_MATRIX_ANOMALY.md)
- [Claims ledger](../CLAIMS.md)
