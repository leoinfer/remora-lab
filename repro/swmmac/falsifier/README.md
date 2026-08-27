# gfx1200 SWMMAC falsifier lane

This lane preserves the negative result that an apparent multi-POPS sparse
INT4 measurement was invalid. A deterministic Rust known-answer fixture shows
the replayed accumulator committing one value where four independent values
were expected; the observed result is exactly one quarter of the expected
value.

Run from the repository root:

```sh
./repro/swmmac/falsifier/run.sh
```

The receipt separates instruction activity from useful committed work. The
historical instruction-rate calculation is retained as context only; no fresh
GPU ISA measurement, useful TOPS claim, or LLM throughput claim is made.
