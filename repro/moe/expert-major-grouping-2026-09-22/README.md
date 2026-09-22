# Expert-major grouping lane

Bounded public disposition for grouped (expert-major) execution of routed
experts on the Alice MoE layer.

```sh
./repro/moe/expert-major-grouping-2026-09-22/run.sh
```

The measured local result is large — 0.185 to 6.115 TMAC/s at B=4096 — and the
parity gate is bit-exact. Neither fact licenses a full-model claim: the grouped
expert GEMM is ~44.6% of the step, giving an Amdahl ceiling of ~1.65x, and the
deployed-configuration gate returned `REVERT` anyway.

The CPU-MoE prefill path measured -85.1% (581.3 to 87.1 t/s at `pp4096`) and is
retired. See
[`research/alice/README.md`](../../../research/alice/README.md).
