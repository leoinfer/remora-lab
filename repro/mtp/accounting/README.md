# MTP acceptance-accounting lane

This lane reproduces the public accounting rule for draft/target acceptance
without a model. It runs 263 deterministic rounds at maximum draft depth 3,
records the accepted-prefix histogram and trace digest, and keeps the
historical 240/263 and approximately 3.16 anchors explicitly separate from the
synthetic fixture.

Run from the repository root:

```sh
./repro/mtp/accounting/run.sh
```

The receipt demonstrates acceptance bookkeeping only. It is not a neural MTP
quality, latency, or speedup result.
