# SSD action-memory lane

Bounded public disposition for symbolic action memory on the cold tier.

```sh
./repro/ngram/action-memory-2026-09-22/run.sh
```

The mechanism is real and correctness was preserved on the tested panel, but the
headline did not survive its own audit: the **5.10x** panel figure is
**retracted as contaminated** by a documented serving-path degradation in the
same window, and the admissible clean multiplier is **1.771x**. The
30-to-8-token and 12-to-4-forward figures quoted alongside it are a timing
proxy, not model inference.

Misses cost ~1.00x, which is what makes the path safe to enable. Packed extents
cut instrumented physical read bytes by 72.1%. State reuse has no identity gate
and is not claimable. See
[`research/ssd-action-memory/README.md`](../../../research/ssd-action-memory/README.md).
