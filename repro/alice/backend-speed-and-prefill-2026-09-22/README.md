# Alice backend speed and prefill lane

Bounded public disposition for the Alice throughput records, with backend and
warm-state labels preserved.

```sh
./repro/alice/backend-speed-and-prefill-2026-09-22/run.sh
```

Two things this lane exists to prevent:

- **18.430 t/s is the historical Vulkan configuration of record**, not current
  HIP performance. The current clean HIP raw `K0` reading is 15.374/15.422 t/s
  against 18.360 t/s Vulkan in the same broader comparison.
- **662.7437 t/s is a hot run**, the third request of a three-repetition
  warm-repeat in one session (cold 91.345, warm 245.614). It is not a stable
  baseline, and the cross-session "9.3x" headline built on a 71.105 t/s
  baseline from a different session, ubatch, and offload configuration is
  inadmissible.

See [`research/alice/README.md`](../../../research/alice/README.md).
