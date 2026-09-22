# Representation utilization lane

Bounded public disposition for the utilization-first representation research.

```sh
./repro/representation/utilization-2026-09-22/run.sh
```

The measured part is the unpack cost ordering: at a real expert shape, effective
throughput falls 297.6 to 229.7 to 188.8 GB/s as the representation becomes
harder to unpack, even though the byte count goes down. That is the argument for
`utilization first, bpw second`.

Everything else here is labelled: the per-weight ALU estimates are modelled, the
`pc4` candidate has no measured performance, the GSQ/RCO synthesis is **not
implemented**, and the Flash-Next 40-56 GB / 2.25-2.5 bpw band is a projection
from another model's ratios. See
[`research/representation/UTILIZATION.md`](../../../research/representation/UTILIZATION.md).
