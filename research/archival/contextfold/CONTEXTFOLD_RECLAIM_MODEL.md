# Reclaim and computational refrigerator

For representation level `r`, block `b` has memory `M_b(r)`, materialization time
`T_b(r)`, reconstruction energy `E_b(r)`, use probability `p_b`, queue cost
`Q_b(r)`, and horizon `H_b`.  A shadow-priced controller chooses

```
r* = argmin [ λ_M M + λ_T p T + λ_E p E + λ_Q Q ]
subject to T <= H / R_token when a deadline exists.
```

The scalar simulator compares hot VRAM, raw RAM, ContextPack, replay, and token
archive representations.  It deliberately shows that a memory-optimal cold
representation may violate a readiness deadline.  Recency is only one signal;
attention mass, query similarity, prefix reuse, layer type, and future MTP use
should feed `p_b` and `H_b`.

No scheduler implementation is proposed for the active runtime.
