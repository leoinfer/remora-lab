---
id: C-05
status: CONJECTURE
source: ../archival/authoritative/REMORA_NEW_CONJECTURES.md
originality_status: unknown
---

## C-05 — Union-Aware Residual Cache (UARC)

**Status: `CONJECTURED`**

### Claim

The right cache unit for speculative MoE blocks is not an individual expert or token, but a `(layer, expert, residual/refinement class)` item scored by:

```text
expected avoided critical-path cost
  + probability of reuse after rejection
  - bytes held
  - eviction opportunity cost
  - validation cost.
```

The cache should prefer residual tiles or refinement classes that serve many branches, even when their parent expert is not the most frequent item.

### Why stronger

It joins H05 residual tiles, H08 future canvas, H10 atlas, H21 compound/isolation, H27 salvage, OP-08 value-weighted residency, and PHASE branches. It attacks union growth rather than only whole-expert hit rate.

### Counterexample search

Generate a route tree where the most frequent expert is branch-specific and a less frequent residual tile is shared across all branches. Compare LRU/expert-frequency/UARC.

### Cheapest decisive test

CPU trace replay with existing route unions and synthetic residual sizes; exact output is not needed to falsify the economic policy.

### Certificate

Record parent artifact/root, residual use count, branch coverage, bytes, holding time, avoided reload cost, and observed salvage use. Credit only realized reuse.

### Affected ideas

H05/H08/H10/H11/H21/H27/H30/H38; manifest `5–8`, `11–15`; N01/N18/N23.

---
