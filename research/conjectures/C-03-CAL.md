---
id: C-03
status: CONJECTURE
source: ../archival/authoritative/REMORA_NEW_CONJECTURES.md
originality_status: unknown
---

## C-03 — Certified Approximation Lattice (CAL)

**Status: `CONJECTURED`**

### Claim

Approximation mechanisms should be ordered by a refinement relation rather than treated as unrelated modes:

```text
cheap anchor
  <= residual refinement
  <= delta-bound local result
  <= verified target result
  <= exact committed state
```

A refinement may be used only if it consumes a certificate that bounds the remaining uncertainty. A node can fall back upward but cannot silently fall sideways into another approximate node.

### Why stronger

It unifies H03 compact skeleton, H04 progressive widening, H05 residual tiles, H07 margins, H23 purification, H24 protection, manifest `27 Delta-Certified skipping`, N17 cascade correction, and N19–N21 format variants.

### Counterexample search

Build a two-layer lattice where local errors are small but state divergence differs. Try to compose a local token certificate without a state certificate; the checker must reject the edge.

### Cheapest decisive test

Finite symbolic lattice checker with explicit error intervals, state roots, and fallback edges. No GPU.

### Certificate

Each node has authority root, uncertainty interval, state-equivalence status, and allowed consumers. Edges state whether they are exact refinement, verified proposal, or approximate transition.

### Affected ideas

H02–H08/H23/H24/H28/H33; manifest `1`, `2`, `6–10`, `25`, `27`; N17/N19–N23.

---
