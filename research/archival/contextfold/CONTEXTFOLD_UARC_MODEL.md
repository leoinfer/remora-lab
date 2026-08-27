# Union-Aware Residual Context Cache

For query residual tile sets `R_1..R_J`, fetch the union rather than each set:

```
B_independent = Σ_j |R_j|
B_union = |∪_j R_j|
S_union = B_independent - B_union
```

The CPU model varies J, tile overlap, and speculative acceptance.  Positive
savings require overlapping residual requests; rejected candidates can cause
wasted materialization.  This is compatible with lossless state if each tile is
reconstructed byte-for-byte, but its arithmetic order remains a separate
contract.  It is not evidence that real Qwen residual tiles overlap.
