# R4X, R4KV, and R4F

**Status:** `EXPERIMENTAL`

These related tracks study the joint choice of representation, packing,
precision, memory placement, and execution kernel. R4X focuses on weight and
block geometry, R4KV on compressed KV pages, and R4F on architecture-aware
Flash-Next execution. Their shared falsifier is exposed-byte and end-to-end
cost accounting at the real tile and sequence shapes.

The system entry points are [`R4X`](../../systems/R4X.md),
[`R4KV`](../../systems/R4KV.md), and [`R4F`](../../systems/R4F.md). Model
payloads and private measurements remain excluded.
