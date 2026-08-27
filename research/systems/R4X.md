# R4X

**Status:** `EXPERIMENTAL`

R4X is a family of model-aware weight/block geometry and execution studies.
The research compares precision layouts, region-specific policies, packing,
sparsity, quantization-aware training, and storage/execution tradeoffs. The
archived names include R4X-D, R4X-H, R4X-S, R4X-D32A, and XP-S; these are
tracks, not interchangeable formats.

The bounded public format work is in [`formats/r4x`](../../formats/r4x/),
with research notes in [`research/r4x`](../r4x/). The source register explains
why model-derived plans, weights, checkpoints, and raw benchmark receipts are
not included. R4X is not a hidden fallback backend for HAR.

## Questions retained

- Does the packing layout reduce exposed bytes at the actual kernel tile size?
- Do regional precision policies preserve quality on held-out prompts?
- Does the storage win survive dequantization, launch, and residency costs?
- Can a public format certificate distinguish a valid pack from an unverified
  model-specific artifact?

The relevant negative knowledge and experiment cards should be read before
turning a geometry hypothesis into an implementation claim.
