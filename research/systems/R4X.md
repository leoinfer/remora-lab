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

The recovered [logical-prefill-row receipt](../../repro/r4x/width-sweep/) is a
historical diagnostic evidence lane. Its W values are prompt-prefill rows
submitted through `n_prompt`; they are not shader workgroup widths and its
rows/s values are not generation tokens/s. The public Rust command validates
the D32A format contract while the exact historical full-model throughput
rerun awaits a Rust-only model executor.

The historical series title “width sweep” is shorthand for a
`llama-bench -p W` logical-prefill-row sweep. It is not a workgroup/kernel
width sweep. Older notes using the latter label are corrected historical
wording only.

## Questions retained

- Does the packing layout reduce exposed bytes at the actual kernel tile size?
- Do regional precision policies preserve quality on held-out prompts?
- Does the storage win survive dequantization, launch, and residency costs?
- Can a public format certificate distinguish a valid pack from an unverified
  model-specific artifact?

The relevant negative knowledge and experiment cards should be read before
turning a geometry hypothesis into an implementation claim.
