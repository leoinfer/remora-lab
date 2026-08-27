# R4X W-semantics correction

This is the durable terminology record for the recovered R4X campaign.

## Authoritative meaning

In the historical command, `llama-bench -p W` sets `n_prompt`. Therefore each
W value in the recovered receipt is a count of logical prefill rows submitted
to the model. `-ub` sets the microbatch chunk size. Neither option selects a
shader local size, subgroup width, workgroup width, or compile-time tile.

The published `results.csv`, `matrix.csv`, expected record, and sanitized
receipt use `logical_prefill_rows` for this dimension so that the meaning is
machine-readable.

## Corrected historical note

Some earlier private summaries called the W64–W2048 series a kernel-width or
workgroup-width sweep. That label was wrong for this receipt and must not be
used as an alternative interpretation. It arose from confusing the logical
prefill-row argument with the physical M dimension used inside the historical
executor.

Physical operator/kernel-shape experiments, including the separate
`test-r4x-widem` surface and its M-shaped probes, remain valid research
artifacts in their own scope. They are not evidence about the semantics of the
full-model `llama-bench -p W` series and must not be merged with it.

## Evidence boundary

The clean authoritative slice is ubatch 512 at logical prefill rows
64, 128, 256, 384, 512, 768, 1024, 1536, and 2048. No measured logical
prefill-row W4096 point was found. The malformed ubatch 4096 attempt is a
microbatch experiment with a W2048 device-loss boundary; it is not a W4096
experiment.

## Naming rule

Future receipts must use `logical_prefill_rows` or `n_prompt` for this series.
The terms kernel width and workgroup width are reserved for measured dispatch
geometry and may be used here only inside an explicit corrected-history note.
