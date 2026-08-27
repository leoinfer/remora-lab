# R4X research

R4X explores compact weight blocks and explicit row-window loading for
caller-supplied model files. The current public implementation records the
D32A geometry and rejects unsupported types. Wider and more aggressive
variants remain experimental; no public release promises model compatibility
or quality parity.

The variant crosswalk, including R4X-D, R4X-H, R4X-S, R4X-D32A, XP-S,
regional precision, storage-vs-execution precision, QAT, 2:4 sparsity, mask
learning, and kernel-shape/logical-prefill-row research, is in the
[`implementation map`](../implementation-map.md). D32A geometry is the
bounded public representation; the other tracks remain research descriptions
until independent vectors, quality evidence, and provenance are cleared.

The recovered full-model diagnostic is documented in the
[R4X logical-prefill-row reproduction lane](../../repro/r4x/width-sweep/). Its clean
`ubatch=512` slice contains `llama-bench -p W` values W64 through W2048 and
peaks at 699.677849 logical prefill rows/s at W512. This is not generation
tokens/s. The historical
`ubatch=4096` attempt is retained as a malformed-prefix diagnostic, and no
authoritative `-p 4096` logical prefill-row measurement was found. The exact historical runtime
was a foreign research implementation; the public validation command remains
Rust-only and does not make that implementation an active HAR alternative.
