# Benchmarks

No benchmark result in this fresh candidate should be read as a performance
claim. Add receipts only when the command, model identity, hardware context,
warm-up policy, output, and comparison baseline can be reproduced publicly.

Hardware context is a versioned phenotype, not just a marketing name. Use the
reference ID `RX9060XT16-NITRO-GFX1200-RADV-2026.08.27-v1` when applicable and
follow [`HARDWARE_PROFILE.md`](../HARDWARE_PROFILE.md) and the detailed
[`benchmark receipt schema`](../docs/benchmarks.md). Factory specifications,
configured settings, measured telemetry, and historical observations must not
be conflated.

The included [`local-bench`](local-bench/) tool estimates configurations from
model metadata and user-supplied hardware profiles. It does not belong to the
HAR runtime. Its optional measurement command may invoke an external serving
binary for research calibration and is not used by any HAR executable.

The recovered R4X width result is documented in the
[R4X width-sweep lane](../repro/r4x/width-sweep/). Its unit is
kernel/prefill diagnostic rows/s, not generation tokens/s. The lane links the
sanitized receipt, full width-by-ubatch status matrix, exact historical source
commit, and the Rust-only public validation command.

The artifact-level crosswalk for benchmark implementations, receipts, and
unreproduced historical results is
[`TECHNICAL_ARTIFACT_INDEX.md`](../TECHNICAL_ARTIFACT_INDEX.md).
