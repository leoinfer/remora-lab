# Benchmarks

No benchmark result in this fresh candidate should be read as a performance
claim. Add receipts only when the command, model identity, hardware context,
warm-up policy, output, and comparison baseline can be reproduced publicly.

The included [`local-bench`](local-bench/) tool estimates configurations from
model metadata and user-supplied hardware profiles. It does not belong to the
HAR runtime. Its optional measurement command may invoke an external serving
binary for research calibration and is not used by any HAR executable.
