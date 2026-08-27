# Resource-complementarity scheduling

**Status:** `PROPOSED`

Resource-complementarity scheduling chooses work that uses currently slack
resources without violating correctness or reserve constraints. It is a
research generalization of overlapping compute, transfer, verification, and
maintenance lanes; overlap is valuable only when the trace shows that the
resources are genuinely complementary.

The main source is the HERMES atlas, especially its asynchronous lanes,
telemetry, and motorway-merge mechanisms. See [`REMORA`](../../systems/REMORA.md)
and [`F8`](../../theory/checkers/F8-checker.md).
