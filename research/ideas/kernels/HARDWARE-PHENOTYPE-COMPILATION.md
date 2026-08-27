# Hardware phenotype compilation

**Status:** `PROPOSED`

Hardware phenotype compilation maps measured device capabilities, memory
hierarchy, transfer costs, and kernel behavior to a safe region of runtime
configurations. It is deliberately broader than selecting one benchmark
optimum: the compiler must retain certificates, bounds, and fallback choices.

The source relationships are H12, H32, H37, H38, H39, and conjecture C-09.
The implementation boundary is the Rust HAR planner/Vulkan path; no native
foreign execution body is implied. See [`HAR`](../../systems/HAR.md) and the
formal checker cards [`F8`](../../theory/checkers/F8-checker.md),
[`F11`](../../theory/checkers/F11-checker.md), and
[`F12`](../../theory/checkers/F12-checker.md).
