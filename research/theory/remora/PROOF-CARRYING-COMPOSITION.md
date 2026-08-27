# Proof-carrying composition

**Status:** `PROPOSED`

Proof-carrying composition requires each optimization stage to emit the
certificate, assumptions, dependency versions, and invalidation conditions
needed by the next stage. A composed plan is admissible only when those
certificates compose; a collection of individually plausible heuristics is
not automatically safe.

The source relationship is H33 and the REMORA certificate line. The checker
queue begins with [`F12`](../checkers/F12-checker.md), [`F14`](../checkers/F14-checker.md),
and [`F15`](../checkers/F15-checker.md); runtime certificate code is in
[`har-certificates`](../../../har/crates/har-certificates/).
