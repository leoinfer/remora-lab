# REMORA-VERIFY

**Status:** `PROPOSED`

REMORA-VERIFY is the progressive verification layer for adaptive state and
speculation. It escalates checks when uncertainty, dependency changes, or
certificate weakness crosses a policy boundary, and records enough trace
context to replay the decision.

Related source records are the REMORA master prompt, certificate formalization,
and [`F12`](../checkers/F12-checker.md). This proposal does not authorize an
unverified fast path in HAR; the runtime remains fail-closed where its
certificate or state contract is incomplete.
