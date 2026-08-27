---
id: F12
status: OPEN_PROBLEM
source: ../../archival/authoritative/REMORA_FORMALIZATION_QUEUE.md
originality_status: unknown
---

## F12 — Certificate composition linter

**Status: `CONJECTURED`; `EXPERIMENTALLY TESTABLE`**

- **Input:** component certificates with `precondition`, `postcondition`, scope, state boundary, authority root, fence epoch, denominators.
- **Check:** postcondition of each edge satisfies next precondition; scopes cover the claimed final result; no `NOT_RUN`/approximate component is silently promoted.
- **Adversarial cases:** remove graph-order field, state hash, route map, timing denominator, or model identity.
- **Output:** composed/not-composed verdict with missing interface fields.
- **Certificate:** `REMORA-COMP-001`.
- **Affected:** OP-11, H02/H33/H39, Q2, RSSO, N01/N02/N18.
