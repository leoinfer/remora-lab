---
id: F10
status: OPEN_PROBLEM
source: ../../archival/authoritative/REMORA_FORMALIZATION_QUEUE.md
originality_status: unknown
---

## F10 — Value-of-computation event ledger

**Status: `PROVED` bookkeeping identity; `EXPERIMENTALLY TESTABLE`**

- **Input:** baseline event graph, optimized event graph, artifact IDs, realized reuse, intervals.
- **Check:** disjoint cost categories sum to total; credits never exceed observed avoided baseline events; approximate/informational events cannot receive exact credits.
- **Adversarial cases:** unused prefetch, double reuse credit, future value guessed but not observed, bytes credit without time witness.
- **Output:** conservation identity, unexplained residual, per-token value.
- **Certificate:** `REMORA-VALUE-001`.
- **Affected:** OP-09, H21/H25/H27/H28/H33/H38, manifest `11–20`, N26.
