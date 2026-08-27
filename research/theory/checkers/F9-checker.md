---
id: F9
status: OPEN_PROBLEM
source: ../../archival/authoritative/REMORA_FORMALIZATION_QUEUE.md
originality_status: unknown
---

## F9 — Predictive residency/Belady replay

**Status: `MACHINE-CHECKED` for equal-page static traces; weighted/live `EXPERIMENTALLY TESTABLE`**

- **Input:** route trace, per-item bytes, cache capacity, arrival/deadline cost, policy.
- **Check:** equal-size finite trace against Belady; weighted items against an explicitly declared weighted oracle; compute value-weighted recall.
- **Adversarial cases:** expensive first-appearance misses, high hit rate on zero-cost residents, prefetch pollution, union single-use tail.
- **Output:** hits/misses, bytes, useful/wasted arrivals, stall avoided, roofline slack.
- **Certificate:** `REMORA-RES-001`.
- **Affected:** OP-08, H06/H10/H14/H15/H18/H27/H38, manifest `3/6/7/14/16`.
