---
id: F8
status: OPEN_PROBLEM
source: ../../archival/authoritative/REMORA_FORMALIZATION_QUEUE.md
originality_status: unknown
---

## F8 — Resource-constrained critical-path checker

**Status: `DERIVED UNDER ASSUMPTIONS`; `EXPERIMENTALLY TESTABLE`**

- **Input:** job DAG, resource vectors, capacities, buffer/fence ownership, measured intervals.
- **Check:** precedence, capacity, memory, fence lifetime, and critical path. Report whether claimed overlap has independent-resource intervals.
- **Adversarial cases:** hidden shared copy engine, queue barrier, CPU/DRAM contention, staging use-before-fence.
- **Output:** feasible schedule, lower bound, binding resource, overlap witness.
- **Certificate:** `REMORA-RCPSP-001`.
- **Affected:** OP-07/OP-10/OP-12, H13/H17/H18/H22/H29/H37/H38, manifest `16–25`.
