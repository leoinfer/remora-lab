---
id: F6
status: OPEN_PROBLEM
source: ../../archival/authoritative/REMORA_FORMALIZATION_QUEUE.md
originality_status: unknown
---

## F6 — Dependency Merkle/causal-closure checker

**Status: `DERIVED UNDER ASSUMPTIONS`; `EXPERIMENTALLY TESTABLE` CPU-only**

- **Input:** artifact DAG, source/tensor/state/config leaves, function/version contract.
- **Check:** recompute Merkle root; compare all transitive leaves; classify exact/approximate/informational.
- **Adversarial cases:** mutate one route ID, state byte, pack offset, graph generation, driver contract, sampler seed, external state.
- **Output:** hit/miss decision and missing dependency list.
- **Certificate:** `REMORA-CCC-001`.
- **Affected:** OP-05, C-01, manifest `12–15/26`, H10/H27/H33, N09/N25.
