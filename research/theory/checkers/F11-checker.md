---
id: F11
status: OPEN_PROBLEM
source: ../../archival/authoritative/REMORA_FORMALIZATION_QUEUE.md
originality_status: unknown
---

## F11 — Hardware phenotype compiler/checker

**Status: `DERIVED UNDER ASSUMPTIONS`; `EXPERIMENTALLY TESTABLE` CPU-only first**

- **Input:** phenotype profile, model/source inventory, workload class, candidate actions, safety margins.
- **Check:** profile identity, capacity intervals, plan references, reserve, fallback, invalidation predicates.
- **Adversarial cases:** same hardware/different model, changed driver hash, context/KV overflow, unsupported kernel, queue risk.
- **Output:** safe region(s), conditional plan, fallback and invalidation reason.
- **Certificate:** `REMORA-PHENO-001`.
- **Affected:** OP-10, H12/H14/H19/H29/H32/H37/H38, manifest `21–25/28`, N05/N09/N22/N23.
