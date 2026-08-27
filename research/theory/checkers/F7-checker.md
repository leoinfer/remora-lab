---
id: F7
status: OPEN_PROBLEM
source: ../../archival/authoritative/REMORA_FORMALIZATION_QUEUE.md
originality_status: unknown
---

## F7 — PHASE branch-DAG enumerator

**Status: `EXPERIMENTALLY TESTABLE`; real accepted outcomes `BLOCKED`**

- **Input:** branch DAG with conditional mass, shared prefix, preparation/validation/holding/rollback costs, salvage rules.
- **Check:** normalize probability mass at every node; enumerate all branch subsets under budget; compare proposed selection to optimum.
- **Adversarial cases:** shared-prefix double count, branch mass >1, unreachable suffix, unvalidated salvage, correlated outcomes.
- **Output:** exact optimum, policy regret, omitted-mass upper bound.
- **Certificate:** `REMORA-PHASE-001`.
- **Affected:** OP-06, H08/H09/H17/H21/H27, manifest `5/8/11/14`.
