---
id: F13
status: OPEN_PROBLEM
source: ../../archival/authoritative/REMORA_FORMALIZATION_QUEUE.md
originality_status: unknown
---

## F13 — Shadow-price policy checker

**Status: `CONJECTURED`; `EXPERIMENTALLY TESTABLE`**

- **Input:** finite resource/DAG simulator, action values/demands, hard viability constraints, price update rule.
- **Check:** compare policy to exhaustive optimum; ensure hard filter precedes price choice; detect oscillation and reserve violations.
- **Adversarial cases:** correctness recovery action with high price, discrete cache jump, price oscillation.
- **Output:** regret, violations, debt trajectory, fallback count.
- **Certificate:** `REMORA-PRICE-001`.
- **Affected:** OP-12, H15/H17/H20/H22/H25/H26/H29/H34, manifest `16–20`.
