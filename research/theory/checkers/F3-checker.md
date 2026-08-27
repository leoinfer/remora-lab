---
id: F3
status: OPEN_PROBLEM
source: ../../archival/authoritative/REMORA_FORMALIZATION_QUEUE.md
originality_status: unknown
---

## F3 — TBEH trace replay verifier

**Status: `BLOCKED` pending valid MTP traces**

- **Input:** `tbeh_trace_schema.json`, held-out temporal traces, five policies.
- **Check:** recompute survival, tail bound, absolute/relative error, exact committed denominator, regret.
- **Required split:** calibration and evaluation must not share the same trace; `rho_bound_validated` must be false unless independently justified.
- **Adversarial cases:** `rho>=1`, empty tail, high-cost rare event, bound fitted on evaluation trace, `NOT_RUN` rows filled with zero.
- **Output:** policy rows and gate report.
- **Certificate:** `REMORA-TBEH-001`.
- **Affected:** TBEH, H15/H23/H24/H29/H34.
