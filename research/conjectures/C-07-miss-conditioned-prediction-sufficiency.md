---
id: C-07
status: CONJECTURE
source: ../archival/authoritative/REMORA_NEW_CONJECTURES.md
originality_status: unknown
---

## C-07 — Miss-Conditioned Prediction Sufficiency

**Status: `CONJECTURED`**

### Claim

A route predictor should be considered sufficient only when its conditional value on the expensive-miss subset exceeds a minimum threshold, not when its all-request F1 or average recall passes. A practical gate is:

```text
VWRecall_expensive_miss >= q
and
wasted_bytes / useful_on_time_bytes <= w
and
arrival_before_deadline >= d.
```

### Why stronger

It formalizes the static Phase 2 observation that history-only predictors can look acceptable globally while being useless on expensive first-appearance churn.

### Counterexample search

Reweight an existing trace so resident hits have zero cost and first-appearance misses dominate. A global-F1 policy should fail the value gate.

### Cheapest decisive test

Existing 17-token route replay with cost labels and held-out split; no hardware.

### Certificate

Store the miss subset definition before evaluation, cost weights, deadlines, useful/wasted bytes, and confidence intervals.

### Affected ideas

H06/H07/H10/H14/H15/H18/H27/H38; manifest `3`, `5`, `6`, `14`, `24`; N05/N06/N18.

---
