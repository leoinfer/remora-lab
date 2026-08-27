---
id: F1
status: OPEN_PROBLEM
source: ../../archival/authoritative/REMORA_FORMALIZATION_QUEUE.md
originality_status: unknown
---

## F1 — Accepted-token roofline checker

**Status: `PROVED` accounting inequality; `MACHINE-CHECKED` only for static inputs**

- **Input:** interval ledger `{accepted_tokens, resource_bytes, capacities, critical_path, overlap_intervals}`.
- **Check:**

```python
T_bound = max(critical_path,
              max(work[r] / capacity[r] for r in resources))
rate_bound = accepted / T_bound if T_bound else 0
assert measured_rate <= rate_bound + tolerance or evidence_class != "MEASURED"
```

- **Units:** bytes/second, seconds, exact committed tokens; no draft denominator.
- **Adversarial cases:** rejected work omitted, cache bytes mislabeled physical, double-divided decode interval, `accepted=0`.
- **Output:** per-resource bounds, binding bottleneck, allowed bytes/token for target rates.
- **Certificate:** `REMORA-ROOF-001`.
- **Affected:** H01/H18/H27/H38; OP-02; all manifest speed claims.
