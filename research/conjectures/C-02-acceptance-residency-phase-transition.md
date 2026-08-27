---
id: C-02
status: CONJECTURE
source: ../archival/authoritative/REMORA_NEW_CONJECTURES.md
originality_status: unknown
---

## C-02 — Acceptance–Residency Phase Transition

**Status: `CONJECTURED` with `DERIVED UNDER ASSUMPTIONS` necessary condition**

### Claim

Once exposed demand satisfies the accepted-token roofline only at a required hit fraction `h*`, improvements in predictor accuracy have near-zero end-to-end value until the system crosses the demand/churn threshold. There is a phase transition between:

1. **saturated regime:** even a perfect predictor cannot meet the byte/critical-path bound;
2. **slack regime:** prediction and scheduling can convert slack into accepted-token throughput.

Necessary condition for target rate `r`:

```text
(1-h) * demand_bytes_per_token * r <= bottleneck_bandwidth.
```

### Why stronger

It turns the static DSpark “oracle stalls like baseline” result into a general stopping rule for Route Scout, PHASE, and predictive residency. It says “improve representation/demand first,” not “train a better predictor.”

### Counterexample search

Construct traces with the same predictor recall but varying demand/churn. Check that the predictor only changes critical-path time after the roofline has slack.

### Cheapest decisive test

Offline replay with a synthetic multiplicative demand reducer and the same route predictor; report stall avoided per byte.

### Certificate

Resource ledger must show `demand`, `exposed`, `accepted`, `bandwidth`, `critical path`, and whether the policy is saturation-limited.

### Affected ideas

H06/H08/H10/H15/H18/H38; manifest `6`, `7`, `14`, `24`, `27`; N01/N02/N18/N23.

---
