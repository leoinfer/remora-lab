---
id: C-06
status: CONJECTURE
source: ../archival/authoritative/REMORA_NEW_CONJECTURES.md
originality_status: unknown
---

## C-06 — Slack-Priced Elastic Horizon (SPEH)

**Status: `CONJECTURED`**

### Claim

Elastic horizon and resource complementarity should share a scalar **verified-token slack price** derived from current critical-path slack and resource debt. Horizon extension is allowed only when its conservative tail value exceeds both its own cost and the price of consuming scarce future slack.

```text
extend if upper_tail_value
  > incremental_cost + lambda_slack * slack_consumed.
```

### Why stronger

TBEH prices omitted tail; H17/H20/H22/H29 price resource reserve; SPEH connects them without assuming all resources are fungible.

### Counterexample search

Create a trace where an extension is locally profitable but consumes the only recovery slot needed by a likely near-term correction. A token-only TBEH policy should lose to SPEH.

### Cheapest decisive test

Finite resource/DAG replay with one recovery event and one horizon extension; compare exhaustive optimum, TBEH, and SPEH.

### Certificate

Record slack definition, price update, resource debt, upper-tail bound, and post-action reserve. Reject if `lambda_slack` is fitted on the evaluation trace without a split.

### Affected ideas

TBEH, H15/H17/H20/H22/H29/H34; manifest `1`, `2`, `16–20`; OP-01/OP-07/OP-12.

---
