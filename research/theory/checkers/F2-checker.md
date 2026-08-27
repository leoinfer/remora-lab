---
id: F2
status: OPEN_PROBLEM
source: ../../archival/authoritative/REMORA_FORMALIZATION_QUEUE.md
originality_status: unknown
---

## F2 — Finite optimal-stopping checker

**Status: `PROVED` for finite abstract DP; `EXPERIMENTALLY TESTABLE`**

- **Input:** finite state graph, actions `{stop, extend}`, conditional outcomes, costs, salvage value.
- **Check:** exhaustive Bellman value and policy; compare TBEH bound policy.

```python
V[s, L] = max(stop_value[s, L],
              max(expected_reward[a] - cost[a] + V[next_state, L+1]
                  for a in extend_actions))
```

- **Adversarial cases:** non-monotone marginals, zero-value positions, late high-value branch, state-dependent cost.
- **Output:** optimal policy, regret, bound coverage, false-stop/false-extend cases.
- **Certificate:** `REMORA-STOP-001`.
- **Affected:** OP-01, TBEH, manifest `1/2/5/11/19/20`.
