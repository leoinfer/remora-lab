---
id: F5
status: OPEN_PROBLEM
source: ../../archival/authoritative/REMORA_FORMALIZATION_QUEUE.md
originality_status: unknown
---

## F5 — Delta bound and adversarial margin checker

**Status: `PROVED` local argmax lemma; `EXPERIMENTALLY TESTABLE` bound checker**

- **Input:** full logits, cheap logits or interval, claimed `epsilon`, top-k, state/route metadata.
- **Check:**

```python
actual_eps = max(abs(full[i] - cheap[i]) for i in vocab)
margin = full[top1] - full[top2]
local_exact = (actual_eps <= epsilon and margin > 2*epsilon)
```

- **Sequence check:** require state hash equality or a separate state bound; token equality alone returns `NOT_CERTIFIED`.
- **Adversarial cases:** top-tie, `gamma=2epsilon`, hidden-state drift, stochastic sampler.
- **Output:** local argmax certificate or explicit rejection.
- **Certificate:** `REMORA-DELTA-001`.
- **Affected:** OP-04, manifest `27`, H03–H05/H07/H23/H24, N17/N19.
