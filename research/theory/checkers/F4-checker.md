---
id: F4
status: OPEN_PROBLEM
source: ../../archival/authoritative/REMORA_FORMALIZATION_QUEUE.md
originality_status: unknown
---

## F4 — RSSO finite-state exactness checker

**Status: `EXPERIMENTALLY TESTABLE`; live target data `BLOCKED`**

- **Input:** toy recurrent transition, causal attention dependency, candidate block `K<=4`, rejection at each prefix.
- **Check:** compare sequential and proposed wavefront state/output hashes at every boundary; verify rejected suffix cannot mutate authority.

```python
for schedule in enumerate_schedules(K):
    seq = sequential(target, candidates)
    got = run_schedule(schedule, candidates)
    assert got.accepted_state_hash == seq.state_hash[got.A]
    assert got.committed_outputs == seq.outputs[:got.A]
    assert got.authority_state_writes <= got.A
```

- **Adversarial cases:** recurrent state read before update, causal mask removed, suffix state committed, shared buffer recycled before fence.
- **Output:** valid schedule set, invalid schedule witnesses, state-memory lower bound.
- **Certificate:** `REMORA-RSSO-001`.
- **Affected:** OP-03, H08/H09/H13/H21/H33/H35/H36, manifest `6–10`.
