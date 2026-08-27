---
id: F15
status: OPEN_PROBLEM
source: ../../archival/authoritative/REMORA_FORMALIZATION_QUEUE.md
originality_status: unknown
---

## F15 — Trace schema completeness checker

**Status: `MACHINE-CHECKED` for current missing fields; `BLOCKED` for live replay**

- **Input:** DeepSeek `.tr`/JSONL traces, Qwen JSON/telemetry, TBEH schema.
- **Check:** whether a trace contains target authority, state, draft/verify/accept, resource costs, deadlines, and split identity required by its claimed question.
- **Output:** `usable_for_route`, `usable_for_residency`, `usable_for_horizon`, `usable_for_exactness` flags.
- **Adversarial cases:** route-only trace presented as target economics, stale/truncated trace, blank metrics as zero.
- **Certificate:** `REMORA-TRACE-001`.
- **Affected:** DeepSeek H06/H08/H09, TBEH, PHASE, RSSO, N26.

## Formalization priority

| Priority | Item | Why first |
|---:|---|---|
| 0 | F0/F15 | Resolves missing ranking/status and prevents source/trace overclaim |
| 1 | F1/F10 | Cheaply rejects impossible byte/value claims |
| 2 | F4/F6/F12/F14 | Exactness/state/provenance composition is the central blocker |
| 3 | F2/F5/F7/F9 | Makes horizon, delta, branch, and residency policies falsifiable |
| 4 | F8/F11/F13 | Scheduler/phenotype/controller models after invariants close |
| 5 | F3 | TBEH live replay only after B0 produces valid traces |

## Provisional family track — PFM

**Progressive Future Materialization (PFM-A / PFM-B)** is a newly registered
formal research family, not an additional `F0–F15` identifier. Its canonical
specification is:

- `[local path omitted]`
- `[local path omitted]`

PFM-A may proceed only through CPU/static schemas, deterministic artifact
transition checks, deadline/resource oracle replay and adversarial cost-ledger
cases. It depends on the existing F1/F4/F6/F8/F9/F10/F12/F14/F15 contracts and
B0 repeatability; it does not reopen B0 or authorize RSSO, target batching,
ExpertPack runtime work, TBEH live replay or a production hot path.

The PFM oracle must compare sequential out-of-core inference, predictive
prefetch, ordinary speculation, SpecExec-style verification, PFM-A width 1,
PFM-A width 2 and the perfect-knowledge PFM-A upper bound under equal model,
prompt, memory, draft, context and measurement boundaries. It must charge
preparation, promotion, state/KV, verification, queueing, contention, rollback,
disposal and wasted work. The oracle kill gate is at least 10% throughput gain
without meaningful energy regression, or at least 10% joule/token reduction
without meaningful throughput regression. PFM-B remains training/formal-only
until PFM-A oracle economics and future-state compressibility/correction
measurements justify explicit authorization.

PFM does not alter the exact `F0–F15` count; its state, packet, transition,
validity, cost and gate definitions are maintained in the dedicated
architecture/experiment documents. Missing fields remain `UNKNOWN`/`BLOCKED`,
never zero.

**Investigation result (2026-08-03):** the fair perfect-information oracle
failed PFM-0, PFM-1, PFM-2 and PFM-4 in the established project-parameter
envelope; PFM-3 remains blocked for lack of measured future-state residuals.
PFM-A is therefore **REJECTED as a distinct incremental execution mechanism**.
Its typed state/accounting overlay may remain as static infrastructure. PFM-B is
**DEFERRED**, with no training authorization.

## Queue exit rule

A formalization may hand off to the experimental agent only if its checker can produce `PASS`, `FAIL`, `BLOCKED`, or `NOT_RUN` without interpreting a blank field as zero and without invoking live GPU work implicitly.
