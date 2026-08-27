# REMORA Formalization Queue

CPU-only and source-level queue. Nothing here owns the GPU or modifies a hot path. Items are ordered by information value and dependency closure.

## Queue policy

A formalization is complete only when it has:

1. an input schema with units and evidence class;
2. a deterministic checker or exhaustive finite-state procedure;
3. adversarial/property-based cases;
4. a fail-closed result and certificate artifact;
5. a mapping to the relevant source IDs;
6. no hidden promotion from simulation to performance.

## F0 — Source and identifier audit

**Status: `MACHINE-CHECKED` / `BLOCKED` for missing ranking**

- **Input:** source manifest of all audited paths, SHA-256, line counts, status timestamps.
- **Check:** required names `H01–H39`, `N01–N26`, manifest ideas `1–30`, `TBEH` appear exactly once in the crosswalk; required source ranking path exists or emits `BLOCKED`.
- **Adversarial cases:** duplicate MARC acronym, absent AutoSurgeon artifact, stale active-state field.
- **Output:** source-audit JSON and identifier crosswalk.
- **Certificate:** `REMORA-SRC-001`.
- **Affected:** all families; N26.

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

## F3 — TBEH trace replay verifier

**Status: `BLOCKED` pending valid MTP traces**

- **Input:** `tbeh_trace_schema.json`, held-out temporal traces, five policies.
- **Check:** recompute survival, tail bound, absolute/relative error, exact committed denominator, regret.
- **Required split:** calibration and evaluation must not share the same trace; `rho_bound_validated` must be false unless independently justified.
- **Adversarial cases:** `rho>=1`, empty tail, high-cost rare event, bound fitted on evaluation trace, `NOT_RUN` rows filled with zero.
- **Output:** policy rows and gate report.
- **Certificate:** `REMORA-TBEH-001`.
- **Affected:** TBEH, H15/H23/H24/H29/H34.

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

## F6 — Dependency Merkle/causal-closure checker

**Status: `DERIVED UNDER ASSUMPTIONS`; `EXPERIMENTALLY TESTABLE` CPU-only**

- **Input:** artifact DAG, source/tensor/state/config leaves, function/version contract.
- **Check:** recompute Merkle root; compare all transitive leaves; classify exact/approximate/informational.
- **Adversarial cases:** mutate one route ID, state byte, pack offset, graph generation, driver contract, sampler seed, external state.
- **Output:** hit/miss decision and missing dependency list.
- **Certificate:** `REMORA-CCC-001`.
- **Affected:** OP-05, C-01, manifest `12–15/26`, H10/H27/H33, N09/N25.

## F7 — PHASE branch-DAG enumerator

**Status: `EXPERIMENTALLY TESTABLE`; real accepted outcomes `BLOCKED`**

- **Input:** branch DAG with conditional mass, shared prefix, preparation/validation/holding/rollback costs, salvage rules.
- **Check:** normalize probability mass at every node; enumerate all branch subsets under budget; compare proposed selection to optimum.
- **Adversarial cases:** shared-prefix double count, branch mass >1, unreachable suffix, unvalidated salvage, correlated outcomes.
- **Output:** exact optimum, policy regret, omitted-mass upper bound.
- **Certificate:** `REMORA-PHASE-001`.
- **Affected:** OP-06, H08/H09/H17/H21/H27, manifest `5/8/11/14`.

## F8 — Resource-constrained critical-path checker

**Status: `DERIVED UNDER ASSUMPTIONS`; `EXPERIMENTALLY TESTABLE`**

- **Input:** job DAG, resource vectors, capacities, buffer/fence ownership, measured intervals.
- **Check:** precedence, capacity, memory, fence lifetime, and critical path. Report whether claimed overlap has independent-resource intervals.
- **Adversarial cases:** hidden shared copy engine, queue barrier, CPU/DRAM contention, staging use-before-fence.
- **Output:** feasible schedule, lower bound, binding resource, overlap witness.
- **Certificate:** `REMORA-RCPSP-001`.
- **Affected:** OP-07/OP-10/OP-12, H13/H17/H18/H22/H29/H37/H38, manifest `16–25`.

## F9 — Predictive residency/Belady replay

**Status: `MACHINE-CHECKED` for equal-page static traces; weighted/live `EXPERIMENTALLY TESTABLE`**

- **Input:** route trace, per-item bytes, cache capacity, arrival/deadline cost, policy.
- **Check:** equal-size finite trace against Belady; weighted items against an explicitly declared weighted oracle; compute value-weighted recall.
- **Adversarial cases:** expensive first-appearance misses, high hit rate on zero-cost residents, prefetch pollution, union single-use tail.
- **Output:** hits/misses, bytes, useful/wasted arrivals, stall avoided, roofline slack.
- **Certificate:** `REMORA-RES-001`.
- **Affected:** OP-08, H06/H10/H14/H15/H18/H27/H38, manifest `3/6/7/14/16`.

## F10 — Value-of-computation event ledger

**Status: `PROVED` bookkeeping identity; `EXPERIMENTALLY TESTABLE`**

- **Input:** baseline event graph, optimized event graph, artifact IDs, realized reuse, intervals.
- **Check:** disjoint cost categories sum to total; credits never exceed observed avoided baseline events; approximate/informational events cannot receive exact credits.
- **Adversarial cases:** unused prefetch, double reuse credit, future value guessed but not observed, bytes credit without time witness.
- **Output:** conservation identity, unexplained residual, per-token value.
- **Certificate:** `REMORA-VALUE-001`.
- **Affected:** OP-09, H21/H25/H27/H28/H33/H38, manifest `11–20`, N26.

## F11 — Hardware phenotype compiler/checker

**Status: `DERIVED UNDER ASSUMPTIONS`; `EXPERIMENTALLY TESTABLE` CPU-only first**

- **Input:** phenotype profile, model/source inventory, workload class, candidate actions, safety margins.
- **Check:** profile identity, capacity intervals, plan references, reserve, fallback, invalidation predicates.
- **Adversarial cases:** same hardware/different model, changed driver hash, context/KV overflow, unsupported kernel, queue risk.
- **Output:** safe region(s), conditional plan, fallback and invalidation reason.
- **Certificate:** `REMORA-PHENO-001`.
- **Affected:** OP-10, H12/H14/H19/H29/H32/H37/H38, manifest `21–25/28`, N05/N09/N22/N23.

## F12 — Certificate composition linter

**Status: `CONJECTURED`; `EXPERIMENTALLY TESTABLE`**

- **Input:** component certificates with `precondition`, `postcondition`, scope, state boundary, authority root, fence epoch, denominators.
- **Check:** postcondition of each edge satisfies next precondition; scopes cover the claimed final result; no `NOT_RUN`/approximate component is silently promoted.
- **Adversarial cases:** remove graph-order field, state hash, route map, timing denominator, or model identity.
- **Output:** composed/not-composed verdict with missing interface fields.
- **Certificate:** `REMORA-COMP-001`.
- **Affected:** OP-11, H02/H33/H39, Q2, RSSO, N01/N02/N18.

## F13 — Shadow-price policy checker

**Status: `CONJECTURED`; `EXPERIMENTALLY TESTABLE`**

- **Input:** finite resource/DAG simulator, action values/demands, hard viability constraints, price update rule.
- **Check:** compare policy to exhaustive optimum; ensure hard filter precedes price choice; detect oscillation and reserve violations.
- **Adversarial cases:** correctness recovery action with high price, discrete cache jump, price oscillation.
- **Output:** regret, violations, debt trajectory, fallback count.
- **Certificate:** `REMORA-PRICE-001`.
- **Affected:** OP-12, H15/H17/H20/H22/H25/H26/H29/H34, manifest `16–20`.

## F14 — Source-level invariant checker

**Status: `EXPERIMENTALLY TESTABLE`; no production edits required**

- **Input:** read-only source paths and architecture contracts.
- **Check:** presence of original-ID→slot mapping, fence/epoch guard, source range bounds, per-position ID tensor, fail-closed counters, no exact claim from approximate flag.
- **Adversarial cases:** rank-as-slot offset, one route vector broadcast to K rows, reset before fence, zero-fill after allocation failure.
- **Output:** source invariant report with file/symbol/line evidence.
- **Certificate:** `REMORA-SRCINV-001`.
- **Affected:** H02/H10/H11/H13/H33, N01, RSSO LayerPack, ExpertPack.

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
