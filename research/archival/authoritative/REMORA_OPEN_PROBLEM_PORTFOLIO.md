# REMORA Open Problem Portfolio

**Scope:** theory-first synthesis over HERMES `H01–H39`, broader families `N01–N26`, REMORA manifest ideas `1–30`, and TBEH.
**Evidence rule:** a label applies only to the statement immediately attached to it.
**Non-interference:** no live GPU run, no production-code change, no hardware-lock ownership.

## Evidence labels used

Each portfolio entry follows the requested method: precise problem, evidence/failed routes, obstruction, candidate lemma/algorithm, active counterexamples, cheapest falsification, certificate design, affected ideas, and the next justified experiment. Proof, derivation, simulation, conjecture, and intuition are kept separate by the labels below; a numerical source observation is never silently upgraded.

- **PROVED:** follows from the stated abstract assumptions and a supplied derivation.
- **MACHINE-CHECKED:** an existing trace/certificate/checker or a finite/static computation checks the scoped statement.
- **DERIVED UNDER ASSUMPTIONS:** algebra/model consequence whose assumptions are not yet established for the target runtime.
- **CONJECTURED:** plausible synthesis not yet derived or measured.
- **COUNTEREXAMPLE FOUND:** a constructed or preserved trace invalidates a broad claim.
- **FALSIFIED:** the stated claim fails its specified gate.
- **EXPERIMENTALLY TESTABLE:** a bounded decisive test exists, but has not run.
- **BLOCKED:** a required artifact, authority, or live gate is unavailable.

---

## OP-01 — Elastic-horizon optimality

**Preserved ideas:** REMORA manifest `1 Elastic MTP generation depth`, `2 Elastic verification depth / continuous horizon`, `5 PHASE outcome-tree prediction`, `TBEH`; H08, H15, H21, H23, H24, H25, H29, H34; N18.

### 1. Precise problem

At state `s` and current prepared horizon `L`, choose `stop` or extend a target-authoritative speculative/verification horizon. The objective is net value of exact committed tokens, not draft positions:

```text
J(policy | s) = E[sum accepted_k * value_k]
                - draft_cost - target_cost - transfer_cost
                - rollback_cost - contention_cost - reserve_cost.
```

Let `q_k(s)` be conditional survival to position `k`, `v_k(s)` be gross value if position `k` is covered, and `c_k(s)` be incremental cost. `q_k` must be conditional on the preceding accepted prefix, not an unconditional confidence score.

### 2. Evidence and failed approaches

- TBEH specifies `c[t,k]`, cumulative survival `a[t,k]`, retained value `R_t(L)`, omitted tail `T_t(L)`, and a fail-closed fallback.
- `tbeh_policy_replay.csv` contains explicit `NOT_RUN` rows for fixed depth, threshold, expected-value, TBEH, and oracle policies.
- B0 dense Qwen repeatability is `0/3` clean pairs, so valid MTP traces are not available for a live replay.
- Static PHASE/DSpark analysis shows horizon value/byte peaking near `H=4` on a small trace, but this is not a TBEH result.
- A simple confidence threshold fails conceptually when confidence ignores future value, residency, and contention.

**Evidence status: `MACHINE-CHECKED` for the `NOT_RUN` gate; `BLOCKED` for target replay.**

### 3. Physical/mathematical obstruction

The stopping reward is non-stationary. Extending the horizon changes cache state, queue debt, memory pressure, and later acceptance opportunities. Therefore a scalar threshold over confidence is not generally optimal. A scalar horizon policy is valid only under a single-crossing or monotone-marginal assumption that has not been shown.

### 4. Candidate lemmas and algorithms

#### Lemma A — Bellman stopping condition

For a finite state/action model, define `V_stop(s,L)` and `V_ext(s,L)`. The optimal value is:

```text
V*(s,L) = max(V_stop(s,L), V_ext(s,L)).
```

If `V_ext(s,L) <= V_stop(s,L)`, stopping is optimal at that state. This is **`PROVED`** for the finite dynamic-programming model; it does not prove that a confidence threshold estimates `V_ext` correctly.

#### Lemma B — geometric omitted-tail bound

Assume, for all `j >= 1`,

```text
P(A >= L+j | A >= L) <= rho^(j-1),   0 <= rho < 1,
0 <= v_(L+j) <= v_max.
```

Then:

```text
E[omitted gross value after L]
  <= q_(L+1) * v_max * sum(j=1..infinity) rho^(j-1)
  = q_(L+1) * v_max / (1-rho).
```

This is **`PROVED`** under the displayed assumptions. A fitted `rho` is not a proof of those assumptions. On held-out data it is at most a calibrated upper bound unless an independent sound construction exists.

#### Safe stop rule

Let `U_L` upper-bound the gross value of every continuation after `L`, including future salvage and avoided work. Let `C_min,L` lower-bound the cost of every continuation that reaches any later value. If:

```text
U_L <= C_min,L
```

then stopping is optimal among those continuations, assuming no omitted external reward and nonnegative continuation cost. This is **`PROVED`** under the stated bounds. Comparing `U_L` to an arbitrary one-position cost is not sound unless that cost lower-bounds every continuation.

#### Algorithm

1. Compute conditional survival and state-dependent cost features.
2. Estimate a held-out upper interval for `q_k` or tail value.
3. Include state value: reuse, residency, recovery reserve, and contention.
4. Stop only when the full-continuation upper bound is below a cost lower bound.
5. If calibration, cost, state, or authority is unknown, use fixed safe depth/full target.

### 5. Active counterexamples

- **Non-monotone marginal value:** position 1 has net value `-1`, position 2 has net value `+99`, and position 2 is reachable after position 1. A first-negative marginal threshold stops too early.
- **State-dependent cost:** the same confidence can justify extension for a resident pack and rejection for a cold pack.
- **Tail bound misuse:** a bound on token count does not bound target verification, rollback, memory, or contention value.

Status: **`COUNTEREXAMPLE FOUND`** against universal first-negative threshold optimality.

### 6. Cheapest decisive falsification

After B0 closes, use temporal/held-out real MTP traces and compare:

1. fixed depth;
2. simple confidence threshold;
3. expected-value marginal controller;
4. TBEH;
5. post-hoc oracle best depth.

Reject TBEH if held-out regret is worse after all costs, coverage fails under miss stress, or it never beats the simpler policies. Do not implement live code for this test.

Status: **`EXPERIMENTALLY TESTABLE`**, currently **`BLOCKED`** by B0.

### 7. Certificate design

A machine-checkable row contains:

```text
model_hash, runtime_commit, prompt_hash, temporal_split,
conditional_acceptance[], survival[], rho, rho_bound_validated,
upper_tail_value, lower_continuation_cost,
generated, verified, accepted, exact_committed,
draft_cost, verify_cost, rollback, memory, contention,
absolute_error, relative_error, coverage, fallback_reason.
```

A replay checker must recompute `a_k`, the tail bound, policy action, and exact committed-token denominator. A property-based generator must include non-monotone `q_k`, zero-value positions, and high-cost late branches.

### 8. Mapping and justified next experiment

A valid result would affect manifest ideas `1`, `2`, `5`, `11`, `16–20`, `26`, `27`, and `TBEH`, plus H15/H17/H20/H21/H23–H25/H29/H34 and N18. The next justified experiment is **offline replay only**, after the repeatability gate; no live controller is justified now.

---

## OP-02 — Accepted-token roofline

**Preserved ideas:** H01, H18, H19, H27, H28, H37, H38; REMORA `15 Waste Ledger`, `24 REMORA Flow`; N02, N17, N18, N23.

### 1. Precise problem

Given a run interval with `A` exact committed tokens, resource work `B_r` on resource `r`, and sustainable effective capacity `beta_r`, derive an upper bound on accepted-token rate while distinguishing:

- logical bytes;
- cache-served bytes;
- uploaded bytes;
- physically exposed bytes;
- speculative/rejected work;
- exact committed tokens.

### 2. Evidence and failed approaches

The HERMES Q8 model gives approximately `3.2124 GiB` of six-expert routed payload per token. Measured local bandwidth records include approximately `2.53 GiB/s` cold QD1 storage, `7.20 GiB/s` warm storage, and `6.4 GiB/s` GTT→device copy. The Phase 2 static analysis derives a `94–95%` byte-hit requirement for 30 tok/s. Qwen Q2 changed-slot transport reduces uploaded bytes but repeated exact full-core C is slower and higher gross GPU J/token than F.

**Evidence status: `MACHINE-CHECKED` for the preserved arithmetic inputs; no runtime speed claim.**

### 3. Obstruction

Multiple resources may overlap. The correct lower bound is a maximum of resource loads and the precedence critical path, not an automatic sum. Conversely, if a resource is on the causal critical path, its aggregate load is a direct bottleneck.

### 4. Lemma and derivation

For every resource `r`:

```text
T >= B_r / beta_r.
```

For a precedence DAG with work/latency `W_j` and critical path `CP`:

```text
T >= max(CP, max_r(B_r / beta_r)).
```

Therefore:

```text
accepted_tok/s = A/T
             <= A / max(CP, max_r(B_r/beta_r)).
```

If one bottleneck resource has exposed bytes per accepted token `b=B/A`:

```text
accepted_tok/s <= beta/b.
```

This is **`PROVED`** as an accounting inequality under measured aggregate capacity and correct denominator definitions.

For DeepSeek's nominal full six-expert payload, traffic-only ceilings are:

| Effective bandwidth | Traffic-only upper bound |
|---:|---:|
| 2.53 GiB/s | `0.7876` token/s |
| 7.20 GiB/s | `2.2413` token/s |

At a target of 30 tok/s, allowed exposed bytes are:

| Bandwidth | Bytes/accepted token |
|---:|---:|
| 2.53 GiB/s | `0.084333 GiB = 86.357 MiB` |
| 7.20 GiB/s | `0.240000 GiB = 245.760 MiB` |

For full-payload demand, the required hit fraction is:

```text
h >= 1 - beta / (demand * target_rate).
```

With `demand=3.2124 GiB/token`, this is `97.375%` at 2.53 GiB/s and 30 tok/s, or `92.529%` at 7.20 GiB/s and 30 tok/s. These are **`DERIVED UNDER ASSUMPTIONS`**: actual useful bandwidth, overlap, demand and accepted-token behavior must be measured.

### 5. Counterexamples

- Reduced logical bytes can leave CPU packing, fences, or graph setup unchanged and therefore not improve time.
- More bytes can be faster if they arrive while otherwise idle and replace serial work; byte count alone is not a total-order speed metric.
- Page-cache residency can make an apparent read count differ from physical storage traffic.
- Draft positions/s can rise while exact committed tok/s falls.

Status: **`COUNTEREXAMPLE FOUND`** against “reduced bytes imply speed.”

### 6. Cheapest decisive falsification

For each proposed mechanism, replay a trace and emit an interval ledger containing `A`, all `B_r`, `CP`, resource capacities, overlap intervals, and exact committed tok/s. A proposal that cannot meet the byte bound under its own measured traffic is falsified as a path to that target, without a GPU run.

Status: **`EXPERIMENTALLY TESTABLE`** and mostly **`MACHINE-CHECKED`** as an offline accounting gate.

### 7. Certificate design

The verifier must reject rows when:

- `A` is raw generated or drafted tokens rather than exact committed tokens;
- rejected work is omitted from `B_r` or time;
- logical/cache bytes are substituted for exposed physical bytes without a label;
- capacity is inferred from an unrelated standalone benchmark;
- overlap is inferred from asynchronous API calls rather than intervals.

A symbolic checker should verify units, interval sums, and `rate <= bound + tolerance`.

### 8. Mapping and justified experiment

This result gates H38/H01 and every byte/traffic claim in H03–H18, H21–H24, H27, H37, N01/N02/N17/N18/N23, and REMORA ideas `11–20`, `24`, `27`. It justifies CPU-only cost-model rejection before any live implementation.

---

## OP-03 — RSSO wavefront exactness and break-even

**Preserved ideas:** REMORA `6 RSSO resident skeleton + streamed oracle`, `7 Layer-stationary speculative wavefront`, `8 Small speculative wavefront tree`, `9 Latent Inertial Drafting`, `10 Dense organ map`; H03, H08, H09, H10, H13, H21, H33, H35, H36, H38; N18, N24.

### 1. Precise problem

Can a candidate block of `K` positions be prepared while a LayerPack or target layer is resident, then verified exactly with the same target state and output as `K` sequential target steps, at a lower cost per exact committed token?

### 2. Evidence and failed approaches

- RSSO static freeze, dependency graph, LayerPack design, cost model, residency planner and skeleton search are complete as static artifacts.
- Qwen3.6-27B has 64 trunk blocks, 48 recurrent layers, 16 full-attention layers, and one separate MTP block. Recurrent state and attention KV are different state classes.
- No K-position target batching, skeleton agreement, rollback, overlap, or RSSO energy input exists.
- Qwen Q2 currently certifies one-position transport; multi-position input is explicitly fail-closed.
- HERMES H09 union curves show possible grouping but also large union growth and single-use members.

**Status: `MACHINE-CHECKED` static dependency facts; live exactness `BLOCKED`.**

### 3. Physical obstruction

Weight reuse is not state reuse. For recurrent layers, position `i+1` depends on the state produced by position `i`. For attention layers, candidate positions require causal masking and hidden inputs from preceding layers. A layer-stationary schedule is exact only if it preserves those dependencies or uses a mathematically exact scan/associative transformation.

### 4. Candidate invariant and theorem

Let `F_i` be the deterministic target transition for position `i`, including all recurrent/KV state. Let `s_0` be the authoritative state. Sequential execution is:

```text
(y_i, s_i) = F_i(x_i, s_(i-1)).
```

A wavefront verifier is exact for accepted prefix `A` if, for every `i <= A`, every scheduled primitive reads inputs observationally equal to the sequential execution and the committed boundary state satisfies:

```text
hash(s_A_wavefront) = hash(s_A_sequential)
```

with the same output/logit contract. By induction on `i`, this invariant implies target-equivalent output/state for the accepted prefix. This is **`PROVED`** as a transition-system invariant.

It does **not** prove that a proposed schedule satisfies the invariant. For this model, a layer-stationary block needs one of:

1. exact recurrent scan/associative composition;
2. serial recurrent state order within the layer;
3. a source-level proof that the graph's state update is independent across positions.

No such live proof is present. Status: **`BLOCKED`**.

### Break-even condition

For actual accepted prefix `A` and block time:

```text
T_block(K) = T_draft + T_target_oracle
             + T_transfer + T_state_snapshot/commit
             + T_rollback + T_scheduler + T_contention.
```

Exact wavefront economics beat sequential target time `T_seq` only if:

```text
T_block(K) < A * T_seq.
```

This is **`PROVED`** by comparing total wall time for the same number of committed tokens. Replacing `A` with draft length or verified positions is invalid.

Static recurrent-state sizing gives approximately 3,145,728 F32 bytes per recurrent-layer snapshot before additional state, or roughly 144 MiB for 48 layers for one state snapshot. A K-position rollback design may require `Omega(K * state_size)` storage unless it has a sound reversible/scan representation. This is **`DERIVED UNDER ASSUMPTIONS`**.

### 5. Counterexamples

- `K=2`, `A=1`, no true weight reuse, and one extra state snapshot can be slower than two sequential steps.
- A later candidate's hidden state can be computed from a speculative predecessor but cannot be committed if an earlier prefix rejects.
- A shared LayerPack can reduce transfer bytes while increasing scatter, state, and synchronization time.
- Equal final token IDs do not establish equal recurrent state.

Status: **`COUNTEREXAMPLE FOUND`** against “resident weights imply exact wavefront gain.”

### 6. Cheapest decisive falsification

First use a CPU-only finite-state toy with one recurrent recurrence and one causal attention-like dependency. Enumerate sequential and candidate schedules for `K<=4`, including rejection at every prefix. Any schedule that violates state hashes is rejected. Only then use a trace-replay cost model with existing route unions. Live K=2 target verification is justified only after B0 and the RSSO gate change.

Status: **`EXPERIMENTALLY TESTABLE`**, currently **`BLOCKED`**.

### 7. Certificate design

Per block record:

```text
candidate_token_ids,
sequential_state_hash[i], wavefront_state_hash[i],
per-layer dependency schedule,
causal mask proof/reference,
recurrent state snapshot IDs,
accepted_prefix A, rejected_suffix state discard,
LayerPack source hashes, bytes and fence ownership,
raw/verified/accepted/committed denominators,
rollback and contention costs.
```

The source-level checker should assert that no committed suffix state is copied into authoritative state. A trace-replay verifier should compare all accepted boundary state hashes, not only tokens.

### 8. Mapping and justified experiment

A pass would justify REMORA ideas `6–10`, H08/H09/H13/H21/H33/H35/H36/H38, N18 and possibly N24. The next justified action is formal CPU finite-state checking and offline break-even replay; no hot-path RSSO code is justified under the current gate.

---

## OP-04 — Delta-Certified skipping

**Preserved ideas:** REMORA `27 Delta-Certified skipping`; H02, H03, H04, H05, H07, H23, H24, H28, H33; N17, N19–N23.

### 1. Precise problem

When may a cheap computation omit a target block/expert and still certify the next greedy token, route decision, or exact state? The cheap path must carry an explicit worst-case error bound, not a calibrated confidence score.

### 2. Evidence and failed approaches

- The manifest proposes `||z_full-z_cheap||_∞ <= epsilon` and top-one margin `gamma > 2 epsilon`.
- HERMES numerical certificates are explicitly “numerically close, not bit-exact.”
- Q6 MoE-Skipper broad predictor paths fail direct-logit/KL/top-token gates despite very high raw throughput.
- Dense/Qwen near-tie diagnostics show that small numerical differences can amplify through greedy argmax; no exact epsilon bound is available.

**Status: `DERIVED UNDER ASSUMPTIONS`; end-to-end sequence skip `BLOCKED`.**

### 3. Obstruction

A local output bound does not automatically bound downstream hidden/state error. A token argmax guarantee does not guarantee a sampled distribution, route identity, or next-step recurrent state.

### 4. Lemmas

#### Argmax margin lemma

Let `z` be full logits and `z'` cheap logits. If:

```text
||z-z'||_infinity <= epsilon
margin(z) = z_top1 - z_top2 > 2*epsilon,
```

then `argmax(z') = argmax(z)`. Proof: the top logit can fall by at most `epsilon`, the runner-up can rise by at most `epsilon`, so their order remains strict. **`PROVED`**.

For a top-k set, the analogous boundary condition is the margin between the kth and `(k+1)`th full logits greater than `2 epsilon`.

#### Layer-composition bound

If downstream maps have known Lipschitz constants `L_j` and local approximation errors are `epsilon_i`, then a conservative final bound is:

```text
||delta_final|| <= sum_i epsilon_i * product_(j downstream of i) L_j.
```

This is **`PROVED`** under the stated norm/Lipschitz assumptions, but useful constants are not established for the target graph.

#### Sequence exactness condition

Per-position argmax equality is insufficient for recurrent/dense sequence exactness unless the authoritative state at each accepted boundary is also equal or a future state-equivalence theorem is supplied. Status: **`PROVED` by the transition invariant in OP-03**, not a live result.

### 5. Counterexamples

- If `gamma <= 2 epsilon`, an adversary can swap the top two logits within the allowed interval.
- A cheap path can emit the same token while leaving a different recurrent state, causing later divergence.
- A small average RMS can hide a single near-tie or structured-output failure.
- For stochastic sampling, equal argmax is irrelevant; valid speculative rejection sampling or a distributional certificate is required.

Status: **`COUNTEREXAMPLE FOUND`** against using empirical RMS/calibration as an exact certificate.

### 6. Cheapest decisive falsification

CPU-only adversarial search: take recorded full logits and search perturbations in the claimed epsilon box. If any top token flips for a row claimed certified, reject the bound. Then test route-bound and state-bound conditions separately. Current near-tie rows make this a cheap high-value check, but the actual perturbation radius is not yet measured.

Status: **`EXPERIMENTALLY TESTABLE`**; no live skip is authorized.

### 7. Certificate design

Store per decision:

```text
full_logit_hash,
cheap_logit_hash,
epsilon_source and proof,
full_top1/top2 margin,
router margin if routing is affected,
state input/output hashes,
sequence position,
mode = exact_local | verified | approximate.
```

The verifier must reject `exact_sequence` if only token IDs are present. Interval arithmetic, exact integer/float replay where possible, and source-level norm bounds are preferable to calibration.

### 8. Mapping and justified experiment

This directly gates manifest idea `27`, H03–H05/H07/H23/H24/H28/H33 and approximate N17/N19–N23. The first justified experiment is an offline adversarial bound checker, not a runtime skip.

---

## OP-05 — Dependency-versioned exact cognition reuse

**Preserved ideas:** REMORA `12 REMORA Reclaim`, `13 Computational refrigerator / artifact provenance`, `14 Value-weighted salvage cache`, `26 Dependency-versioned cached cognition`; H10, H27, H30, H33; N04, N05, N09, N24, N25.

### 1. Precise problem

When can an intermediate activation, state, route decision, expert packet, or “cognition artifact” be reused exactly rather than merely as an approximate draft?

### 2. Evidence and failed approaches

- REMORA requires provenance fields, expiry, and validation rules.
- Qwen B0 found same-process prompt-cache state despite request-level `cache_prompt=false`; cache controls did not explain fresh-process divergence but exposed an omitted-state hazard.
- HERMES stale arena, slot, and epoch failures show that model hash alone is insufficient.
- MARC-Symbiote explicitly requires semantic and hardware fingerprints plus refresh triggers; a final hidden vector alone is insufficient.

**Status: `MACHINE-CHECKED` for preserved failure examples; exact reuse theorem `DERIVED UNDER ASSUMPTIONS`.**

### 3. Obstruction

The word “same prompt” does not identify the same computation. Reuse can depend on token IDs, position, prior recurrent/KV state, model/tensor hashes, router path, precision, backend arithmetic, graph version, hardware state, RNG/sampler state, and external/tool state.

### 4. Causal-closure theorem

Let a deterministic artifact be:

```text
a = f(D, state, config, implementation_version)
```

Let `Root(D,state,config,implementation_version)` be a collision-resistant Merkle root over the complete transitive dependency closure. If:

1. all causal inputs are represented in the closure;
2. the function is deterministic under the recorded implementation contract;
3. the root and artifact payload hash match;
4. no external mutable state is omitted;

then reusing `a` is observationally exact for that contract. This is **`PROVED`** by function extensionality under equal inputs/version; cryptographic collision resistance is an engineering assumption, not a mathematical identity guarantee.

Partial reuse is valid only for a subgraph whose entire dependency closure is unchanged. A root change in one leaf invalidates dependent artifacts but need not invalidate independent subgraphs.

### 5. Counterexamples

- Same prompt text, different tokenization or prior KV/recurrent state.
- Same model hash, different backend/kernel/driver arithmetic, if backend-level exactness is part of the authority contract.
- Same route IDs, different expert byte pack or slot contents.
- Same output token, different hidden state.
- Same process, stale prompt-cache/slot state not declared in the request.

Status: **`COUNTEREXAMPLE FOUND`** against prompt-hash-only or model-hash-only caching.

### 6. Cheapest decisive falsification

Build a CPU-only Merkle-key replay checker. Recompute the key after changing exactly one dependency: tokenization, state snapshot, source tensor byte, quantization type, graph version, sampler seed, or hardware phenotype. Every changed causal dependency must cause a cache miss; changing an independent dependency should not.

Status: **`EXPERIMENTALLY TESTABLE`**, no GPU required.

### 7. Certificate design

Artifact record:

```text
artifact_id, function_id, module/layer, position,
input_token_hash, dependency_root,
state_snapshot_hash, model_hash, source_pack_hash,
precision/backend/driver contract, sampler/RNG state,
creation_cost, owner, expiry, validation_rule,
exact_or_approximate, fallback_authority.
```

A trace-replay verifier recomputes closure roots from source manifests and rejects reuse without a matching validation rule. “Calibrated confidence” is never accepted as an exactness field.

### 8. Mapping and justified experiment

This unifies manifest ideas `12–15`, `26`, H10/H27/H30/H33, N04/N05/N09/N24/N25, ExpertPack, LayerPack and Symbiote artifact management. The first justified work is a CPU/property-based provenance checker and artifact schema; no cache hot path is justified.

---

## OP-06 — PHASE branch economics

**Preserved ideas:** REMORA `5 PHASE outcome-tree prediction`, `8 Small speculative wavefront tree`, `11 REMORA Portion`, `12 REMORA Reclaim`, `14 Value-weighted salvage cache`; H08, H09, H17, H21, H24, H27, H29, H34; N18.

### 1. Precise problem

Select which alternative future branches to prepare while current work runs. Branches may share prefixes, have conditional probabilities, consume memory, and produce salvageable artifacts. The decision must maximize expected exact committed-token value after all costs.

### 2. Evidence and failed approaches

- PHASE requires B=1/2/4/8 coverage replay and actual-outcome coverage.
- TBEH proposes omitted branch mass times maximum avoidable future cost as an upper bound.
- Static union analysis shows branch/future route working sets can grow rapidly; 64% of union experts are single-use in one K=7 analysis.
- No valid PHASE branch trace exists; B0 blocks target-authoritative MTP replay.

**Status: `BLOCKED` for live evidence; model is `DERIVED UNDER ASSUMPTIONS`.**

### 3. Obstruction

Branches are a DAG, not an independent list. Shared-prefix work is paid once. Branch probabilities are conditional on the current state and prefix. A branch prepared after an earlier rejection may be causally irrelevant. Memory opportunity and validation costs are state-dependent.

### 4. Candidate economics

For prepared branch nodes `b`, let `I_b` indicate preparation, `q_b` be probability the node is reached/validated, `v_b` be avoided authoritative cost or accepted value, `c_b` generation/transfer cost, `m_b` holding/opportunity cost, and `s_b` observed salvage value:

```text
E[net] = sum_b q_b*(v_b + s_b - validation_b)
         - sum_b I_b*(generation_b + transfer_b + holding_b)
         - contention - rollback.
```

This is **`DERIVED UNDER ASSUMPTIONS`** and must be evaluated on a dependency DAG.

#### Omitted-mass upper bound

If omitted probability mass is `M`, every omitted branch has avoidable value at most `V_max`, and `p` is a sound upper probability bound, then:

```text
expected omitted value <= M * V_max.
```

If this upper bound is no larger than a lower bound on the complete cost of adding omitted branches, adding them is not profitable under the model. This is **`PROVED`** under those bounds. It is a rejection certificate, not a positive-gain proof.

#### Complexity warning

Selecting optional branches with memory and cost budgets contains knapsack as a special case. Therefore a universal exact greedy branch selector is not expected. This is **`PROVED`** by reduction at the abstract combinatorial level.

### 5. Counterexamples

- Marginal branch probabilities sum above one because overlapping branches were counted independently.
- A branch's later positions are credited even though its prefix would be rejected.
- Shared-prefix setup is charged once in reality but once per branch in the model, or vice versa.
- Unverified prepared work is treated as exact reuse without a dependency certificate.

Status: **`COUNTEREXAMPLE FOUND`** against independent-probability additive economics.

### 6. Cheapest decisive falsification

Use synthetic finite trees first: enumerate B=1,2,4,8 with shared prefixes, rejection at each depth, and explicit artifact reuse. Compare exhaustive optimum with the proposed policy. The current 17-token route trace may only test union/multiplicity; it cannot decide accepted-prefix economics without target outcomes.

Status: **`EXPERIMENTALLY TESTABLE`**, current target replay **`BLOCKED`**.

### 7. Certificate design

A branch certificate includes:

```text
node_id, parent_id, conditional_probability,
shared_prefix_id, candidate_token/state hashes,
prepared/validated/accepted flags,
all generation/transfer/memory/validation/rollback costs,
salvage artifact dependency root and later observed use.
```

The checker must prove total conditional probability mass is valid at each node, charge shared work once, and never credit unvalidated suffixes as exact.

### 8. Mapping and justified experiment

This affects PHASE, TBEH, H08/H09/H17/H21/H24/H27/H29/H34 and N18. The next justified experiment is an exhaustive CPU tree checker and synthetic cost replay; live branch prep is not justified.

---

## OP-07 — Resource-complementarity scheduling

**Preserved ideas:** H12–H22, H29, H31, H32, H37, H38; REMORA `16–25`; N02, N05, N09, N22, N23.

### 1. Precise problem

Schedule expert reads, CPU population, H2D transfers, kernels, verification, state snapshots, and recovery over heterogeneous resources while respecting precedence, capacities, memory, fences, and deadlines. Determine when two operations are genuinely complementary and can overlap.

### 2. Evidence and failed approaches

- HERMES deferred staging and Q3→Q8 ReBAR repair show transport path selection matters.
- Graphics/transfer queue experiments caused context loss; queue overlap is not a free assumption.
- Qwen XP0 `upload_us` excludes actual DMA, queue submit, fence wait, and MMID; a timer label cannot be promoted to end-to-end transport time.
- Qwen energy and B0 telemetry show memory/allocator state can drift across runs.

**Status: `MACHINE-CHECKED` failure boundaries; full scheduler `CONJECTURED`/`EXPERIMENTALLY TESTABLE`.**

### 3. Obstruction

A job has a resource-demand vector, not one duration:

```text
CPU, DRAM, NVMe, PCIe/GTT, VRAM BW, GPU compute,
VRAM capacity, staging capacity, queue/fence, thermal/power.
```

Two jobs with complementary nominal resources can still share a hidden queue, cache, allocator, or fence. Memory capacity is a cumulative packing constraint, not a bandwidth number.

### 4. Candidate formulation and lemmas

Let `J` be jobs, `r` resources, `w_jr` resource work, `C_r` capacity, and `CP` the precedence critical path. Any schedule satisfies:

```text
T >= CP,
T >= sum_j(w_jr)/C_r for every r,
peak_memory(t) <= capacity - reserve.
```

This lower bound is **`PROVED`** for a resource-constrained schedule under aggregate capacity assumptions.

Two jobs may overlap only if:

1. their precedence constraints permit it;
2. their simultaneous resource demands fit all capacity constraints;
3. their buffers/fences have distinct ownership;
4. the backend actually exposes independent execution engines.

This is **`DERIVED UNDER ASSUMPTIONS`**.

Use a time-expanded min-cost flow for transfer paths and an RCPSP/ILP/CP-SAT solver for small offline traces. Online use can be a shadow-price or admission approximation, never an unbounded claim of optimality.

### 5. Counterexamples

- Asynchronous submission calls can serialize behind a barrier.
- A prefetch that uses DRAM can delay the current CPU population and increase the critical path.
- Two transfers can fit in separate nominal queues but contend for one physical copy engine.
- A larger staging pool can reduce resets while causing VRAM/GTT exhaustion.

Status: **`COUNTEREXAMPLE FOUND`** against API-as-overlap inference.

### 6. Cheapest decisive falsification

CPU-only first: generate a small timestamped DAG from existing ledger scopes and compare the predicted critical path to an interval replay. Reject any model that claims overlap without a distinct resource/interval witness. A later live micro-test must use one variable and a safe queue configuration.

Status: **`EXPERIMENTALLY TESTABLE`**; no live run performed.

### 7. Certificate design

Record per job:

```text
job_id, predecessor_ids, resource intervals,
bytes, queue, buffer ownership, fence/epoch,
start/end timestamps, cancellation, failure, thermal state.
```

A checker verifies precedence, no overlapping over-capacity intervals, no use-before-fence, and that reported overlap reduces the critical path rather than merely moving work into a queue.

### 8. Mapping and justified experiment

This formalizes H13/H14/H15/H17/H18/H19/H20/H21/H22/H26/H29/H31/H32/H37/H38, REMORA `16–25`, N02/N05/N09/N22/N23. The next justified step is an offline resource ledger and critical-path checker, not queue-flag activation.

---

## OP-08 — Predictive MoE residency bounds

**Preserved ideas:** H06, H10, H14, H15, H17, H18, H27, H30, H38; REMORA `6`, `14`, `16–18`, `24`; N01/N02/N05/N06/N23.

### 1. Precise problem

Bound the maximum useful benefit of predictive expert residency/prefetch for a fixed route trace and capacity. Determine when better prediction can matter and when demand/churn must first be reduced.

### 2. Evidence and failed approaches

Static Phase 2 findings report:

- miss-conditioned F1 `0.107–0.172` versus overall `0.354`;
- previous-token retention effectively a prefetch no-op;
- Belady bound approximately `0.574` hit at 8 GiB, with practical LRU requiring about 24 GiB for comparable behavior;
- route-history learning behaves approximately like LFU;
- K=7 union factor approximately `1.79`, with 64% single-use union experts;
- VRAM promotion needs estimated `P(needed) ≳ 0.27` in the measured cost model;
- 30 tok/s requires roughly `94–95%` byte hit rate in that model;
- oracle prefetch stalls almost identically in the saturated regime.

These are small/model-dependent static traces, not universal DeepSeek/Qwen runtime facts.

**Status: `MACHINE-CHECKED` for the preserved static replay; generalization `BLOCKED`.**

### 3. Obstruction

Prediction has value only on expensive, on-time misses. A high score on already-resident or cheap requests can inflate F1 without reducing the critical path. Variable expert sizes make ordinary page-count Belady results insufficient unless the state is expanded into weighted units.

### 4. Bounds and algorithms

#### Belady bound

For a fixed finite trace of equal-sized pages and cache capacity `C`, evicting the page whose next use is farthest in the future minimizes misses. This is **`PROVED`** for the standard offline paging model. The preserved Phase 2 Belady rows are **`MACHINE-CHECKED`** within that model.

It is not automatically a bound for:

- variable-size Qwen records;
- prefetch cost and pollution;
- queue/deadline effects;
- multi-resource transfer;
- speculative branches.

#### Value-weighted predictor metric

For request `i`, let `d_i` be avoided critical-path cost if prepared on time, `p_i` predictor inclusion, and `u_i` useful on-time preparation. Define:

```text
VWRecall = sum_i d_i * u_i / sum_i d_i.
```

A predictor should be promoted only if:

```text
expected avoided_cost
  > load_cost + holding_cost + opportunity_cost + misprediction_cost.
```

This is **`DERIVED UNDER ASSUMPTIONS`** and subsumes the static `p >= 0.27` threshold as a model-specific cost-model output, not a universal constant.

#### Saturation bound

If each exact accepted token demands `D` bytes on a bottleneck of bandwidth `beta`, hit fraction `h` leaves `(1-h)D` exposed traffic. Necessarily:

```text
(1-h)D * target_rate <= beta.
```

This is the OP-02 roofline applied to residency and is **`PROVED`** under the same accounting assumptions.

### 5. Counterexamples

- Overall route F1 can be high while expensive miss F1 is low.
- A previous-window oracle can predict many future experts but still fail the byte budget because the union is too large.
- An exact Belady cache can have a high hit rate on a trace but no end-to-end speedup if compute/synchronization dominates.
- Keeping a frequent expert can evict a rare but very expensive next miss.

Status: **`COUNTEREXAMPLE FOUND`** against F1/hit-rate-only promotion.

### 6. Cheapest decisive falsification

Run the existing finite trace replay with cost-weighted misses and compare LRU, LFU, reuse-weighted eviction, Belady, and predictor policies. For each policy emit exposed bytes, useful on-time bytes, evictions, wasted prefetch, and critical-path stall. No GPU required. Do not transfer Qwen top-8 numbers to DeepSeek top-6.

Status: **`EXPERIMENTALLY TESTABLE`**; current live state is gated.

### 7. Certificate design

A trace verifier must include exact route sequence, expert sizes, cache capacity, source arrival model, queue/deadline assumptions, and policy decision. For fixed equal-sized traces, compare hits/misses to Belady exactly. For weighted traces, verify the chosen weighted oracle or label it heuristic.

### 8. Mapping and justified experiment

This gates Route Scout, persistent atlas, residency, ExpertPack, DSpark memory lookahead, HERMES roofline, N01/N02/N05/N06/N23 and REMORA salvage/reserve/flow ideas. The next justified action is value-weighted CPU replay and no live prefetch change.

---

## OP-09 — Value-of-computation conservation

**Preserved ideas:** REMORA `11–20`, `15 Waste Ledger`, `17 Reserve mobilization`, `24 REMORA Flow`; H01, H14, H15, H21, H22, H25, H27, H28, H29, H33, H38; N26.

### 1. Precise problem

Prevent a controller from claiming value multiple times or claiming future value before it is observed. Every computation, transfer, prefetch, snapshot, and discarded branch must be assigned a cost and a single value class.

### 2. Evidence and failed approaches

- REMORA Reclaim proposes exact reusable, conditionally reusable, informational, and unrecoverable classes.
- HERMES and Qwen failures preserve skipped uploads, stale payload, and invalid quality rows rather than crediting them.
- Qwen Q2 demonstrates that byte savings can be real while speed/energy benefit is absent.
- MARC V0 reports cost-policy savings but its heuristic judge does not establish semantic quality.

**Status: `MACHINE-CHECKED` at the evidence-policy level; conservation theorem is an accounting derivation.**

### 3. Obstruction

A unit of work can be useful in more than one hypothetical counterfactual but can only receive one realized credit. Reuse value is unknown at creation. A cache hit is not a saved critical-path interval unless it avoids a measured reload/recompute or changes an exposed dependency.

### 4. Ledger invariant

For a fixed exact workload, define baseline cost `B` and optimized cost `C`. Partition optimized events into:

- `U`: authoritative accepted work;
- `R`: observed reused artifacts that avoid a baseline event;
- `O`: overlap/critical-path credits with interval witnesses;
- `X`: extra speculative/preparation/holding/rollback work;
- `Q`: contention and reserve debt.

A valid accounting identity is:

```text
C = U + X + Q - R - O
```

or equivalently:

```text
B - C = avoided_baseline_work + measured_overlap
        - extra_work - contention - unreturned_reserve_debt.
```

The identity is **`PROVED` by disjoint ledger definitions**. It is not a physical law and cannot rescue incomplete instrumentation.

If all resource loads, precedence, and critical-path intervals are unchanged, no speed improvement can be claimed; this follows from the OP-02 lower bound. Status: **`PROVED` under the same schedule contract**.

### 5. Counterexamples

- Credit a prefetch at admission, then credit it again when it hits.
- Count reduced upload bytes as avoided wall time without measuring the replaced operation.
- Count a cache hit on a non-critical expert as a token-level speed credit.
- Count later reuse of an artifact whose state/version key was invalid.

Status: **`COUNTEREXAMPLE FOUND`** against unversioned/double-counted value ledgers.

### 6. Cheapest decisive falsification

Use a CPU event ledger with a deliberately useless prefetch, a reused artifact, and a rejected branch. The checker must report no reuse credit for the unused prefetch, one credit for the observed reuse, and all rejected work as waste unless a validity rule later proves reuse.

Status: **`EXPERIMENTALLY TESTABLE`**, no live run needed.

### 7. Certificate design

Every event gets a unique ID, parent/causal IDs, resource cost, artifact root, predicted value, realized value, and credit recipient. A validator checks:

```text
sum category costs == total measured cost,
credits <= avoided baseline events,
no artifact credited before valid reuse,
no exact credit for approximate/informational output.
```

### 8. Mapping and justified experiment

This is the formal backbone for REMORA Portion/Reclaim/Refrigerator/Salvage/Waste Ledger, H14/H15/H21/H22/H25/H27/H28/H29/H33/H38, and N26. It justifies a CPU ledger/checker before any adaptive controller.

---

## OP-10 — Hardware phenotype compilation

**Preserved ideas:** REMORA `21 REMORA Link`, `23 REMORA Morph`, `24 REMORA Flow`, `25 REMORA Verify`; H12, H14, H18, H19, H29, H32, H37, H38; N05, N09, N22, N23.

### 1. Precise problem

Compile measured hardware, model, and workload facts into a safe runtime phenotype: resident set, staging depth, queue use, kernel family, transfer source, reserve, and fallback policy. Determine what can be compiled statically and what must remain online.

### 2. Evidence and failed approaches

- Hardware audits measure GTT, NVMe, DRAM, ReBAR, VRAM, kernel, clock, queue, and staging sensitivities.
- Queue flags caused context loss; advertised hardware capability was not an execution certificate.
- Q3/Q8 gap analysis shows quantization changes byte geometry and staging feasibility.
- Qwen full-core placement changes device-local residency dramatically while leaving GTT nearly flat; the same placement fact is not a speed proof.

**Status: `DERIVED UNDER ASSUMPTIONS`; profile safety is `EXPERIMENTALLY TESTABLE`.**

### 3. Obstruction

Hardware phenotype alone is insufficient. A plan is a function of:

```text
hardware identity × model/source geometry × runtime version
× workload/route state × objective/correctness mode.
```

A kernel microbenchmark that does not match the critical shape is not a phenotype result.

### 4. Candidate compilation contract

Let `h` contain measured capacity intervals and topology, `m` model/source geometry, and `w` workload/state. Compile `P = Compile(h,m,w)`. Admission is safe when interval upper bounds satisfy:

```text
resident_bytes + runtime_bytes + reserve <= measured_capacity_lower,
load_r/C_r_lower <= deadline slack,
all queue/fence contracts hold,
correctness mode is unchanged,
profile identity matches exactly.
```

This is **`PROVED` as a sufficient safety checker** if the intervals and upper bounds are sound. It does not prove performance.

Use static compilation for identity, layouts, compatible kernels, and safe capacity regions. Keep route-conditioned residency, horizon, and admission online.

### 5. Counterexamples

- Same GPU, different model quantization: different expert bytes and optimal staging.
- Same model, different context/KV occupancy: different VRAM reserve.
- Same command, different driver/queue state: different overlap and correctness behavior.
- Same hardware, different workload: route-stable versus first-appearance churn requires different cache decisions.

Status: **`COUNTEREXAMPLE FOUND`** against a universal hardware-only profile.

### 6. Cheapest decisive falsification

CPU-only profile checker first: feed two synthetic models and two workload states into one hardware profile and verify that the compiler either emits different plans or explicitly marks the plan conditional. Then validate identity invalidation by changing model/driver/shader hash.

Status: **`EXPERIMENTALLY TESTABLE`**; no startup GPU autotune is authorized by this pass.

### 7. Certificate design

```text
phenotype_id,
hardware/GPU/driver/Vulkan/CPU/RAM/storage identity,
model_hash, source_inventory_hash, shader/build hashes,
measurement command and raw artifact hashes,
capacity intervals, confidence/sample counts,
compiled plan, safety margins, reserve,
invalidation predicates, fallback plan.
```

A source-level checker verifies all plan references exist and all capacity constraints are conservative. A replay checker separates profile selection from measured speed.

### 8. Mapping and justified experiment

This formalizes H12/H14/H18/H19/H29/H32/H37/H38, REMORA `21–25`, N05/N09/N22/N23 and the MARC hardware fingerprint. The next justified action is a CPU phenotype compiler/schema checker; live microbenchmarks wait for the owning experiment gate.

---

## OP-11 — Proof-carrying composition of local certificates

**Preserved ideas:** H02, H10, H11, H13, H33, H39; REMORA `22`, `25`; N01, N02, N18, N26.

### 1. Precise problem

When do local certificates compose into an end-to-end exact certificate? A one-layer source/slot hash, a fence oracle, and a full-core placement certificate each pass separately; their conjunction is not automatically a full-model state/throughput proof.

### 2. Evidence and failed approaches

- Qwen Q2 one-token, eight-token, mixed, and full-core artifacts are intentionally scoped.
- HERMES upload replay and control certificates are not bit-exact full-model proofs.
- The Qwen Q2 initial invalid attempt had exact source hashes but wrong graph order; the interface contract was incomplete.
- RSSO static dependency graph has all tensor ranges but no live rollback/state proof.

**Status: `CONJECTURED` as a general composition system; failure examples are `MACHINE-CHECKED`.**

### 3. Obstruction

Certificate interfaces must expose the state boundary and scope. If certificate A ends before a fence/graph dependency and certificate B assumes the next state without hashing it, the composition has a gap.

### 4. Candidate Hoare-style rule

For component `C_i`, require a contract:

```text
{ Pre_i } C_i { Post_i }
```

where `Post_i` contains all state hashes, ownership/fence conditions, model/source roots, and correctness class required by `Pre_(i+1)`. If `Post_i` syntactically/reflexively satisfies `Pre_(i+1)` for every edge, and the final component emits the target-authority invariant, then composition preserves exactness. This is **`PROVED`** for the abstract Hoare composition rule; the project-specific contracts are not yet complete.

### 5. Counterexamples

- Exact payload but wrong slot map.
- Exact slot map but copy occurs before route callback.
- Exact token IDs but divergent recurrent state.
- Full-core placement but cross-placement logits differ; the certificate only covers within-placement compact/control equality.

Status: **`COUNTEREXAMPLE FOUND`** against “all local PASS markers imply end-to-end exactness.”

### 6. Cheapest decisive falsification

Create a JSON contract linter that deliberately removes one required interface field: source root, state hash, fence sequence, route map, scope, or timing denominator. It must reject composition. Use existing Q2/HERMES certificates as fixtures without running inference.

Status: **`EXPERIMENTALLY TESTABLE`**, CPU-only.

### 7. Certificate design

Add `precondition`, `postcondition`, `scope`, `authority_root`, `state_boundary_hashes`, `fence_epoch`, `denominators`, and `evidence_class` to each certificate. A composition manifest lists the directed component graph and verifier version.

### 8. Mapping and justified experiment

This affects every H33 certificate, H39 integration, N01/N02/N18/N26, RSSO and REMORA Verify. The next justified experiment is a static contract linter and composition checker.

---

## OP-12 — Shadow-price scheduling for resource complementarity

**Preserved ideas:** H15, H17, H18, H19, H20, H22, H25, H26, H29, H34, H37; REMORA `16–25`; N05, N09, N26.

### 1. Precise problem

Convert the multi-resource scheduling problem into an online policy that prices scarce resources and chooses the next exact/recoverable action without pretending the discrete problem is convex.

### 2. Evidence and failed approaches

- HERMES and DSpark designs enumerate resource/cost terms but do not provide a common online selector.
- Static oracle and cache policies show that local hit-rate/frequency objectives can miss the global bottleneck.
- Queue flags and staging failures show hard discrete feasibility boundaries.

**Status: `CONJECTURED`; finite replay is `EXPERIMENTALLY TESTABLE`.**

### 3. Obstruction

A Lagrangian price can smooth resource tradeoffs but cannot by itself express hard fence, slot, route, and branch constraints. Discrete cache admission and state rollback create nonconvex jumps.

### 4. Candidate algorithm

At state `s`, maintain nonnegative shadow prices `lambda_r` for resource debt. For candidate action `a`:

```text
score(a|s) = expected_exact_value(a|s)
             + salvage_value(a|s)
             - sum_r lambda_r * demand_r(a)
             - risk(a) - rollback(a).
```

Update prices slowly from measured utilization/debt, but pass candidates through a hard viability filter first. The dual objective supplies a lower/upper bound only when the relaxed model's assumptions hold. Status: **`DERIVED UNDER ASSUMPTIONS`**.

### 5. Counterexamples

- A high-price action can be the only action that restores correctness; hard authority must override price.
- A cache admission has a discontinuous eviction effect not represented by a linear price.
- Oscillating price updates can cause alternating prefetch/eviction policies.

Status: **`COUNTEREXAMPLE FOUND`** against unfiltered price-only control.

### 6. Cheapest decisive falsification

Use an exhaustive finite resource/DAG simulator with 2–4 jobs, one cache, one fence, and one recovery action. Compare shadow-price policy to exact enumeration and test overload/oscillation. No GPU is needed.

Status: **`EXPERIMENTALLY TESTABLE`**.

### 7. Certificate design

Log candidate set, hard-filter reasons, prices, demands, selected action, realized value, resource debt, and fallback. Verify that no selected action violates capacity or authority and that price updates use realized, not predicted, utilization.

### 8. Mapping and justified experiment

This gives a possible bounded core for H15–H17/H22/H25/H26/H29/H34, REMORA reserve/flow, N05/N09, and N26. It is a simulator research item, not a live controller.
