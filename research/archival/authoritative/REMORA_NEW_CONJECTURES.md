# REMORA New Conjectures

These are stronger cross-family ideas not present as complete named mechanisms in the preserved atlas. They are deliberately bounded and falsifiable. None is a performance claim.

## C-01 — Causal-Closure Cache (CCC)

**Status: `CONJECTURED`**

### Claim

Exact reuse is best implemented as a content-addressed cache over the **transitive causal closure** of an artifact, not over prompt text, token ID, model hash, or semantic fingerprint alone.

```text
key = MerkleRoot(
  model/source roots,
  token/input roots,
  recurrent/KV state root,
  route/slot/pack root,
  precision/backend/driver contract,
  graph/function version,
  sampler/RNG state,
  external-state version
)
```

### Why stronger

It joins REMORA Reclaim/Refrigerator, dependency-versioned cognition, H10 stable slots, ExpertPack/LayerPack, and MARC-Symbiote hardware state into one exactness primitive. Approximate cache hits may use a relaxed key only as drafts.

### Counterexample search

Delete each key field in turn and search for a pair of artifacts that collide but differ in tensor/state/output. The dense Qwen prompt-cache controls and HERMES stale-arena history are seed counterexamples.

### Cheapest decisive test

CPU/property-based Merkle-key checker with synthetic recurrent state and mutable pack/driver versions. No model inference required.

### Certificate

`CCC-v1` records closure members, root, artifact hash, validation rule, and exact/approximate mode. The checker requires a miss on every changed causal leaf.

### Affected ideas

Manifest `12–15`, `26`; H10, H27, H30, H33; N04/N05/N09/N25; ExpertPack and LayerPack.

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

## C-03 — Certified Approximation Lattice (CAL)

**Status: `CONJECTURED`**

### Claim

Approximation mechanisms should be ordered by a refinement relation rather than treated as unrelated modes:

```text
cheap anchor
  <= residual refinement
  <= delta-bound local result
  <= verified target result
  <= exact committed state
```

A refinement may be used only if it consumes a certificate that bounds the remaining uncertainty. A node can fall back upward but cannot silently fall sideways into another approximate node.

### Why stronger

It unifies H03 compact skeleton, H04 progressive widening, H05 residual tiles, H07 margins, H23 purification, H24 protection, manifest `27 Delta-Certified skipping`, N17 cascade correction, and N19–N21 format variants.

### Counterexample search

Build a two-layer lattice where local errors are small but state divergence differs. Try to compose a local token certificate without a state certificate; the checker must reject the edge.

### Cheapest decisive test

Finite symbolic lattice checker with explicit error intervals, state roots, and fallback edges. No GPU.

### Certificate

Each node has authority root, uncertainty interval, state-equivalence status, and allowed consumers. Edges state whether they are exact refinement, verified proposal, or approximate transition.

### Affected ideas

H02–H08/H23/H24/H28/H33; manifest `1`, `2`, `6–10`, `25`, `27`; N17/N19–N23.

---

## C-04 — Epoch-Namespace State Machine (ENSM)

**Status: `CONJECTURED`, motivated by `COUNTEREXAMPLE FOUND`**

### Claim

Every mutable transport/state object should be namespaced by an epoch that includes both logical decode epoch and runtime allocation/graph generation. A payload, slot, staging slice, cache artifact, or recurrent snapshot from epoch `e` cannot satisfy a consumer requiring `(e, generation)` unless its certificate explicitly proves compatibility.

```text
namespace = (model_root, graph_generation, decode_epoch, sequence_id)
```

### Why stronger

It combines Q3 stale-arena repair, HERMES NT-store publication, Qwen Q2 fence/thrash, dense Qwen host-buffer/memory drift, and dependency-versioned cognition. A monotonic decode epoch alone may be insufficient when allocator/graph generation changes without a new logical token.

### Counterexample search

Replay a same-process cache/graph-reuse sequence with an unchanged token stream but changed allocation generation. Any reused buffer without a generation match is a deliberate stale-state injection.

### Cheapest decisive test

Static source-level invariant checker over artifact metadata and a finite scheduler model.

### Certificate

Every copy, state read, publication, and cache hit carries source and consumer namespaces plus fence sequence. Fail closed on unknown generation.

### Affected ideas

H02/H10/H13/H14/H29/H33; manifest `13`, `22`, `25`, `26`; N01/N05/N09/N26.

---

## C-05 — Union-Aware Residual Cache (UARC)

**Status: `CONJECTURED`**

### Claim

The right cache unit for speculative MoE blocks is not an individual expert or token, but a `(layer, expert, residual/refinement class)` item scored by:

```text
expected avoided critical-path cost
  + probability of reuse after rejection
  - bytes held
  - eviction opportunity cost
  - validation cost.
```

The cache should prefer residual tiles or refinement classes that serve many branches, even when their parent expert is not the most frequent item.

### Why stronger

It joins H05 residual tiles, H08 future canvas, H10 atlas, H21 compound/isolation, H27 salvage, OP-08 value-weighted residency, and PHASE branches. It attacks union growth rather than only whole-expert hit rate.

### Counterexample search

Generate a route tree where the most frequent expert is branch-specific and a less frequent residual tile is shared across all branches. Compare LRU/expert-frequency/UARC.

### Cheapest decisive test

CPU trace replay with existing route unions and synthetic residual sizes; exact output is not needed to falsify the economic policy.

### Certificate

Record parent artifact/root, residual use count, branch coverage, bytes, holding time, avoided reload cost, and observed salvage use. Credit only realized reuse.

### Affected ideas

H05/H08/H10/H11/H21/H27/H30/H38; manifest `5–8`, `11–15`; N01/N18/N23.

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

## C-08 — Exactness Requires a State Boundary, Not Just an Authority Endpoint

**Status: `DERIVED UNDER ASSUMPTIONS`**

### Claim

Every exact/verified module interface must expose both an authority output and an authority state boundary. An interface that returns only logits/tokens cannot support exact continuation for a stateful host.

### Why stronger

It is a direct synthesis of RSSO recurrent state, MARC-Symbiote refresh, Qwen B0 drift, and HERMES fence/state contracts.

### Counterexample search

Use two states with equal current argmax token but different next-step logits. Any token-only interface falsely claims continuation equivalence.

### Cheapest decisive test

Two-state finite recurrence exhaustive checker.

### Certificate

`output_hash`, `state_hash`, `state_schema_version`, `commit_prefix`, `discard_suffix`, and authority root are mandatory.

### Affected ideas

H02/H08/H09/H13/H25/H29/H33; manifest `6–10`, `22`, `25–27`; N09/N18/N24/N25.

---

## C-09 — Phenotype Compiler Must Emit a Safe Region, Not One Optimum

**Status: `CONJECTURED`**

### Claim

A startup autotuner should compile a set of safe operating regions with fallback transitions, not one allegedly optimal configuration. The region is indexed by workload/state class and bounded by intervals.

### Why stronger

It combines H19 lanes, H20 fatigue, H29 viability, H32 autotuner, H37 configuration leads, and current queue/clock failures.

### Counterexample search

Vary context/KV, route churn, or power/driver state while holding hardware fixed. A single fixed optimum should violate either capacity, latency, or correctness reserve in at least one class.

### Cheapest decisive test

CPU phenotype planner with synthetic contexts and measured constants; require safe fallback at every region boundary.

### Certificate

Region predicates, lower/upper capacities, hysteresis, transition policy, profile identity, and fallback plan.

### Affected ideas

H16/H19/H20/H29/H32/H37; manifest `19`, `20`, `23–25`, `28`; N05/N09/N22/N23.

---

## C-10 — Certificate-First Autonomous Experimentation

**Status: `CONJECTURED`**

### Claim

An autonomous local research loop should be allowed to propose or run an experiment only when it can construct the verifier schema and failure-preservation path before execution. If no certificate design exists, the experiment is not yet executable research.

### Why stronger

It makes N26 a formal gate rather than a queue runner. It joins H33, Current-State, the failure ledgers, and the user instruction that empirical correlation cannot become proof.

### Counterexample search

Give a queue item with a speed metric but no exactness/timing denominator. The system must return `BLOCKED`, not run and average.

### Cheapest decisive test

CPU-only queue slice with one valid static checker, one missing artifact, and one injected failure. Require immutable evidence outputs and no promotion.

### Certificate

Experiment manifest, preconditions, allowed resources, raw artifact paths, verifier version/hash, decision, and invalidation reason.

### Affected ideas

N26, H02/H14/H29/H33, every manifest batch and the seven handoff packages.
