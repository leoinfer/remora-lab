# REMORA new-idea master manifest

**Program:** Qwen dense → REMORA

**Authority:** `[archival master prompt omitted] (1).md`

Authority SHA-256: `b2d11473b8d9693dc3c58dd363cbb7c9f6af1616f277af9bdbb4e64595573e8f` (public source fingerprint; local path omitted)

**Host:** frozen Qwen3.6-27B Q8_K_XL, no weight changes, no training initially,
full target remains authoritative. Batch 0 currently has verdict
**DEFER — REPEATABILITY BLOCKER**; no Batch 1 live work is authorized.

## Idea index

| # | Name | Batch | Form | Frozen-host compatible | Training initially | Dependencies | Cheapest falsification | Status / verdict |
|---:|---|---:|---|---|---|---|---|---|
| 1 | Elastic MTP generation depth | B1 | exact/hybrid controller | yes | no | B0, MTP traces | replay marginal draft cost/value | DEFERRED behind B0 |
| 2 | Elastic verification depth / continuous horizon | B1 | exact controller | yes | no | B0, target verification traces | replay discrete horizon economics | DEFERRED behind B0 |
| 3 | Neuralink/REMORA future packets | B1 | predictive controller | yes | no | B0, feature traces | offline feature leakage/coverage test | DEFERRED |
| 4 | Multi-drafter fusion | B1 | hybrid draft | yes | no | B0, isolated draft traces | teacher-forced agreement/cost replay | DEFERRED |
| 5 | PHASE outcome-tree prediction | B1 | exact target plus speculative prep | yes | no | B0, branch traces | B=1/2/4/8 coverage replay | DEFERRED |
| 6 | RSSO resident skeleton + streamed oracle | B2 | exact/hybrid | yes | no | B0, target batching/state | dependency/byte lower bound | DEFERRED |
| 7 | Layer-stationary speculative wavefront | B2 | exact/hybrid | yes | no | B0, LayerPack and K batching | weight-reuse simulator | DEFERRED |
| 8 | Small speculative wavefront tree | B2 | hybrid/approximate draft | yes | no | B0, branch traces | offline tree cost/coverage | DEFERRED |
| 9 | Latent Inertial Drafting | B2 | approximate draft, target exact | yes | no | B0, teacher-forced state traces | finite-difference agreement | DEFERRED |
| 10 | Dense organ map by value per byte | B2 | exact/hybrid search | yes | no | B0, block ablations | static dependency/cost bound | DEFERRED |
| 11 | REMORA Portion | B3 | exact controller | yes | no | B0, salvage/waste traces | policy replay; TBEH tail risk | DEFERRED |
| 12 | REMORA Reclaim | B3 | exact artifact policy | yes | no | B0, validity traces | avoided-cost ledger replay | DEFERRED |
| 13 | Computational refrigerator / artifact provenance | B3 | exact metadata | yes | no | B0, state identity | validity-rule audit | DEFERRED |
| 14 | Value-weighted salvage cache | B3 | exact policy | yes | no | B0, reuse traces | value-per-byte replay | DEFERRED |
| 15 | Waste Ledger / circular efficiency | B3 | accounting | yes | no | B0, all cost traces | denominator/completeness audit | DEFERRED |
| 16 | Tiered Inference Reserve | B4 | exact policy | yes | no | B0, artifact lifetime | reserve decay replay | DEFERRED |
| 17 | Reserve mobilization | B4 | exact policy | yes | no | B0, deficit traces | prevented-work accounting | DEFERRED |
| 18 | Moving maintenance setpoint | B4 | exact controller | yes | no | B0, stable baseline | setpoint replay | DEFERRED |
| 19 | Uncertainty-adjusted safe surplus | B4 | exact controller | yes | no | B0, calibration traces | calibration stress replay | DEFERRED |
| 20 | Fast/slow adaptation clocks | B4 | exact controller | yes | no | B0, temporal traces | clock-ablation replay | DEFERRED |
| 21 | Portable parasitic neural hypervisor | B5 | host ABI/controller | yes | no | B0–B4 gates | interface/state audit | DEFERRED |
| 22 | REMORA Link / host receptor | B5 | exact ABI | yes | no | B0, state API | ABI round-trip test | DEFERRED |
| 23 | REMORA Morph | B5 | measured phenotype | yes | no | hardware microbenchmarks | fingerprint repeatability | DEFERRED |
| 24 | REMORA Flow | B5 | exact scheduler | yes | no | transfer/compute traces | critical-path replay | DEFERRED |
| 25 | REMORA Verify / progressive escalation | B5 | exact verifier | yes | no | rollback/state gates | fail-closed rollback test | DEFERRED |
| 26 | Dependency-versioned cached cognition | B6 | exact reuse / approximate draft | yes | no | B3 provenance | dependency mismatch replay | DEFERRED |
| 27 | Delta-Certified skipping | B6 | exact bound / target authority | yes | no | B0, logit/error traces | bound feasibility replay | DEFERRED |
| 28 | Native hardware-morphic Symbiote | B6 | architecture | yes | no | B5 phenotype | design consistency audit | DEFERRED |
| 29 | Universal learned parasite / Neuralink MTP | B6 | future training ladder | partly | later/explicit approval | B5, explicit authorization | training-free V1 boundary | DEFERRED |
| 30 | Dense-to-MoE translation | B6 | transfer architecture | yes | no | Qwen dense evidence, separate MoE gate | transfer matrix audit | DEFERRED |
| TBEH | REMORA Tail-Bounded Elastic Horizon | cross-cutting B1/B3/B6 | theoretical/offline controller | yes | no | B0, real MTP traces, held-out calibration | fixed-vs-threshold-vs-EV-vs-TBEH replay | **THEORETICAL / OFFLINE-REPLAY ONLY** |
| PFM | Progressive Future Materialization (PFM-A / PFM-B) | cross-cutting B1-B6 | exact/hybrid temporal materialization plus future-state research | PFM-A yes; PFM-B no | PFM-A no; PFM-B later | B0, state/cost/trace closure, oracle economics, exact fallback | oracle versus sequential offload/prefetch/speculation/SpecExec with full contention and state costs | **PFM-A REJECTED AS DISTINCT MECHANISM; PFM-B DEFERRED** |

## TBEH detailed record

- **Name:** REMORA Tail-Bounded Elastic Horizon (TBEH).
- **Batches:** Batch 1 Ideas 1, 2 and 5; Batch 3 REMORA Portion; Batch 6
  Delta-Certified skipping.
- **Falsifiable hypothesis:** on held-out real MTP traces, a conservative
  omitted-tail bound can select a smaller or larger speculative horizon with
  lower total generation/verification/memory/rollback/contended cost than fixed
  depth and simple confidence thresholds, while preserving exact target output.
- **Physical bottleneck attacked:** speculative waste, repeated target traversal,
  synchronization, cold LayerPack movement, latency, energy and memory
  opportunity cost; not model arithmetic alone.
- **Applicability:** dense, hybrid and later MoE; first evidence must be dense
  Qwen MTP and must not be transferred to DeepSeek.
- **Form:** exact target verification with a predictive/hybrid horizon controller;
  future rare-event predictors remain approximate drafts only.
- **Frozen-host compatible:** yes.
- **Training required initially:** no; logged features, calibrated empirical or
  conformal bounds, and offline replay only.
- **Dependencies:** B0 exact repeatability; real MTP traces; temporal held-out
  splits; cost/energy/state/rollback measurements; target-authoritative output.
- **Cheapest falsification:** use retained real traces to compare fixed depth,
  simple threshold, expected-value marginal control, TBEH and post-hoc oracle;
  reject if TBEH has worse held-out regret after all costs or its bound coverage
  fails.
- **Current evidence:** theoretical only. No TBEH trace replay has run; B0 is
  correctness-blocked.
- **Implementation status:** design artifacts created; no runtime code.
- **Vault links:** [[Architecture/REMORA-Tail-Bounded-Elastic-Horizon]],
  [[Experiments/REMORA-TBEH]], [[Evidence/Hypothesis-Ledger]],
  [[Evidence/Claims-Ledger]].
- **Verdict:** DEFERRED — B0 prerequisite; do not begin live Batch 1.

## PFM detailed record

- **Name:** Progressive Future Materialization.
- **Family:** `PFM-A` frozen-model temporal staging and `PFM-B` trainable
  future-state materialization.
- **Central hypothesis:** future work can progress from a future hypothesis to
  an execution plan, partially prepared computation and finally an exact
  verified state before the next token is committed, if the physical value
  exceeds preparation, promotion, state, verification, contention and waste
  costs.
- **Authoritative state:**
  `X_t=(h_t,K_t,R_t,G_t,M_t,E_t)` for exact committed prefix, KV, recurrent/
  architecture state, sampler/RNG state, model/build/runtime identity and
  execution epoch/namespace. Future hidden/KV states are candidate-path
  specific; no generic exact future state is assumed.
- **Artifact classes:** P0 planning artifacts, P1 path-exact candidate-state
  artifacts, and P2 approximate predictive artifacts. P2 is advisory only in
  strict mode. P1 reuse requires candidate-prefix, dependency, epoch and
  expiry closure.
- **State machine:** `CREATED -> SKETCH -> BLUEPRINT -> BUILD -> VERIFIED ->
  COMMITTED`, with explicit `EXPIRED`, `DEMOTED`, `REJECTED`, `RECLAIMED` and
  `INVALIDATED` paths. (The canonical spelling is `DEMOTED`; the architecture
  document is authoritative.) Failed or stale paths fall back to the exact
  target and cannot mutate committed state.
- **Physical bottleneck:** cold out-of-core weights/experts, tier traffic,
  queueing, promotion and exact target verification.
- **Applicability:** dense, hybrid and MoE with model-specific state/route
  adapters. PFM-A is frozen-host compatible; PFM-B requires later training
  authorization.
- **Form:** exact/hybrid controller above PHASE/TBEH/REMORA Flow/RSSO/ExpertPack/
  Verify/Causal-Closure/ENSM/Reclaim/Waste Ledger and resource pricing.
- **Default geometry:** width one. A second path requires positive full-cost EV;
  entropy alone is not a branch gate.
- **Oracle gate:** perfect future knowledge must beat the strongest baseline by
  at least 10% throughput without meaningful energy regression, or 10% lower
  joules/token without meaningful throughput regression, after all costs.
- **Additional gates:** failed background work must add at most 2% critical-path
  latency; promotion must satisfy `p_used*C_saved > C_promotion +
  C_wasted_promotion`; median residual fraction `rho < 0.5` is minimally
  interesting and `< 0.25` is stronger; PFM-A must beat sequential offload,
  predictive prefetch, ordinary speculation and SpecExec under equal limits.
- **Cheapest falsification:** CPU/offline oracle replay with exact future tokens,
  routes, use/staleness, residual costs and measured capacities, comparing the
  listed baselines and PFM-A widths. This is an upper bound, not a runtime
  result.
- **Current evidence:** MACHINE-CHECKED formal invariants, STATIC prior-art
  audit and SIMULATED fair oracle sweep. Across 50 evidence/project-parameter
  rows, PFM-0 and positive promotion EV both have zero passes. No PFM economics,
  future-state compressibility, cheap residual correction, exact multi-position
  execution or runtime benefit is established.
- **Implementation status:** no runtime code; PFM-A is rejected as a distinct
  mechanism and only its static state/accounting overlay is retained. PFM-B is
  blocked by missing compressibility/correction/calibration evidence and lacks
  training authorization. B0 remains closed; no PFM live implementation is
  allowed.
- **Vault links:** [[Architecture/Progressive-Future-Materialization]],
  [[Experiments/Progressive-Future-Materialization]],
  [[Evidence/Validation-Protocol]], [[Evidence/Measurement-Ledger]].
- **Canonical specification:**
  `[local path omitted]`
  contains the exact packet fields, transitions, cost equations, invariants,
  oracle protocol and kill criteria.
- **Verdict:** **REJECT PFM-A** as a distinct incremental execution mechanism
  in the established project-parameter envelope; **DEFER PFM-B** pending
  future-state compressibility, correction-cost, quality and calibration
  evidence plus explicit training authorization. Retain only the static
  accounting/state overlay; no PFM hot path is authorized.
