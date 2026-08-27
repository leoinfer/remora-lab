# REMORA Discovery Notes

**Program:** REMORA / Local-AI discovery lab
**Audit date:** 2026-08-03
**Role:** theory-first synthesis; no production-code edits, no live GPU inference, no hardware-lock ownership
**Primary output set:** `REMORA_OPEN_PROBLEM_PORTFOLIO.md`, `REMORA_CROSS_IDEA_SYNTHESIS_GRAPH.md`, `REMORA_NEW_CONJECTURES.md`, `REMORA_COUNTEREXAMPLE_LEDGER.md`, `REMORA_FORMALIZATION_QUEUE.md`, `REMORA_DISCOVERY_TO_EXPERIMENT_HANDOFF.md`

## 1. Audit disposition

### 1.1 Required source that is absent

`LOCAL_AI_IDEA_RANKING_2026-08-03.md` was searched for in the local research tree and was not found. No ranking is reconstructed from memory and no ranking-dependent claim is made.

**Result: `BLOCKED` — missing requested source.**

The durable substitutes used only for name preservation and dependency context are:

- `[local path omitted]`;
- `[local path omitted]`;
- `[local path omitted]`;
- `archival/HERMES_V4_COMPLETE_IDEA_ATLAS.md` and its byte-identical `.txt` copy;
- the current vault state and evidence ledgers.

The missing ranking remains an explicit unresolved input. Its absence does not authorize changing the experiment queue.

### 1.2 Authority ordering discovered

The source graph contains a time/state inconsistency:

- `COMPLETE_RESEARCH_GRAPH.md` describes Qwen Q2 ordinary-layer transport as the active load-bearing line.
- The later `Current-State.md` and `Active-Experiment.md` identify **B0-repeatability-foundation** on dense Qwen3.6-27B Q8_K_XL as the only active live line.
- `REMORA_NEW_IDEA_MASTER_MANIFEST.md` and `tbeh_gate_review.md` explicitly defer Batch 1/TBEH behind B0.

For this discovery pass, the operational authority is therefore:

1. `Active-Experiment.md` and `Current-State.md` for what is live now;
2. the manifest and gate reviews for authorization;
3. the complete atlas/graph for preserved names and historical dependencies;
4. dated experiment/certificate artifacts for scoped evidence;
5. older HERMES/Q3 documents for historical mechanisms and failures.

**Result: `DERIVED UNDER ASSUMPTIONS` — temporal authority rule.** The older graph is not deleted or rewritten; it is treated as a historical dependency graph whose status fields require timestamp resolution.

### 1.3 Complete-source audit register

| Source family | Audited artifacts | Discovery use | Current status |
|---|---|---|---|
| Idea inventory | `COMPLETE_RESEARCH_ATLAS.md`, `COMPLETE_RESEARCH_GRAPH.md`, HERMES atlas | Preserve `H01–H39`, `N01–N26`, names, dependencies | `MACHINE-CHECKED` by preserved tables |
| REMORA | `REMORA_MASTER_PROMPT_2026-08-03-1.md`, `REMORA_NEW_IDEA_MASTER_MANIFEST.md`, `REMORA_TAIL_BOUNDED_ELASTIC_HORIZON.md`, vault TBEH notes | Batch gates, B0 block, TBEH specification | `MACHINE-CHECKED` / `BLOCKED` for live replay |
| RSSO | `RSSO_APPROXIMATE_SPARSE_ORACLE_DESIGN.md`, cost model, break-even simulation, foundation freeze, implementation gate, LayerPack, dependency graph, residency planner, skeleton search, vault RSSO notes | Dense state/rollback/break-even formulation | `MACHINE-CHECKED` static artifacts; live RSSO `BLOCKED` |
| PHASE / DSpark | HERMES `PHASE0_AUDIT`, publication `08_PHASE2_STATIC_RESEARCH`, phase2 README/results, Q8/Phase0 vault note | Static union, cache, saturation, branch and deadline evidence | `MACHINE-CHECKED` static/model-dependent; live validation gated |
| ExpertPack / LayerPack | Qwen XP0/XP1/XP2 docs, format certificate, source map; RSSO LayerPack and vault design | Physical packing and reversibility boundaries | XP1/XP2 `MACHINE-CHECKED` for limited byte scope; speed `BLOCKED` |
| Symbiote / MARC | MARC-Symbiote architecture, schemas/policy/audits, MARC V0 docs/reports, mini twin, Qwen→DeepSeek transfer | Semantic×hardware binding; negative keyword-only evidence | static `CONJECTURED` / V0 proxy `MACHINE-CHECKED` only as proxy |
| Dense-to-MoE | REMORA Batch 6 transfer section, `QWEN_TO_DEEPSEEK_TRANSFER.md`, atlas N-family mappings; no separate standalone dense-to-MoE design file was found | Transfer matrix and non-transfer constraints | `DERIVED UNDER ASSUMPTIONS`; no live transfer; standalone artifact `BLOCKED` |
| Old Q3 | original Q3 journey, Q3 architecture, Q3→Q8 gap tables, compact correctness, ReBAR/deferred staging, transport contracts | Reusable mechanisms and failure-induced invariants | historical evidence, scoped by model/config |
| Qwen current | Q2 one-token/four-layer/mixed/layer-1/eight-token/fence/full-core/energy certificates; dense stock/B0 certificates; raw JSON/telemetry | Separate exact Q2 MoE and blocked dense Q8 lanes | mixed: Q2 limited exact PASS; dense B0 `DEFER` |
| DeepSeek current | route-only `.tr`/JSONL traces, E1/E1A certificates, HERMES transport/certificate records | Route/margin/union evidence and invalid skeleton rows | model-specific, no Qwen transfer |
| Complete experiment queue | `COMPLETE_EXPERIMENT_QUEUE.md` | Preserve `E001–E096`, dependency gates, active Q2/E021 boundary, and static-preparation limits | `MACHINE-CHECKED` queue read; no experiment run |
| Evidence control | `Current-State`, `Active-Experiment`, Claims/Failure/Measurement/Certificate ledgers | Claim boundaries and invalidation rules | `MACHINE-CHECKED` as archival policy |

### 1.4 Trace and certificate audit highlights

**DeepSeek traces.** `dsv4-16token-trace.jsonl` contains 731 records for 17 token states across 43 layers, with six expert IDs and six router weights per record. `dsv4-first-trace.jsonl` is a shorter route trace. `0731_q8_m1_baseline.tr` is route-only; `0731_q8_128tok.tr` is explicitly truncated/stale and not usable. The six HERMES trace files contain route sequences at much larger size but not an end-to-end accepted-token certificate.

**Qwen Q2.** The one-token, four-layer, mixed layer, exceptional layer-1, eight-token persistence, fence/thrash, and full-core artifacts show exact source/slot byte identity and zero fail-closed counters for their stated scopes. The Q2 energy artifact is GPU-only and explicitly does not promote a winner.

**Dense Qwen3.6-27B Q8.** The primary 256-token MTP-off/on row is a single matched control. Fresh B0 A/B/A/B/A/B gives `0/3` clean pairs, output classes `A/B/C/A/B/C`, host-buffer-size mismatches, and memory drift. The raw JSON carries identical declared model/prompt/seed settings but divergent token hashes. This is the key live blocker.

**HERMES certificates.** The `CTRL-Hello` and `E1-b0_k43` certificates distinguish numerical closeness from bit identity. `E1-b4_k43` and repeated E1A Q4 rows are correctness failures because of staging failures, not evidence that Q4 quantization itself is bad. One requested negative certificate path was absent (`E1-b0_k43-NEGATIVE/cert.json`), so absence is recorded rather than inferred.

**Result: `MACHINE-CHECKED` for artifact-scope observations; no performance claim is made here.**

## 2. Core discoveries

### D-01 — The real common object is a verified-token state transition

Apparently separate ideas—TBEH, PHASE, RSSO, REMORA Portion/Reclaim, P1/P2/P4/P6, MARC-OS, and the HERMES GPS—are all policies over a state transition:

```text
(state, candidate work, resource state)
    -> (accepted prefix, authoritative state, retained artifacts, resource debt)
```

The correct objective is not “tokens drafted,” “router hits,” “bytes reduced,” or “kernel work.” It is the net value of an **exactly committed accepted prefix** after draft, target, transfer, memory, rollback, contention, and reserve costs.

**Result: `DERIVED UNDER ASSUMPTIONS`.** The equivalence is a common formalization, not proof that the implementations are interchangeable.

### D-02 — Exactness is a dependency-closure property, not a token-match property

The old Q3 stale-arena failure, Qwen Q2's missing graph dependency, dense Qwen's repeatability failure, and RSSO's recurrent-state warning all point to the same rule:

> A result is exact only when all transitive inputs that can affect the committed output and authoritative state are fixed, ordered, and certified.

A greedy token match is weaker than tensor/logit parity; tensor parity at one layer is weaker than state parity across a recurrent block; one-layer exact transport is weaker than full-graph exactness.

**Result: `CONJECTURED` as a project-wide theorem pending a formal interface checker; individual dependency examples are `MACHINE-CHECKED`.**

### D-03 — Predictor value is tail-weighted, not classification-weighted

The PHASE/Route Scout/DSpark family often starts from prediction accuracy. Static PHASE results show why this is insufficient: miss-conditioned F1 is much worse than overall F1, prior-token retention can be a no-op, and the saturated-regime simulation makes even an oracle controller stall when physical demand remains too high.

The correct target is:

```text
avoided critical-path cost per predicted byte
```

or, for residency:

```text
value-weighted miss recall
= avoided-cost mass on correctly prepared misses
  / total avoided-cost mass
```

**Result: `DERIVED UNDER ASSUMPTIONS`; the static trace findings are `MACHINE-CHECKED` within their small/model-dependent traces.**

### D-04 — Resident transport and semantic memory are two instances of a typed associative map

H10's physical expert atlas maps `(layer, expert_id)` to a slot. N25/MARC-Symbiote maps semantic references to context/module artifacts. They are not the same artifact, but they share the same hard requirements:

- stable key;
- content/version identity;
- owner/lifetime;
- eviction/holding cost;
- invalidation trigger;
- fallback authority.

This gives a principled bridge between ExpertPack, dependency-versioned cognition, semantic fingerprints, and hardware fingerprints without conflating physical slots with semantic memory.

**Result: `CONJECTURED` — a unifying data model, not an established implementation.**

### D-05 — A "wavefront" is not exact merely because weights are resident

RSSO LayerPack and H09 describe processing multiple candidate positions while a layer/pack is resident. The dense Qwen source has 48 recurrent layers and 16 full-attention layers. For recurrent layers, later-position state depends on earlier-position state; a layer-stationary schedule requires either an exact associative scan or a schedule that respects the recurrence. For attention, all-position grouping still requires the preceding layer's hidden inputs and causal masking.

**Result: `DERIVED UNDER ASSUMPTIONS`; live wavefront exactness is `BLOCKED`.**

### D-06 — Hardware phenotype must compile jointly with workload phenotype

A hardware fingerprint alone cannot select a universal plan. The same RX 9060 XT has different useful policies for DeepSeek MXFP4, Qwen Q6/Q8, a dense Q8 model, and a route-stable versus cold workload. The compiled plan must be keyed by at least:

```text
hardware identity × model/source identity × workload/state class
```

**Result: `DERIVED UNDER ASSUMPTIONS`.** This rejects a hidden assumption in generic startup autotuning while preserving H32/H37.

### D-07 — Existing low-bit formats are not a refinement lattice

The compression architecture explicitly records that independent Q3_K and Q4_K streams are not nested: block geometry, scale packing, and affine conventions differ. Therefore H05/manifest idea 27 cannot be implemented by treating one existing GGUF as a direct lower-bit prefix of another. A true refinement path needs a new sidecar/anchor+residual/bit-plane encoding or an authority reload.

**Result: `MACHINE-CHECKED` as a format/architecture audit; residual-lattice implementation `CONJECTURED`.**

### D-08 — DSpark has a four-layer evidence gap

The DSpark audit distinguishes official tensor presence, converter preservation, loader support, and execution. Official DeepSeek `mtp.*` presence is not current GGUF preservation, and current HERMES has neither loading nor execution. This is stronger than the weaker statement “DSpark is absent from official 0731,” which is disallowed by the Claims Ledger.

**Result: `MACHINE-CHECKED` source audit; restoration `BLOCKED`.**

## 3. Contradictions and incompatible assumptions

1. **Q2 versus dense B0 active state.** Qwen Q2 evidence is not authorization for live dense RSSO or Batch 1. The older graph's active label is stale relative to `Active-Experiment.md`.
2. **Q2 exact transport versus compact skeleton.** N01 copies unchanged Q6_K/Q8_0 authority bytes. H03 changes representation. Exact Q2 transport cannot be used as skeleton fidelity evidence.
3. **DeepSeek Q8 compact versus Q3 bit identity.** The HERMES Q8 compact path is numerically close, not bit-exact; Q3 CPU compact identity does not transfer automatically.
4. **Single-row Qwen MTP parity versus repeatability.** A primary matched row can coexist with a failed repeatability gate. It remains a scoped control, not a baseline certificate.
5. **Tail-bound stop versus single-step cost.** TBEH's omitted-tail test is only sound when its cost comparison prices the complete continuation or has a proven lower cost for every continuation. Comparing a gross tail bound to one arbitrary next-step cost is not a universal optimality theorem.
6. **Layer-stationary wavefront versus recurrent state.** Loading a LayerPack once is a transport statement; exact multi-position execution additionally needs a state schedule or scan proof.
7. **F1 versus critical-path value.** A high overall route-prediction score can be irrelevant when expensive misses are the hard cases.
8. **Reduced bytes versus speed.** Qwen Q2 changed-slot bytes fall while repeated full-core C is slower and has higher gross GPU J/token than F. This is an observed counterexample to byte-only promotion.
9. **Semantic fingerprint versus authority.** MARC-Symbiote explicitly says a fingerprint can select a body but cannot gate correctness; keyword/hash V0 is not semantic success.
10. **Local certificate versus end-to-end certificate.** Qwen one-layer exactness and HERMES upload replay do not certify all-layer graph/state/throughput behavior.
11. **Asynchronous API calls versus overlap.** Queue flags and deferred submission can still serialize behind fences or fail at the driver; overlap must be measured on the critical path.
12. **Qwen-to-DeepSeek transfer.** Field schemas and analysis methods may transfer; dimensions, route IDs, thresholds, state semantics, expert sizes, and quality results do not.
13. **Branch probabilities versus branch DAG.** PHASE branch mass must be conditional and shared-prefix-aware. Summing marginal branch probabilities can double count both probability and cost.
14. **MARC V0 proxy versus MARC-X/Symbiote.** Token-budget policy economics are not evidence for internal semantic routing, residency control, or a temporary execution body.
15. **R4I8/R5I8/R6I8 names.** Only R4I8 has a structural implementation artifact. The other two are placeholders, not a format ladder result.

## 4. Failed approaches that remain reusable

| Failed route | Why it failed | Reusable remainder |
|---|---|---|
| Q3 CPU-arena callback population before scheduler copy | graph leaf copy occurred before callback; stale/zero arena | explicit graph dependency, owned staging, fence/epoch contract |
| Rank-indexed slot upload | router rank is not physical slot identity | per-layer original-ID→slot map and source/slot hash certificate |
| Early staging reset / NT-store publication | deferred consumers or old stores outlived the apparent completion | monotonic epochs, `_mm_sfence`, release/acquire publication |
| Q4×43 skeleton rows with staging failures | resource invalidation masqueraded as model-quality failure | fail-closed capacity checker and distinct quantization/transport conclusions |
| Vulkan argsort near-tie path | selection order differed at boundaries | route identity/margin is part of exactness; CPU argsort control |
| Q6 predictor fast path | KL/PPL/top-token gates failed; output was garbage | direct-logit gates before performance, exact-tail quarantine |
| MARC keyword/hash fingerprint | not semantic, no internal model signals or hardware state | architecture vocabulary and negative baseline |
| Previous-token/short-history prefetch | expensive miss F1 low; previous retention was effectively no-op | miss-conditioned/value-weighted predictor evaluation |
| Oracle prefetch in saturated regime | perfect knowledge did not reduce stall when demand exceeded byte roofline | reduce demand/churn before improving prediction |
| Prompt-cache assumptions | `cache_prompt=false` did not eliminate same-process slot state | cache key/state audit and fresh-process controls |
| Qwen Q2 byte saving as speed proxy | lower bytes did not yield a repeated speed/energy win | separate transport, time, and energy certificates |
| Risky graphics/transfer queue overlap | driver/context loss in hardware audit | resource-topology profile and fail-closed queue admission |

## 5. New theory agenda produced by the audit

The detailed derivations and counterexamples are in the companion files. The highest-value formal objects are:

1. **Elastic-horizon optimal stopping with tail bounds** — `OP-01`.
2. **Accepted-token multi-resource roofline** — `OP-02`.
3. **RSSO causal-state wavefront theorem and break-even** — `OP-03`.
4. **Delta-certified local argmax versus sequence exactness** — `OP-04`.
5. **Merkle dependency closure for exact cognition reuse** — `OP-05`.
6. **PHASE branch-DAG economics** — `OP-06`.
7. **Resource-constrained complementarity scheduling** — `OP-07`.
8. **Value-weighted predictive residency and Belady bounds** — `OP-08`.
9. **Value-of-computation conservation ledger** — `OP-09`.
10. **Hardware phenotype compilation** — `OP-10`.
11. **Proof-carrying composition of local certificates** — `OP-11`.
12. **Shadow-price control for heterogeneous resources** — `OP-12`.

## 6. Negative result discipline

- No current document in this discovery set establishes a new throughput result.
- No reduced-byte result is rewritten as speed.
- No calibrated confidence is rewritten as an exactness certificate.
- No Qwen result is rewritten as a DeepSeek result.
- No B0 diagnosis is promoted to a Vulkan/RADV root-cause proof.
- No TBEH/RSSO/PHASE live controller is authorized.
- No production code or hot path was modified by this discovery pass.
