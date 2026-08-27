# Complete Experiment Queue

This is the complete dependency-aware queue, not the earlier 12-lane shortlist. It preserves 96 individually falsifiable experiments across the 65 named idea families in the atlas.

## Queue contract

- **Exact decomposed experiments:** **96**, IDs `E001`–`E096`, with no gaps.
- **Preserved families:** **65** strict named families, or **56** top-level bundles when the 17 broader requested bullets are not decomposed.
- **One load-bearing experiment at a time.** Only the current Qwen Q2 item owns the GPU/build path.
- Every hardware result gets a source/model/binary/environment hash, correctness class, timing-validity label, raw log, and certificate.
- No heuristic prefetch, fingerprint training, MTP/DSpark runtime, long timeout, dataset collection, or multi-layer expansion is active during Q2.
- A simulator/trace result is never reported as an end-to-end speed result.
- A failed gate stops its descendants and remains in the ledger.

## Status labels

- **ACTIVE:** current load-bearing line.
- **READY-STATIC:** can be prepared or tested without taking the load-bearing GPU slot.
- **GATED:** protocol is ready but waits for an upstream gate.
- **DESIGN:** retained hypothesis; protocol still needs implementation.
- **MUTEX:** competing representation/policy; compare under equal budget, do not combine claims.
- **BLOCKED:** missing artifact or unresolved definition.

---

# Group 0 — controls, evidence, and static queue integrity (`E001`–`E012`)

| ID | Experiment | Families | Dependency | Smallest protocol and falsifier | State |
|---|---|---|---|---|---|
| E001 | Source-of-truth mode audit | H02 | model/control hashes | Run canonical, transport-only, close-logit, and degraded paths; falsify if reports collapse them into one correctness label. | READY-STATIC |
| E002 | Frozen Qwen baseline and environment freeze | H01,N03 | model path and branch | Freeze Qwen GGUF SHA, commit, NGL=20, batch/KV/env, greedy output and VRAM/GTT; any mixed configuration invalidates later A/B. | READY-STATIC |
| E003 | Hardware/traffic control ledger | H37,H38 | E002 | Measure NGL=20 Vulkan, GTT, warm/cold decode, queue flags off, and stage timings; falsify a proposed gain if the control is not repeated. | GATED |
| E004 | Certificate fault injection | H33 | H02 | Inject wrong ID, slot permutation, skipped upload, stale payload, malformed marker, and parser error; verifier must reject every case. | READY-STATIC |
| E005 | Qwen tensor geometry and authority index | N01,N02 | E002 | Enumerate 40×256 routed tensors, types, `ggml_nbytes`, host/mmap access, and normal/layer-1 strides; any mismatch blocks Q2. | READY-STATIC |
| E006 | Qwen NGL=20 VRAM/GTT placement control | N02 | E002,E005 | Record 14,385.22-MiB-class Vulkan control and GTT before/during/after decode; fail if the control changes across trials. | GATED |
| E007 | DeepSeek E1A corrected closure | H35,H33 | DeepSeek lock and current E1A run | Finish corrected staging/epoch-order run, zero staging failures, one-slot oracle, and relabel invalid b4_k43; any nonzero failure invalidates that row. | GATED/SEPARATE GPU LINE |
| E008 | Clean DeepSeek E2 non-MoE floor | H36 | E007, route-stable control | Measure attention/KV slope at controlled context with no I/O/build contamination; a high-RMS fit remains low confidence. | GATED |
| E009 | Accepted-token roofline refresh | H38 | E003 | Compute SSD/H2D/VRAM budgets from measured bytes and accepted tokens; falsify any target that exceeds the physical budget without a new source of bytes. | READY-STATIC |
| E010 | Telemetry schema and overhead smoke | H14,N05 | E002,E005 | Add/parse route, entropy, slot, staging, and timing fields in off/summary/full modes; fail if trace changes IDs/routes or materially adds time. | READY-STATIC |
| E011 | Applicability/transfer matrix | H02,N03,N04,N05 | atlas source ingestion | Mark each mechanism Qwen/DeepSeek/both/future MARC-native and direct/adapter-only; falsify any cross-model claim lacking topology adapter. | READY-STATIC |
| E012 | Queue dependency and omission lint | N26,H39 | atlas/graph/queue | Verify all 65 family IDs and all 96 experiment IDs occur, no cycle is runnable without a gate, and no 12-lane alias hides a name. | READY-STATIC |

# Group 1 — active Qwen Q2 transport (`E013`–`E022`)

`N01 — Qwen compact expert transport` is the active completed Q2 line. E013–E021 have exactness/transport/placement evidence; E022 and all wider lines remain gated. No speed claim is attached to the PASS rows.

| ID | Experiment | Families | Dependency | Smallest protocol and falsifier | State |
|---|---|---|---|---|---|
| E013 | Q2 manager compile/API smoke | N01,H10,H13 | E005, imported Vulkan hooks | Build isolated branch, resolve API/compile errors, and instantiate dynamic manager without graph use; fail closed on allocation/geometry errors. | **PASS** — `29e0b3904` |
| E014 | Q2 selected-layer CPU/mmap authority | N01 | E013,E005 | Force one ordinary layer's full expert authority to host/mmap and prove gate/up/down pointers, types, strides, and source spans; any device authority or missing span blocks. | **PASS** — host/mmap authority; Q6/Q6/Q8 spans |
| E015 | Q2 one-layer compact graph substitution | N01,H10 | E014 | Replace only one layer's gate/up/down and IDs in `build_moe_ffn`; assert `[2048,512,10]`, `[512,2048,10]`, and top-8 ID view; any unrelated layer change fails. | **PASS** — layer 21, ten slots |
| E016 | Q2 route→map→stage callback | N01,H10,H13 | E015 | Read `ffn_moe_topk` after routing, map original IDs associatively, stage changed gate/up/down and eight IDs, and prove copy-before-MMID ordering; reject multi-position input. | **PASS** — route/ID/fence checks |
| E017 | Q2 one-token correctness certificate | N01,H02,H33 | E016 | One greedy token/one position; compare control output/token, source/slot hashes, mapped IDs, fence/epoch order, and all five required counters; any nonzero counter is FAIL. | **PASS** — exact; certificate written |
| E018 | Q2 eight-token upload-all control | N01,H38 | E017 | Deliberate fresh/upload-all policy; report changed slots, upload bytes/calls, staging peak, fence count, timing, VRAM/GTT; it is the baseline, not a speed claim. | **PASS transport** — timing/residency deferred |
| E019 | Q2 eight-token changed-slot persistence | N01,H10,H38 | E017,E018 | Compare associative changed-slot uploads against upload-all over the same greedy eight-token trace; require identical output and report hits/bytes/timing. | **PASS transport** — exact; timing deferred |
| E020 | Q2 stale/fence one-slot thrash oracle | N01,H10,H13,H33 | E017 | Force one replacement slot (`top_k=8`, slots=9), repeated rank permutations, and fence boundaries; require identical source/slot hashes and zero repurpose-before-fence/stale reads. | **PASS** — exact per-token bytes; all counters zero |
| E021 | Q2 full-core device budget allocation | N02,N01,H37 | E017,E006 | Keep routed authority CPU/mmap, allocate dense/shared core plus compact slots and reserve 700–1,000 MiB; fail if VRAM/GTT or graph placement violates budget. | **PASS placement/residency** — 40/41 core layers, 12.04-GiB decode-VRAM reduction, GTT flat, greedy tokens equal; reserve/speed deferred |
| E022 | Q2 ordinary versus exceptional layer/revert matrix | N01,N02,H02 | E017,E021 | Compare ordinary Q6/Q6/Q8 layer, layer 1 all-Q8, selected layer 21, and disabled/revert control; any DeepSeek stride assumption fails. | GATED |

# Group 2 — Qwen trace, fingerprints, and MARC layers (`E023`–`E034`)

These are prepared while Q2 runs, but the larger P1 collection and any learned/binder decision wait for the Qwen policy gate.

| ID | Experiment | Families | Dependency | Smallest protocol and falsifier | State |
|---|---|---|---|---|---|
| E023 | Qwen trace/no-trace parity pair | N03,N04,H33 | E002, qwen_trace build | Same prompt and configuration with callback absent/present; token IDs and route IDs must be identical. | READY-STATIC |
| E024 | Qwen signal completeness/overhead | N04,N05,H14 | E023 | Check top-8 IDs/probs, margins, hidden/output fields, entropy, RSS/timing, malformed records, and bytes/token; missing required signals or material overhead fails. | READY-STATIC |
| E025 | Qwen P1 phase-transition trace collection | N03,N04 | E023,E024 | Six phase classes, held-out prompts, explicit labels, no future leakage; this is a gated later collection, not Q2 data. | GATED |
| E026 | B0–B5 fingerprint baseline analysis | N04,N05 | E025 | Compare keyword/hash, final-hidden, routes/margins, hidden/entropy, structured semantic, and simulated hardware fingerprints with identical splits. | GATED |
| E027 | Phase-boundary and fingerprint lifetime gate | N04 | E026 | Measure AMI, boundary precision/recall, lifetime and leave-one-class-out stability; fail after two extraction redesigns if fixed gates miss. | GATED |
| E028 | Future hotset prediction holdout | N04,H06 | E025,E026 | Predict next-window route unions using only history/current signals; require lift over last-route baseline and report per-class/seed spread. | GATED |
| E029 | Structured versus opaque/fixed executor twin | N04,N09 | E026,E027 | CPU twin compares typed fields→fixed body, opaque latent→fixed body, and typed fields→modular body; fail if structure adds no held-out value. | GATED |
| E030 | Simulated hardware fingerprint binder | N05,N09,H10 | E019,E024 | Feed slot/cache/staging/queue snapshots to a CPU twin and compare decisions to actual Q2 trace; stale fields must be treated as unknown. | READY-STATIC |
| E031 | MARC-Symbiote P1 structured phase analysis | N09,N04,N05 | E027,E030 | Run only trace/twin logic; no graph/body; pass requires structured multi-signal clustering and hotset lift over both baselines. | GATED |
| E032 | MARC-X route/residency shadow | N06,H06,H10 | E019,E024 | Shadow hotset/prefetch/keep decisions against Q2 actual routes; apply nothing and score useful/wasted bytes and displacement. | READY-STATIC |
| E033 | MARC-OS budget-policy re-evaluation | N07,N04,N05,H15 | E026,E030 | Re-evaluate historical prompt-budget policy under hardware/quality-aware held-out Qwen traces; do not treat keyword savings as transport proof. | READY-STATIC |
| E034 | MARC-Synapse controlled toy replication | N08,N04,N25,H28 | historical V0 audit | Reproduce equal-active-FLOP modular versus monolithic baseline with non-keyword signal ablation; failure preserves the negative toy result. | DESIGN/GATED |

# Group 3 — HERMES representation, movement, and control mechanisms (`E035`–`E054`)

| ID | Experiment | Families | Dependency | Smallest protocol and falsifier | State |
|---|---|---|---|---|---|
| E035 | E1 compact-skeleton fidelity matrix | H03,H35,H02 | HERMES E1A closure/E007 | Bits 2/3/4 × shallow/deep layers; compare logits/routes/tokens/ability/bytes; no Qwen claim until Q2 control is clean. | GATED |
| E036 | Q8 residual-tile recovery curve | H05,H03,H28 | E035, canonical source | One layer/expert: output error versus ranked exact tile bytes; kill if near-complete expert is required. | GATED |
| E037 | P1/P2/P4/P6 incremental path test | H04,H07,H23 | E035,E036 | Compute nested paths with reuse, calibrate escalation, and require savings under fixed quality; fail if most positions immediately reach max or recompute. | GATED |
| E038 | Route Scout one-step shadow | H06,H10,H14 | E010, route traces | Compare transition/history predictors on useful bytes, arrival deadlines, waste, and stall avoided; F1 alone is not success. | READY-STATIC |
| E039 | Margin-aware routing calibration | H07,H02 | E024 or DeepSeek margin trace | Reliability-bucket margin versus disagreement/repair; fail if raw margin is not calibrated enough for a bounded threshold. | READY-STATIC |
| E040 | Persistent atlas policy replay | H10,H27 | E019, route history | Compare no map, LRU, LFU, Belady trace oracle, and salvage/value-aware per-layer maps; report hits, bytes, remaps, and displacement. | READY-STATIC |
| E041 | ExpertPack two-layer microprototype | H11,H10 | E005/E040, pack index | Pack exact gate/up/down spans, replay routes, verify reversible hashes and compare I/O/CPU/staging; no model quality change allowed. | READY-STATIC |
| E042 | RDNA4-native authority/compact baseline | H12,H37,N23 | E017 or DeepSeek exact kernel shape | Benchmark exact winning shapes, wave modes, fused/dequant variants with identical bytes and parity; isolated gain is insufficient. | GATED |
| E043 | Three-lane timestamp trace | H13,H14 | E017,E010 | Timestamp route, read, stage, upload, kernel, verify, commit and fence ownership; fail if apparent overlap is queue hiding or stale staging. | GATED |
| E044 | Full telemetry satellite attribution | H14,H13 | E010,E043 | Add per-token cache/I/O/upload/VK/sync/thermal/queue summaries and predict one tail event with negligible normal overhead. | READY-STATIC |
| E045 | Dynamic MoE GPS replay | H15,H14,H38 | E040,E044 | Compare static, greedy, receding-horizon, and oracle action policies; fail if planning overhead removes savings. | DESIGN/GATED |
| E046 | Safe/Balanced/Autobahn lane replay | H16,H15,H29 | E045 | Same trace/workload through fixed profiles; report accepted throughput, p95, waste, reserve, and recovery. | DESIGN/GATED |
| E047 | Motorway ramp metering | H17,H14,H16 | E043,E046 | Inject speculative load; compare unlimited/fixed/telemetry admission and recovery starvation. | DESIGN/GATED |
| E048 | Local multi-source fabric | H18,H10,H14 | E040,E044 | Select VRAM/RAM/NVMe by measured arrival cost; only after local win consider private LAN; no internet path. | DESIGN/GATED |
| E049 | Tailwind/headwind/sweet-spot sweep | H19,H14,H38 | E044,E046 | Thermal-stabilized concurrency/canvas sweep; identify repeatable regimes or reject universal sweet-spot assumptions. | DESIGN/GATED |
| E050 | Fatigue/recovery/RIR soak | H20,H29 | E044,E049 | Long generation with fatigue on/off under induced queue/cache/thermal stress; compare drift and recovery. | DESIGN/GATED |
| E051 | Compound versus isolation repair | H21,H09,H08 | E068/E069 or route replay | Compare shared block repair and local layer/position repair at equal work; fail if compound repair dominates or under-batching loses all reuse. | GATED |
| E052 | Macro C/P/F reserve sweep | H22,H17,H20 | E047,E050 | Sweep productive, verification, and reserve budgets; require better sustained accepted economics with nonzero reserve. | DESIGN/GATED |
| E053 | Water-purification cascade | H23,H04,H07 | E037,E039 | Calibrate uncertainty reduction and stopping at each stage; fail on unsafe early stops or no saved work. | DESIGN/GATED |
| E054 | Sunscreen protection budget | H24,H08,H07 | E053,E066 | Uniform versus risk-weighted verification with prefix-sensitive positions; fail if later protection hides early-prefix risk. | DESIGN/GATED |

# Group 4 — learning, arbitration, and slow adaptation (`E055`–`E065`)

| ID | Experiment | Families | Dependency | Smallest protocol and falsifier | State |
|---|---|---|---|---|---|
| E055 | Behaviorism shadow bandit | H25,H14,H28 | E044,E058 | Train/evaluate one bounded action on held-out traces; fallback on drift; fail if it cannot beat frozen rules without KPI reward hacking. | DESIGN/GATED |
| E056 | Id/Superego/Ego arbiter ablation | H26,H29 | E045,E055 | Log proposals, hard rejections, and selected actions; ablate proposer/filter and require safety/opportunity separation. | DESIGN/GATED |
| E057 | Capital/salvage-value policy | H27,H10,H21 | E040,E051,E066 | Compare latency/byte/hit-only against future-reuse/salvage value; score total accepted-token economics. | READY-STATIC |
| E058 | Inference-IQ/ability-per-joule battery | H28,H02 | fixed canonical/degraded paths | Establish battery sensitivity, then evaluate compact/pruned/skip paths; fail if it cannot distinguish deliberate degradation. | READY-STATIC |
| E059 | Maturana viability governor | H29,H14,H17 | E044,E047,E050 | Inject overload, stale state, thermal/queue pressure and require shrink/cancel/revert/fail-closed behavior. | DESIGN/GATED |
| E060 | Wolff/Inference Mechanostat replay | H30,H10,H27 | E040,E057 | Slow EMA placement/remodeling with hysteresis versus LRU/LFU on held-out traces; fail on prompt-local overfit or instability. | DESIGN/GATED |
| E061 | Neuro-inspired control-plane ablation | H31,H14,H26,H29 | E044,E056,E059 | Implement safety interrupt, rule selector, slow planner, and bus; each module must have measurable responsibility. | DESIGN/GATED |
| E062 | Startup autotuner | H32,H37,H38 | E003,E042,E044 | Bounded quick/full profile, cache by hardware/model/shader identity, restart/reproduce; fail on stale profile or unrepeatable gain. | READY-STATIC |
| E063 | Explorer-Verifier certificate replay | H33,H02 | E004,E017 | Produce L0–L6 package from a narrow run and independently replay it; fail if parser/environment/hash ambiguity remains. | READY-STATIC |
| E064 | Reasoning-distilled controller | H34,H15,H25,H29 | E045,E055,E059 | Distill one oracle decision; compare student/rule/oracle with bounded runtime cost and rollback. | DESIGN/GATED |
| E065 | Integrated HERMES invariant dry-run | H39,H02,H33 | E012,E063 | Static dependency/kill-switch/label audit for a two-component composition; fail if any hidden authority, byte, or correctness transition exists. | READY-STATIC |

# Group 5 — DSpark/MTP and cascade systems (`E066`–`E075`)

| ID | Experiment | Families | Dependency | Smallest protocol and falsifier | State |
|---|---|---|---|---|---|
| E066 | Trace-only future-token canvas economics | H08,H09,H38 | route traces, no runtime | K=1/2/4/8/12/16 union, reuse, exposed-byte and acceptance-weighted simulation; fail if union/byte cost erases all possible value. | READY-STATIC |
| E067 | DSpark/MTP tensor and converter audit | N18,H08 | model inventories | Enumerate official/current tensors, Markov/confidence heads, converter gaps, and storage; missing authority or incompatible topology blocks restore. | READY-STATIC |
| E068 | Teacher-forced expert-major batching | H09,H21 | E066, recorded rows | Two layers/K=2/4 grouped execution or replay; compare numeric results, physical loads, and scatter/causal state. | READY-STATIC |
| E069 | True multi-position graph prototype | H09,H13,N18 | E017,E068 | Minimal Qwen/DeepSeek two-position graph with explicit KV snapshots; fail on broadcast IDs, stale state, or no real load amortization. | GATED |
| E070 | Future route predictor comparison | H06,H08,N18 | E066,E067 | History, transition, token-conditioned, and optional hidden-head predictors under temporal holdout; score miss-weighted useful bytes, not F1 alone. | GATED |
| E071 | Adaptive/byte-budgeted horizon | H08,H15,H38,N18 | E070,E044 | Stop at marginal value/byte using confidence, cache, and queue state; fail if adaptive is no better than best fixed horizon. | GATED |
| E072 | Deadline prefetch and rejected-token salvage | H08,H13,H17,H27,N18 | E069,E070,E071 | EDF/slack scheduling, useful/wasted/late bytes, wrong-token/right-memory and cache pollution; fail if prefetch only saturates the link. | GATED |
| E073 | MoE-Skipper single-layer quality gate | N17,H02,H28,H33 | Qwen exact supervision | One layer direct logits, KL/PPL/top-token/task gate on held-out corpus; no speed test before pass. | EVIDENCE/REPRODUCE ONLY |
| E074 | MoE-Skipper cascade correction | N17,H02,H28 | E073 | Two/three-layer cascade-aware correction, independent corpus, exact-tail test; fail on the documented error wall or workload-dependent regression. | GATED |
| E075 | DSpark/MTP custom runtime end-to-end | N18,H08,H09,H13 | E067,E069,E071,E072, Q2/full-core | K=2 first, then bounded K; strict acceptance, draft cost, bytes, salvage, and exact output; no MTP enablement in Q2 baseline. | BLOCKED/GATED |

# Group 6 — pruning, AutoSurgeon, variants, and R4I8 (`E076`–`E084`)

| ID | Experiment | Families | Dependency | Smallest protocol and falsifier | State |
|---|---|---|---|---|---|
| E076 | Real REAP calibration | N10,H28 | model/checkpoint access | Real held-out calibration by layer, route counts/weights, seed stability; synthetic-only ranking cannot promote pruning. | READY-STATIC/GATED |
| E077 | Laguna exact inventory and streaming converter | N11,N12 | E076, model artifacts | Stream BF16 shards into a manifest-selected GGUF without full materialization; hash retained tensors and verify router rewrite. | READY-STATIC |
| E078 | Pruning ratio quality frontier | N12,N10,H02,H28 | E076 | REAP versus random at 10–60% per-layer pruning; logits, task ability, route validity, bytes, and speed only after quality. | GATED |
| E079 | AutoSurgeon dry-run manifest | N13,N10,N12,H33 | E076,E077 | Preserve exact name, emit reversible surgery/quantization manifest, source/variant hashes, rollback, and no writes in dry-run; missing semantics remain BLOCKED. | BLOCKED/READY-STATIC |
| E080 | Coverage-first variant | N14,N13 | E079,N19 or native low-bit format | All 256 routed experts near 2.36 bpw; equal-byte held-out layer/logit/task gate; fail if coverage does not beat alternatives. | MUTEX/GATED |
| E081 | Precision-first variant | N15,N13 | E079,E076 | Approximately 67 high-precision survivors, router rewrite, rare-expert stress prompts; equal-byte comparison and quality gate. | MUTEX/GATED |
| E082 | Heterogeneous variant | N16,N13,N19-N21 | E079,E076 | Approximately 40 Q6 + 80 R4I8 + 80 2.5-bit + 56 pruned; validate per-layer and end-to-end quality. | MUTEX/GATED |
| E083 | R4I8 CPU/Vulkan quality certificate | N19,N22,N23,H33 | R4I8 byte-order fix | CPU/Vulkan reconstruction, BOS metadata, varied output, held-out logits/task quality; fail if structural dispatch is mistaken for quality. | READY-STATIC/GATED |
| E084 | Equal-byte three-variant comparison | N14,N15,N16,N19 | E080,E081,E082,E083 | Compare coverage/precision/heterogeneous under identical routed bytes, prompts, and kernels; select or kill, never average the claims. | MUTEX/GATED |

# Group 7 — future formats, kernels, context/memory, and final gates (`E085`–`E096`)

| ID | Experiment | Families | Dependency | Smallest protocol and falsifier | State |
|---|---|---|---|---|---|
| E085 | R5I8 format prototype | N20,N19,N22 | E083 | Define one block layout, CPU reconstruction and bytes; stop if the format is not independently specified or R4I8 is being silently renamed. | DESIGN/BLOCKED |
| E086 | R6I8 format prototype | N21,N20,N22 | E085 | Define one higher-fidelity layout and compare reconstruction/overhead; no model conversion until the format gate passes. | DESIGN/BLOCKED |
| E087 | RDNA4 cooperative-matrix A/B | N22,H12,H37 | E083 or E085/E086, exact shape | Same bytes/weights with and without coopmat; CPU/Vulkan parity and end-to-end accepted-token time required. | GATED |
| E088 | RDNA4 GEMV/GEMM shape sweep | N23,H12,H09 | E017,E042,E087 | Qwen `[2048,512,10]` and `[512,2048,10]`, single-position and grouped cases; compare launch, VRAM bandwidth, and total decode. | GATED |
| E089 | Q2 selected-layer kernel/config A/B | N01,N23,H37 | E017,E088 | Benchmark layer 21-like ordinary Q6/Q6/Q8 path with control flags fixed; revert any isolated gain that does not survive end-to-end. | GATED |
| E090 | Context-only streaming twin | N24,H36,H02 | E008, model memory-state access | Exact recent context plus compressed older checkpoints/selective retrieval; fail on recall loss, high fallback, or no KV/latency savings. | DESIGN/GATED |
| E091 | Associative semantic/context memory | N25,N04,N09 | E027,E090 | CPU twin retrieves stable facts/segments with provenance and invalidation; fail on stale execution or retrieval overhead canceling savings. | DESIGN/GATED |
| E092 | Autonomous local self-experiment loop | N26,H33,H29 | E012,E063 | Run one CPU-only hypothesis→protocol→fault→verification cycle with finite budget and failure preservation; no autonomous graph mutation. | READY-STATIC |
| E093 | Qwen full-core long A/B | N02,N01,H38 | E021,E022,E089 | Compare NGL=20, Q2 compact, and Q2+full-core over bounded one-token/8-token/short generation; include VRAM/GTT, bytes, hits, timings, tokens. | GATED |
| E094 | Joint semantic×hardware binder twin | N09,N04,N05,N25 | E030,E031,E091 | Bind phase and live resource state before selecting explicit primitive; compare semantic-first and joint binders, fail on stale or unsafe actions. | GATED |
| E095 | Staged HERMES composite | H39,H02,H33,N01,N02 | E017,E040,E043,E058,E093 | Compose only certified Q2 transport + full-core + telemetry/one control; exact output and exposed-byte accounting decide whether integration continues. | GATED |
| E096 | Final portfolio/claim gate | H39,N26,H02,H33,H38 | E065,E075,E084,E093-E095 | Produce accepted/rejected/blocked table, ablations, hashes, correctness classes, roofline position, and next queue; any unsupported speed/quality claim is removed. | GATED |

---

## Active experiment

**N01 / Qwen Q2 compact expert transport** is complete for its ordinary-layer
line through E021. The certified scope is one routed layer and one position
per decode, with Qwen Q6_K_XL, NGL=20 controls, queue flags disabled,
CPU/mmap authority, persistent Vulkan arenas, associative ID remapping, native
`MUL_MAT_ID`, fence-before-epoch reset, and fail-closed errors. Eight-token
persistence, the slots=9 fence oracle, and full-core placement/residency A/B
all have reports; no speed claim is made. E022 and wider work remain gated.

Required zero-error counters:

```text
staging_failures=0
missing_expert_lookups=0
unmapped_route_ids=0
slot_repurpose_before_fence=0
stale_slot_reads=0
```

Expected diagnostic markers include `QWEN_Q2_ALLOC`, `QWEN_Q2_SLOT`, `QWEN_Q2_IDS`, `QWEN_Q2_VERIFY`, `QWEN_Q2_STATS`, and `QWEN_Q2_FAIL`.

## Next dependent experiments

The ordinary Q2 line is complete through E021. The next Qwen item is **E022**
(ordinary versus exceptional layer/revert matrix), which remains gated by the
no-wider-experiment policy. E089/E093 performance work and all fingerprint,
DeepSeek, and composite lines remain gated. The full-core report already
contains the bounded placement/VRAM/GTT A/B; it does not authorize a speed
claim.

## Ideas missing from the earlier 12-lane queue

The earlier queue is not authoritative and its durable copy was not found during this audit. The complete missing-as-independent-work-item list is therefore the restored set:

- **HERMES names:** H05–H07, H09–H34, H35–H39 (Q8 residuals, Route Scout, margin calibration, expert-major batching, stable atlas, ExpertPack, asynchronous lanes, telemetry, GPS, all named traffic/bodybuilding/purification/protection/learning/control analogies, startup autotuner, certificates, distilled controller, E1, E2, kernel/configuration lead, roofline, and integrated architecture).
- **New names:** N01–N26 (Qwen compact transport, Qwen full-core residency, Qwen-first policy, semantic/hardware fingerprints, MARC-X/OS/Synapse, MARC-Symbiote body, REAP/Laguna/pruning/AutoSurgeon, all three compressed variants, MoE-Skipper cascade/correction, DSpark/MTP restoration/custom runtime, R4I8/R5I8/R6I8, cooperative matrix, GEMV/GEMM, context-only streaming, associative memory, and autonomous local self-experimentation).

This is why the queue contains 96 experiments rather than 12 lanes. The active schedule still deliberately runs only Q2.

## Static preparation permitted while Q2 runs

1. Build/repair Qwen trace parser, topology-aware analysis, parity checker, and Q2 certificate parser.
2. Add CPU-only route-union/changed-slot simulations and test expected bytes/hits without making runtime decisions.
3. Prepare Qwen graph shape assertions and loader override tests.
4. Prepare ExpertPack offset/index schema and reversible hash tests; do not claim an I/O gain.
5. Prepare Qwen P1 gate code, anti-leakage tests, and prompt manifest; do not run a broad collection during Q2.
6. Prepare CPU format fixtures for R4I8/R5I8/R6I8 and native kernel build targets; preserve Q6 control.
7. Prepare MoE-Skipper quality parsers and exact-tail controls; do not expand layers.
8. Prepare context-memory and semantic-memory CPU twins.
9. Prepare bounded autonomous queue/certificate orchestration with finite budgets and no graph mutation.

## Contradictions/duplicates requiring interpretation

- **MARC acronym collision:** H07 Margin-Aware Routing Calibration versus N06–N09 Modular Architecture with Routing and Control.
- **Q2 exact compact transport versus H03 low-bit skeleton:** same word “compact,” different representation and correctness.
- **H04 progressive expert widening versus N17 learned MoE-Skipper:** runtime path width versus learned layer substitution.
- **N01 Q2 versus N02 full-core residency:** complementary goal, but competing placement for routed weights in a given A/B.
- **N18 DSpark versus MTP:** official DeepSeek missing tensors, Qwen local artifacts, and separate custom runtime semantics.
- **N19/N20/N21:** only R4I8 currently has an implementation record; exact desired R5I8/R6I8 layouts are unresolved.
- **N13 AutoSurgeon:** exact artifact/definition not located; do not infer it from REAP/Laguna scripts.
- **N14/N15/N16:** three named equal-byte variant hypotheses, not one averaged model.
- **N22 coopmat versus N23 GEMV/GEMM:** hardware instruction family and kernel-shape scheduling must be measured separately.
- **N24 context-only streaming versus expert-weight streaming:** separate memory ledgers and correctness gates.
- **N25 associative memory versus H10 physical expert atlas:** semantic/context retrieval versus ID-to-slot mapping.
- **H39 integration:** no mechanism may enter the composite merely because its local metric improved; its correctness class, exposed bytes, and ablation must survive.
