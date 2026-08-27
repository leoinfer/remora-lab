# HAR idea implementation matrix

This is a routing matrix, not an implementation claim. `PASSED` means the source evidence/status supports that scoped state; it does not authorize a global performance claim. Agent 1 alone may promote full-model performance claims.

| ID | Taxonomy | Idea | Owner | HAR subsystem | Status | Evidence | Blockers/tests |
|---|---|---|---|---|---|---|---|
| `C-01` | C | Causal-Closure Cache (CCC) | agent-3 | rust-core/ir/provenance | `NOT_STARTED` | CONJECTURED | `CCC-v1` records closure members, root, artifact hash, validation rule, and exact/approximate mode. The checker requires a miss on every changed causal leaf. |
| `C-02` | C | Acceptance–Residency Phase Transition | agent-6 | memory/residency/moe-streaming | `NOT_STARTED` | CONJECTURED` with `DERIVED UNDER ASSUMPTIONS` necessary condition | schema validation; CPU/static fixture |
| `C-03` | C | Certified Approximation Lattice (CAL) | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | CONJECTURED | schema validation; CPU/static fixture |
| `C-04` | C | Epoch-Namespace State Machine (ENSM) | agent-3 | rust-core/ir/provenance | `NOT_STARTED` | CONJECTURED`, motivated by `COUNTEREXAMPLE FOUND | schema validation; CPU/static fixture |
| `C-05` | C | Union-Aware Residual Cache (UARC) | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | CONJECTURED | schema validation; CPU/static fixture |
| `C-06` | C | Slack-Priced Elastic Horizon (SPEH) | agent-2 | language/decode/control | `NOT_STARTED` | CONJECTURED | schema validation; CPU/static fixture |
| `C-07` | C | Miss-Conditioned Prediction Sufficiency | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | CONJECTURED | schema validation; CPU/static fixture |
| `C-08` | C | Exactness Requires a State Boundary, Not Just an Authority Endpoint | agent-3 | rust-core/ir/provenance | `NOT_STARTED` | DERIVED UNDER ASSUMPTIONS | schema validation; CPU/static fixture |
| `C-09` | C | Phenotype Compiler Must Emit a Safe Region, Not One Optimum | agent-5 | compiler/quantization/sidecar | `NOT_STARTED` | CONJECTURED | schema validation; CPU/static fixture |
| `C-10` | C | Certificate-First Autonomous Experimentation | agent-8 | integration/provenance | `NOT_STARTED` | CONJECTURED | Give a queue item with a speed metric but no exactness/timing denominator. The system must return `BLOCKED`, not run and average.; CPU-only queue slice with one valid static checker, one missing artifact, and one injected failure. Require immutable evidence outputs and no promotion. |
| `CE-01` | CE | “Stop at the first non-positive horizon marginal.” | agent-2 | language/decode/control | `NOT_STARTED` | `COUNTEREXAMPLE FOUND` | schema validation; CPU/static fixture |
| `CE-02` | CE | “Same token means same continuation state.” | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | `COUNTEREXAMPLE FOUND` | schema validation; CPU/static fixture |
| `CE-03` | CE | “Prompt hash or model hash is an exact cognition cache key.” | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | `COUNTEREXAMPLE FOUND` | schema validation; CPU/static fixture |
| `CE-04` | CE | “Independent PHASE branch probabilities can be summed.” | agent-2 | language/decode/control | `NOT_STARTED` | `COUNTEREXAMPLE FOUND` | schema validation; CPU/static fixture |
| `CE-05` | CE | “Overall Route Scout F1 predicts useful prefetch.” | agent-6 | memory/residency/moe-streaming | `NOT_STARTED` | `MACHINE-CHECKED` within small trace | schema validation; CPU/static fixture |
| `CE-06` | CE | “Previous-token retention is useful prefetch.” | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | `MACHINE-CHECKED` within small trace | schema validation; CPU/static fixture |
| `CE-07` | CE | “Perfect prediction solves the 30-t/s storage problem.” | agent-6 | memory/residency/moe-streaming | `NOT_STARTED` | `MACHINE-CHECKED` model-dependent | schema validation; CPU/static fixture |
| `CE-08` | CE | “Reduced bytes imply speed or energy win.” | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | `MACHINE-CHECKED` scoped certificate | schema validation; CPU/static fixture |
| `CE-09` | CE | “Local exact certificates compose automatically.” | agent-8 | integration/provenance | `NOT_STARTED` | `MACHINE-CHECKED` | schema validation; CPU/static fixture |
| `CE-10` | CE | “Q4×43 numerical failure proves Q4 is bad.” | agent-3 | rust-core/ir/control-plane | `REJECTED` | `MACHINE-CHECKED` / claim falsified | schema validation; CPU/static fixture |
| `CE-11` | CE | “A fast predictor path is a speedup.” | agent-3 | rust-core/ir/control-plane | `REJECTED` | `FALSIFIED` for quality-valid speed claim | schema validation; CPU/static fixture |
| `CE-12` | CE | “Keyword/hash MARC is a semantic fingerprint.” | agent-3 | rust-core/ir/control-plane | `REJECTED` | `FALSIFIED` for semantic-success claim | schema validation; CPU/static fixture |
| `CE-13` | CE | “Asynchronous submission proves overlap.” | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | `COUNTEREXAMPLE FOUND` | schema validation; CPU/static fixture |
| `CE-14` | CE | “A high cache hit rate is enough.” | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | `COUNTEREXAMPLE FOUND` / static support | schema validation; CPU/static fixture |
| `CE-15` | CE | “Layer-stationary wavefront is exact when the LayerPack is resident.” | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | `COUNTEREXAMPLE FOUND` at dependency level | schema validation; CPU/static fixture |
| `CE-16` | CE | “Argmax margin/calibration is an exactness certificate.” | agent-5 | compiler/quantization/sidecar | `NOT_STARTED` | `PROVED` failure condition; `CONJECTURED` runtime frequency | schema validation; CPU/static fixture |
| `CE-17` | CE | “A larger staging/cache/queue is always safer/faster.” | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | `COUNTEREXAMPLE FOUND` | schema validation; CPU/static fixture |
| `CE-18` | CE | “MTP primary parity establishes a repeatable baseline.” | agent-2 | language/decode/control | `REJECTED` | `FALSIFIED` for repeated-baseline claim | schema validation; CPU/static fixture |
| `CE-19` | CE | “Qwen Q2 exactness transfers to DeepSeek.” | agent-6 | memory/residency/moe-streaming | `REJECTED` | `FALSIFIED` as a transfer claim | schema validation; CPU/static fixture |
| `CE-20` | CE | “Model hash alone fixes hardware phenotype.” | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | `COUNTEREXAMPLE FOUND` | schema validation; CPU/static fixture |
| `CE-21` | CE | “A valid current route prediction is valid after semantic/state expiry.” | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | `CONJECTURED` counterexample family | schema validation; CPU/static fixture |
| `CE-22` | CE | “One metric can be the global objective.” | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | `COUNTEREXAMPLE FOUND` by ledger construction and preserved failures | schema validation; CPU/static fixture |
| `CE-23` | CE | “A fixed hardware configuration is universally optimal.” | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | `COUNTEREXAMPLE FOUND` by model facts | schema validation; CPU/static fixture |
| `CE-24` | CE | “Tail probability coverage alone proves positive horizon economics.” | agent-2 | language/decode/control | `NOT_STARTED` | `COUNTEREXAMPLE FOUND` | schema validation; CPU/static fixture |
| `CE-25` | CE | “Exact route/slot bytes imply exact graph output.” | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | `COUNTEREXAMPLE FOUND` | schema validation; CPU/static fixture |
| `CE-26` | CE | “All expert size records can use one geometry.” | agent-6 | memory/residency/moe-streaming | `NOT_STARTED` | `MACHINE-CHECKED` | schema validation; CPU/static fixture |
| `CE-27` | CE | “A route-only trace is enough to prove block economics.” | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | `BLOCKED` claim | upstream dependency or authority gate remains open; see source status |
| `CE-28` | CE | “A calibrated confidence score is a certificate.” | agent-8 | integration/provenance | `REJECTED` | `FALSIFIED` as exactness wording | schema validation; CPU/static fixture |
| `E001` | E | Source-of-truth mode audit | agent-8 | integration/provenance | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E002` | E | Frozen Qwen baseline and environment freeze | agent-1 | external-oracle/evidence | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E003` | E | Hardware/traffic control ledger | agent-8 | integration/provenance | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E004` | E | Certificate fault injection | agent-8 | integration/provenance | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E005` | E | Qwen tensor geometry and authority index | agent-3 | rust-core/ir/control-plane | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E006` | E | Qwen NGL=20 VRAM/GTT placement control | agent-6 | memory/residency/moe-streaming | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E007` | E | DeepSeek E1A corrected closure | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | GATED/SEPARATE GPU LINE | upstream dependency or authority gate remains open; see source status |
| `E008` | E | Clean DeepSeek E2 non-MoE floor | agent-6 | memory/residency/moe-streaming | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E009` | E | Accepted-token roofline refresh | agent-1 | external-oracle/evidence | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E010` | E | Telemetry schema and overhead smoke | agent-8 | integration/provenance | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E011` | E | Applicability/transfer matrix | agent-6 | memory/residency/moe-streaming | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E012` | E | Queue dependency and omission lint | agent-3 | rust-core/ir/control-plane | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E013` | E | Q2 manager compile/API smoke | agent-3 | rust-core/ir/control-plane | `PASSED` | **PASS** — `29e0b3904` | schema validation; CPU/static fixture |
| `E014` | E | Q2 selected-layer CPU/mmap authority | agent-3 | rust-core/ir/control-plane | `PASSED` | **PASS** — host/mmap authority; Q6/Q6/Q8 spans | schema validation; CPU/static fixture |
| `E015` | E | Q2 one-layer compact graph substitution | agent-3 | rust-core/ir/control-plane | `PASSED` | **PASS** — layer 21, ten slots | schema validation; CPU/static fixture |
| `E016` | E | Q2 route→map→stage callback | agent-3 | rust-core/ir/control-plane | `PASSED` | **PASS** — route/ID/fence checks | schema validation; CPU/static fixture |
| `E017` | E | Q2 one-token correctness certificate | agent-1 | external-oracle/evidence | `PASSED` | **PASS** — exact; certificate written | schema validation; CPU/static fixture |
| `E018` | E | Q2 eight-token upload-all control | agent-3 | rust-core/ir/control-plane | `PASSED` | **PASS transport** — timing/residency deferred | schema validation; CPU/static fixture |
| `E019` | E | Q2 eight-token changed-slot persistence | agent-3 | rust-core/ir/control-plane | `PASSED` | **PASS transport** — exact; timing deferred | schema validation; CPU/static fixture |
| `E020` | E | Q2 stale/fence one-slot thrash oracle | agent-1 | external-oracle/evidence | `PASSED` | **PASS** — exact per-token bytes; all counters zero | schema validation; CPU/static fixture |
| `E021` | E | Q2 full-core device budget allocation | agent-3 | rust-core/ir/control-plane | `PASSED` | **PASS placement/residency** — 40/41 core layers, 12.04-GiB decode-VRAM reduction, GTT flat, greedy tokens equal; reserve/speed deferred | schema validation; CPU/static fixture |
| `E022` | E | Q2 ordinary versus exceptional layer/revert matrix | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E023` | E | Qwen trace/no-trace parity pair | agent-3 | rust-core/ir/control-plane | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E024` | E | Qwen signal completeness/overhead | agent-3 | rust-core/ir/control-plane | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E025` | E | Qwen P1 phase-transition trace collection | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E026` | E | B0–B5 fingerprint baseline analysis | agent-1 | external-oracle/evidence | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E027` | E | Phase-boundary and fingerprint lifetime gate | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E028` | E | Future hotset prediction holdout | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E029` | E | Structured versus opaque/fixed executor twin | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E030` | E | Simulated hardware fingerprint binder | agent-3 | rust-core/ir/control-plane | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E031` | E | MARC-Symbiote P1 structured phase analysis | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E032` | E | MARC-X route/residency shadow | agent-6 | memory/residency/moe-streaming | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E033` | E | MARC-OS budget-policy re-evaluation | agent-3 | rust-core/ir/control-plane | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E034` | E | MARC-Synapse controlled toy replication | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `E035` | E | E1 compact-skeleton fidelity matrix | agent-5 | compiler/quantization/sidecar | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E036` | E | Q8 residual-tile recovery curve | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E037` | E | P1/P2/P4/P6 incremental path test | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E038` | E | Predictive Expert Route Selection Shadow Step | agent-6 | memory/residency/moe-streaming | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E039` | E | Margin-aware routing calibration | agent-5 | compiler/quantization/sidecar | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E040` | E | Persistent atlas policy replay | agent-6 | memory/residency/moe-streaming | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E041` | E | ExpertPack two-layer microprototype | agent-6 | memory/residency/moe-streaming | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E042` | E | RDNA4-native authority/compact baseline | agent-4 | backend/vulkan/kernels | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E043` | E | Three-lane timestamp trace | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E044` | E | Global Runtime Telemetry Attribution | agent-8 | integration/provenance | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E045` | E | Receding-Horizon MoE Route Control Replay | agent-6 | memory/residency/moe-streaming | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `E046` | E | Risk-Aware Operating Profile Replay | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `E047` | E | Admission Control and Queue Metering | agent-6 | memory/residency/moe-streaming | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `E048` | E | Local multi-source fabric | agent-8 | integration/provenance | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `E049` | E | Sustainable Operating-Point Sweep | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `E050` | E | Resource Fatigue and Recovery Control Soak | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `E051` | E | Batched Work and Scoped Repair | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E052` | E | Adaptive Resource Budget Reserve Sweep | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `E053` | E | Staged Uncertainty Filtering | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `E054` | E | Risk-Weighted Verification Budget | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `E055` | E | Outcome-Driven Policy Shadow Bandit | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `E056` | E | Constraint-Gated Action Selection Ablation | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `E057` | E | Salvage-Aware Work Valuation Policy | agent-3 | rust-core/ir/control-plane | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E058` | E | Energy-Normalized Capability Battery | agent-1 | external-oracle/evidence | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E059` | E | Viability-Region Runtime Control | agent-1 | external-oracle/evidence | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `E060` | E | Long-Horizon Structural Adaptation Replay | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `E061` | E | Multi-Timescale Control-Plane Ablation | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `E062` | E | Startup autotuner | agent-3 | rust-core/ir/control-plane | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E063` | E | Evidence-Certified Exploration and Verification Replay | agent-8 | integration/provenance | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E064` | E | Reasoning-distilled controller | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `E065` | E | Integrated HERMES invariant dry-run | agent-8 | integration/provenance | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E066` | E | Trace-Only Speculative Future-State Economics | agent-3 | rust-core/ir/control-plane | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E067` | E | DSpark/MTP tensor and converter audit | agent-2 | language/decode/control | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E068` | E | Teacher-forced expert-major batching | agent-6 | memory/residency/moe-streaming | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E069` | E | True multi-position graph prototype | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E070` | E | Future route predictor comparison | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E071` | E | Adaptive/byte-budgeted horizon | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E072` | E | Deadline prefetch and rejected-token salvage | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E073` | E | MoE-Skipper single-layer quality gate | agent-6 | memory/residency/moe-streaming | `NOT_STARTED` | EVIDENCE/REPRODUCE ONLY | schema validation; CPU/static fixture |
| `E074` | E | MoE-Skipper cascade correction | agent-6 | memory/residency/moe-streaming | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E075` | E | DSpark/MTP custom runtime end-to-end | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | BLOCKED/GATED | upstream dependency or authority gate remains open; see source status |
| `E076` | E | Real REAP calibration | agent-5 | compiler/quantization/sidecar | `READY` | READY-STATIC/GATED | schema validation; CPU/static fixture |
| `E077` | E | Laguna exact inventory and streaming converter | agent-6 | memory/residency/moe-streaming | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E078` | E | Pruning ratio quality frontier | agent-5 | compiler/quantization/sidecar | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E079` | E | AutoSurgeon dry-run manifest | agent-3 | rust-core/ir/control-plane | `READY` | BLOCKED/READY-STATIC | schema validation; CPU/static fixture |
| `E080` | E | Coverage-first variant | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | MUTEX/GATED | upstream dependency or authority gate remains open; see source status |
| `E081` | E | Precision-first variant | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | MUTEX/GATED | upstream dependency or authority gate remains open; see source status |
| `E082` | E | Heterogeneous variant | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | MUTEX/GATED | upstream dependency or authority gate remains open; see source status |
| `E083` | E | R4I8 CPU/Vulkan quality certificate | agent-4 | backend/vulkan/kernels | `READY` | READY-STATIC/GATED | schema validation; CPU/static fixture |
| `E084` | E | Equal-byte three-variant comparison | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | MUTEX/GATED | upstream dependency or authority gate remains open; see source status |
| `E085` | E | R5I8 format prototype | agent-5 | compiler/quantization/sidecar | `READY_BUT_BLOCKED` | DESIGN/BLOCKED | upstream dependency or authority gate remains open; see source status |
| `E086` | E | R6I8 format prototype | agent-5 | compiler/quantization/sidecar | `READY_BUT_BLOCKED` | DESIGN/BLOCKED | upstream dependency or authority gate remains open; see source status |
| `E087` | E | RDNA4 cooperative-matrix A/B | agent-4 | backend/vulkan/kernels | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E088` | E | RDNA4 GEMV/GEMM shape sweep | agent-4 | backend/vulkan/kernels | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E089` | E | Q2 selected-layer kernel/config A/B | agent-4 | backend/vulkan/kernels | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E090` | E | Context-only streaming twin | agent-7 | kv/contextfold/context-memory | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `E091` | E | Associative semantic/context memory | agent-7 | kv/contextfold/context-memory | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `E092` | E | Autonomous local self-experiment loop | agent-8 | integration/provenance | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `E093` | E | Qwen full-core long A/B | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E094` | E | Joint semantic×hardware binder twin | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E095` | E | Staged HERMES composite | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `E096` | E | Final portfolio/claim gate | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | GATED | upstream dependency or authority gate remains open; see source status |
| `F0` | F | Source and identifier audit | agent-8 | integration/provenance | `READY_BUT_BLOCKED` | MACHINE-CHECKED` / `BLOCKED` for missing ranking | Status: `MACHINE-CHECKED` / `BLOCKED` for missing ranking; Check:** required names `H01–H39`, `N01–N26`, manifest ideas `1–30`, `TBEH` appear exactly once in the crosswalk; required source ranking path exists or emits `BLOCKED`. |
| `F1` | F | Accepted-token roofline checker | agent-1 | external-oracle/evidence | `NOT_STARTED` | PROVED` accounting inequality; `MACHINE-CHECKED` only for static inputs | schema validation; CPU/static fixture |
| `F10` | F | Value-of-computation event ledger | agent-8 | integration/provenance | `NOT_STARTED` | PROVED` bookkeeping identity; `EXPERIMENTALLY TESTABLE | schema validation; CPU/static fixture |
| `F11` | F | Hardware phenotype compiler/checker | agent-5 | compiler/quantization/sidecar | `NOT_STARTED` | DERIVED UNDER ASSUMPTIONS`; `EXPERIMENTALLY TESTABLE` CPU-only first | schema validation; CPU/static fixture |
| `F12` | F | Certificate composition linter | agent-8 | integration/provenance | `NOT_STARTED` | CONJECTURED`; `EXPERIMENTALLY TESTABLE | Output:** composed/not-composed verdict with missing interface fields. |
| `F13` | F | Shadow-price policy checker | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | CONJECTURED`; `EXPERIMENTALLY TESTABLE | schema validation; CPU/static fixture |
| `F14` | F | Source-level invariant checker | agent-8 | integration/provenance | `NOT_STARTED` | EXPERIMENTALLY TESTABLE`; no production edits required | schema validation; CPU/static fixture |
| `F15` | F | Trace schema completeness checker | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | MACHINE-CHECKED` for current missing fields; `BLOCKED` for live replay | Status: `MACHINE-CHECKED` for current missing fields; `BLOCKED` for live replay; / 0 / F0/F15 / Resolves missing ranking/status and prevents source/trace overclaim / |
| `F2` | F | Finite optimal-stopping checker | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | PROVED` for finite abstract DP; `EXPERIMENTALLY TESTABLE | schema validation; CPU/static fixture |
| `F3` | F | TBEH trace replay verifier | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | BLOCKED` pending valid MTP traces | Status: `BLOCKED` pending valid MTP traces |
| `F4` | F | RSSO finite-state exactness checker | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | EXPERIMENTALLY TESTABLE`; live target data `BLOCKED | Status: `EXPERIMENTALLY TESTABLE`; live target data `BLOCKED` |
| `F5` | F | Delta bound and adversarial margin checker | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | PROVED` local argmax lemma; `EXPERIMENTALLY TESTABLE` bound checker | schema validation; CPU/static fixture |
| `F6` | F | Dependency Merkle/causal-closure checker | agent-7 | kv/contextfold/context-memory | `NOT_STARTED` | DERIVED UNDER ASSUMPTIONS`; `EXPERIMENTALLY TESTABLE` CPU-only | Output:** hit/miss decision and missing dependency list. |
| `F7` | F | PHASE branch-DAG enumerator | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | EXPERIMENTALLY TESTABLE`; real accepted outcomes `BLOCKED | Status: `EXPERIMENTALLY TESTABLE`; real accepted outcomes `BLOCKED` |
| `F8` | F | Resource-constrained critical-path checker | agent-8 | integration/provenance | `NOT_STARTED` | DERIVED UNDER ASSUMPTIONS`; `EXPERIMENTALLY TESTABLE | schema validation; CPU/static fixture |
| `F9` | F | Predictive residency/Belady replay | agent-6 | memory/residency/moe-streaming | `NOT_STARTED` | MACHINE-CHECKED` for equal-page static traces; weighted/live `EXPERIMENTALLY TESTABLE | schema validation; CPU/static fixture |
| `H-00` | HANDOFF | Source/trace authority closure | agent-8 | integration/provenance | `READY_BUT_BLOCKED` | BLOCKED | Status: `BLOCKED`; 2. record an explicit missing-source certificate and use the manifest/atlas without inventing a ranking. |
| `H-01` | HANDOFF | Accepted-token roofline and value-ledger replay | agent-1 | external-oracle/evidence | `READY` | READY-STATIC | `PASS`, `FAIL`, or `BLOCKED` per trace. |
| `H-02` | HANDOFF | RSSO exactness toy and break-even checker | agent-3 | rust-core/ir/control-plane | `READY` | BLOCKED` by B0/RSSO gate for live target; `READY-STATIC` for finite-state toy | Status: `BLOCKED` by B0/RSSO gate for live target; `READY-STATIC` for finite-state toy |
| `H-03` | HANDOFF | Delta-certified local bound checker | agent-3 | rust-core/ir/control-plane | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `H-04` | HANDOFF | Dependency-versioned exact artifact checker | agent-3 | rust-core/ir/control-plane | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `H-05` | HANDOFF | PHASE branch-DAG economics | agent-2 | language/decode/control | `READY` | READY-STATIC`; live accepted-outcome data `BLOCKED | Status: `READY-STATIC`; live accepted-outcome data `BLOCKED`; Only offline branch replay is justified. Live PHASE requires B0-valid target traces and a later gate review. |
| `H-06` | HANDOFF | Value-weighted residency replay | agent-6 | memory/residency/moe-streaming | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `H-07` | HANDOFF | Resource-complementarity and hardware phenotype compiler | agent-5 | compiler/quantization/sidecar | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `H-08` | HANDOFF | Proof-carrying certificate composition | agent-8 | integration/provenance | `READY` | READY-STATIC | schema validation; CPU/static fixture |
| `H-09` | HANDOFF | TBEH held-out replay (future only) | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | BLOCKED | Status: `BLOCKED`; A bounded shadow controller may be designed; live authority still requires a new gate. |
| `H-10` | HANDOFF | RSSO target K=2 gate (future only) | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | BLOCKED | Status: `BLOCKED`; 2. RSSO gate changes from `DEFER` to explicit proceed. |
| `H-11` | HANDOFF | Hardware phenotype live microbenchmark (future only) | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | BLOCKED` by current hardware ownership | Status: `BLOCKED` by current hardware ownership; 5. **Every blank metric stays `NOT_RUN`/`BLOCKED`, never zero. |
| `H01` | HERMES | consumer-hardware frontier-model challenge | agent-3 | rust-core/ir/control-plane | `IN_PROGRESS` | ACTIVE THESIS | schema validation; CPU/static fixture |
| `H02` | HERMES | source-of-truth correctness modes | agent-3 | rust-core/ir/reference-boundary | `READY` | POLICY/DONE | schema validation; CPU/static fixture |
| `H03` | HERMES | compact expert skeleton | agent-5 | compiler/quantization/sidecar | `READY` | E1 EVIDENCE | schema validation; CPU/static fixture |
| `H04` | HERMES | P1/P2/P4/P6 progressive MoE | agent-6 | memory/residency/moe-streaming | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `H05` | HERMES | Q8 residual tiles | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `H06` | HERMES | Predictive Expert Route Selection | agent-6 | memory/residency/moe-streaming | `READY` | SHADOW READY | schema validation; CPU/static fixture |
| `H07` | HERMES | margin-aware routing calibration | agent-5 | compiler/quantization/sidecar | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `H08` | HERMES | Speculative Future-State Scheduling | agent-2 | language/decode/control | `READY` | TRACE/SIM READY | schema validation; CPU/static fixture |
| `H09` | HERMES | expert-major multi-position batching | agent-6 | memory/residency/moe-streaming | `READY` | TEACHER-FORCED READY | schema validation; CPU/static fixture |
| `H10` | HERMES | persistent expert atlas and stable slots | agent-6 | memory/residency/moe-streaming | `READY_BUT_BLOCKED` | DEEPSEEK EVIDENCE; QWEN GATED | upstream dependency or authority gate remains open; see source status |
| `H11` | HERMES | ExpertPack | agent-6 | memory/residency/moe-streaming | `NOT_STARTED` | WIP/DESIGN | schema validation; CPU/static fixture |
| `H12` | HERMES | RDNA4-native execution | agent-4 | backend/vulkan/kernels | `READY` | CAPABILITY EVIDENCE | schema validation; CPU/static fixture |
| `H13` | HERMES | asynchronous three-lane pipeline | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | PARTIAL/GATED | upstream dependency or authority gate remains open; see source status |
| `H14` | HERMES | Global Runtime Telemetry | agent-8 | integration/provenance | `READY` | PARTIAL/READY STATIC | schema validation; CPU/static fixture |
| `H15` | HERMES | Receding-Horizon MoE Route Control | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `H16` | HERMES | Risk-Aware Operating Profiles | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `H17` | HERMES | Admission Control and Queue Metering | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `H18` | HERMES | Multi-Source Expert Placement | agent-6 | memory/residency/moe-streaming | `READY` | LOCAL TIERS READY | schema validation; CPU/static fixture |
| `H19` | HERMES | Sustainable Operating-Point Selection | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `H20` | HERMES | Resource Fatigue and Recovery Control | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `H21` | HERMES | Batched Work and Scoped Repair | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | TRACE/DESIGN | schema validation; CPU/static fixture |
| `H22` | HERMES | Adaptive Resource Budgeting | agent-8 | integration/provenance | `NOT_STARTED` | DESIGN | schema validation; CPU/static fixture |
| `H23` | HERMES | Staged Uncertainty Filtering | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `H24` | HERMES | Risk-Weighted Verification Budget | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `H25` | HERMES | Outcome-Driven Policy Learning | agent-2 | language/decode/control | `NOT_STARTED` | SHADOW ONLY | schema validation; CPU/static fixture |
| `H26` | HERMES | Constraint-Gated Action Selection | agent-2 | language/decode/control | `NOT_STARTED` | DESIGN | schema validation; CPU/static fixture |
| `H27` | HERMES | Salvage-Aware Work Valuation | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | TRACE/DESIGN | schema validation; CPU/static fixture |
| `H28` | HERMES | Energy-Normalized Capability Measurement | agent-1 | external-oracle/evidence | `READY` | BATTERY READY | schema validation; CPU/static fixture |
| `H29` | HERMES | Viability-Region Runtime Control | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | DESIGN/GATED | upstream dependency or authority gate remains open; see source status |
| `H30` | HERMES | Long-Horizon Structural Adaptation | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | SIMULATOR GATED | upstream dependency or authority gate remains open; see source status |
| `H31` | HERMES | Multi-Timescale Modular Control Plane | agent-3 | rust-core/ir/control-plane | `READY` | STATIC DESIGN | schema validation; CPU/static fixture |
| `H32` | HERMES | startup autotuner | agent-3 | rust-core/ir/control-plane | `READY` | STATIC READY | schema validation; CPU/static fixture |
| `H33` | HERMES | Evidence-Certified Exploration and Verification | agent-3 | rust-core/ir/reference-boundary | `PASSED` | DEEPSEEK DONE; QWEN Q2 EXACT PASS / RESIDENCY GATED | schema validation; CPU/static fixture |
| `H34` | HERMES | reasoning-distilled inference controller | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | FUTURE/GATED | upstream dependency or authority gate remains open; see source status |
| `H35` | HERMES | E1 compact-skeleton program | agent-5 | compiler/quantization/sidecar | `NOT_STARTED` | DEEPSEEK GO; QWEN PENDING | schema validation; CPU/static fixture |
| `H36` | HERMES | E2 non-MoE floor program | agent-6 | memory/residency/moe-streaming | `NOT_STARTED` | DEEPSEEK LOW-CONFIDENCE | schema validation; CPU/static fixture |
| `H37` | HERMES | kernel/configuration lead program | agent-4 | backend/vulkan/kernels | `IN_PROGRESS` | ACTIVE METHOD | schema validation; CPU/static fixture |
| `H38` | HERMES | accepted-token roofline/exposed-byte budget | agent-1 | external-oracle/evidence | `READY` | ANALYTICAL GATE | schema validation; CPU/static fixture |
| `H39` | HERMES | integrated HERMES-V4 architecture | agent-3 | rust-core/ir/reference-boundary | `NOT_STARTED` | DESIGN/INTEGRATION GATE | schema validation; CPU/static fixture |
| `N01` | N | Qwen compact expert transport | agent-6 | memory/residency/moe-streaming | `PASSED` | Q2 ORDINARY-LAYER PASS / WIDER WORK GATED | schema validation; CPU/static fixture |
| `N02` | N | Qwen full-core GPU residency | agent-6 | memory/residency/moe-streaming | `PASSED` | PLACEMENT PASS / SPEED GATED | schema validation; CPU/static fixture |
| `N03` | N | Qwen-first experimentation | agent-8 | integration/provenance | `IN_PROGRESS` | ACTIVE POLICY | schema validation; CPU/static fixture |
| `N04` | N | semantic fingerprints | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | P0 DONE/P1 GATED | upstream dependency or authority gate remains open; see source status |
| `N05` | N | hardware fingerprints | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | P0 DONE/Q2 GATED | upstream dependency or authority gate remains open; see source status |
| `N06` | N | MARC-X | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | HISTORICAL/PRACTICAL | schema validation; CPU/static fixture |
| `N07` | N | MARC-OS | agent-3 | rust-core/ir/control-plane | `READY` | HISTORICAL/PROXY EVIDENCE | schema validation; CPU/static fixture |
| `N08` | N | MARC-Synapse | agent-3 | rust-core/ir/control-plane | `REJECTED` | NEGATIVE TOY EVIDENCE | schema validation; CPU/static fixture |
| `N09` | N | MARC-Symbiote temporary execution body | agent-3 | rust-core/ir/control-plane | `READY` | STATIC ONLY | schema validation; CPU/static fixture |
| `N10` | N | REAP | agent-3 | rust-core/ir/control-plane | `READY` | PILOT EVIDENCE | schema validation; CPU/static fixture |
| `N11` | N | Laguna | agent-3 | rust-core/ir/control-plane | `READY` | ARCHIVED EVIDENCE | schema validation; CPU/static fixture |
| `N12` | N | model pruning | agent-5 | compiler/quantization/sidecar | `READY` | VIRTUAL READY | schema validation; CPU/static fixture |
| `N13` | N | AutoSurgeon | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | UNLOCATED/BLOCKED | upstream dependency or authority gate remains open; see source status |
| `N14` | N | three practical compressed model variants: coverage-first | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DESIGN/MUTEX | upstream dependency or authority gate remains open; see source status |
| `N15` | N | three practical compressed model variants: precision-first | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DESIGN/MUTEX | upstream dependency or authority gate remains open; see source status |
| `N16` | N | three practical compressed model variants: heterogeneous | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DESIGN/MUTEX | upstream dependency or authority gate remains open; see source status |
| `N17` | N | MoE-Skipper cascade and correction systems | agent-6 | memory/residency/moe-streaming | `READY` | QWEN APPROX EVIDENCE | schema validation; CPU/static fixture |
| `N18` | N | DSpark/MTP restoration and custom runtime | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | BLOCKED/GATED | upstream dependency or authority gate remains open; see source status |
| `N19` | N | R4I8 | agent-3 | rust-core/ir/control-plane | `PASSED` | STRUCTURAL PASS/QUALITY FAIL | schema validation; CPU/static fixture |
| `N20` | N | R5I8 | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | FUTURE PLACEHOLDER | schema validation; CPU/static fixture |
| `N21` | N | R6I8 | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | FUTURE PLACEHOLDER | schema validation; CPU/static fixture |
| `N22` | N | RDNA4 cooperative-matrix kernels | agent-4 | backend/vulkan/kernels | `READY` | CAPABILITY/STRUCTURAL EVIDENCE | schema validation; CPU/static fixture |
| `N23` | N | RDNA4 GEMV/GEMM kernels | agent-4 | backend/vulkan/kernels | `READY` | GENERIC PATH READY | schema validation; CPU/static fixture |
| `N24` | N | context-only streaming | agent-7 | kv/contextfold/context-memory | `NOT_STARTED` | RESEARCH ONLY | schema validation; CPU/static fixture |
| `N25` | N | associative memory | agent-7 | kv/contextfold/context-memory | `NOT_STARTED` | DESIGN ONLY | schema validation; CPU/static fixture |
| `N26` | N | autonomous local self-experimentation | agent-8 | integration/provenance | `READY` | METHOD/STATIC READY | schema validation; CPU/static fixture |
| `OP-01` | OP | Elastic-horizon optimality | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | MACHINE-CHECKED` for the `NOT_RUN` gate; `BLOCKED` for target replay. | Evidence status: `MACHINE-CHECKED` for the `NOT_RUN` gate; `BLOCKED` for target replay.; Status: **`EXPERIMENTALLY TESTABLE`**, currently **`BLOCKED`** by B0. |
| `OP-02` | OP | Accepted-token roofline | agent-1 | external-oracle/evidence | `NOT_STARTED` | MACHINE-CHECKED` for the preserved arithmetic inputs; no runtime speed claim. | schema validation; CPU/static fixture |
| `OP-03` | OP | RSSO wavefront exactness and break-even | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | MACHINE-CHECKED` static dependency facts; live exactness `BLOCKED`. | Status: `MACHINE-CHECKED` static dependency facts; live exactness `BLOCKED`.; No such live proof is present. Status: **`BLOCKED`**. |
| `OP-04` | OP | Delta-Certified skipping | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DERIVED UNDER ASSUMPTIONS`; end-to-end sequence skip `BLOCKED`. | Status: `DERIVED UNDER ASSUMPTIONS`; end-to-end sequence skip `BLOCKED`. |
| `OP-05` | OP | Dependency-versioned exact cognition reuse | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | MACHINE-CHECKED` for preserved failure examples; exact reuse theorem `DERIVED UNDER ASSUMPTIONS`. | REMORA requires provenance fields, expiry, and validation rules.; MARC-Symbiote explicitly requires semantic and hardware fingerprints plus refresh triggers; a final hidden vector alone is insufficient. |
| `OP-06` | OP | PHASE branch economics | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | BLOCKED` for live evidence; model is `DERIVED UNDER ASSUMPTIONS`. | PHASE requires B=1/2/4/8 coverage replay and actual-outcome coverage.; Status: `BLOCKED` for live evidence; model is `DERIVED UNDER ASSUMPTIONS`. |
| `OP-07` | OP | Resource-complementarity scheduling | agent-8 | integration/provenance | `NOT_STARTED` | MACHINE-CHECKED` failure boundaries; full scheduler `CONJECTURED`/`EXPERIMENTALLY TESTABLE`. | HERMES deferred staging and Q3→Q8 ReBAR repair show transport path selection matters. |
| `OP-08` | OP | Predictive MoE residency bounds | agent-6 | memory/residency/moe-streaming | `READY_BUT_BLOCKED` | MACHINE-CHECKED` for the preserved static replay; generalization `BLOCKED`. | 30 tok/s requires roughly `94–95%` byte hit rate in that model;; Status: `MACHINE-CHECKED` for the preserved static replay; generalization `BLOCKED`. |
| `OP-09` | OP | Value-of-computation conservation | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | MACHINE-CHECKED` at the evidence-policy level; conservation theorem is an accounting derivation. | schema validation; CPU/static fixture |
| `OP-10` | OP | Hardware phenotype compilation | agent-3 | rust-core/ir/control-plane | `NOT_STARTED` | DERIVED UNDER ASSUMPTIONS`; profile safety is `EXPERIMENTALLY TESTABLE`. | Same hardware, different workload: route-stable versus first-appearance churn requires different cache decisions. |
| `OP-11` | OP | Proof-carrying composition of local certificates | agent-8 | integration/provenance | `NOT_STARTED` | CONJECTURED` as a general composition system; failure examples are `MACHINE-CHECKED`. | schema validation; CPU/static fixture |
| `OP-12` | OP | Shadow-price scheduling for resource complementarity | agent-8 | integration/provenance | `NOT_STARTED` | CONJECTURED`; finite replay is `EXPERIMENTALLY TESTABLE`. | schema validation; CPU/static fixture |
| `R01` | R | REMORA portable parasitic neural hypervisor | agent-8 | integration/provenance/source-audit | `READY_BUT_BLOCKED` | BLOCKED_NO_CANONICAL_R01_R18_LEDGER | canonical R01-R18 ledger absent; do not alias legacy unpadded R1-R10 |
| `R02` | R | LATCH / REMORA Link host-control ABI | agent-8 | integration/provenance/source-audit | `READY_BUT_BLOCKED` | BLOCKED_NO_CANONICAL_R01_R18_LEDGER | canonical R01-R18 ledger absent; do not alias legacy unpadded R1-R10 |
| `R03` | R | Elastic MTP / DSpark horizon scheduler | agent-8 | integration/provenance/source-audit | `READY_BUT_BLOCKED` | BLOCKED_NO_CANONICAL_R01_R18_LEDGER | canonical R01-R18 ledger absent; do not alias legacy unpadded R1-R10 |
| `R04` | R | Neuralink MTP future packets | agent-8 | integration/provenance/source-audit | `READY_BUT_BLOCKED` | BLOCKED_NO_CANONICAL_R01_R18_LEDGER | canonical R01-R18 ledger absent; do not alias legacy unpadded R1-R10 |
| `R05` | R | PHASE outcome-tree prediction | agent-8 | integration/provenance/source-audit | `READY_BUT_BLOCKED` | BLOCKED_NO_CANONICAL_R01_R18_LEDGER | canonical R01-R18 ledger absent; do not alias legacy unpadded R1-R10 |
| `R06` | R | RSSO | agent-8 | integration/provenance/source-audit | `READY_BUT_BLOCKED` | BLOCKED_NO_CANONICAL_R01_R18_LEDGER | canonical R01-R18 ledger absent; do not alias legacy unpadded R1-R10 |
| `R07` | R | REMORA Metabolism | agent-8 | integration/provenance/source-audit | `READY_BUT_BLOCKED` | BLOCKED_NO_CANONICAL_R01_R18_LEDGER | canonical R01-R18 ledger absent; do not alias legacy unpadded R1-R10 |
| `R08` | R | Hardware-morphic Symbiote / phenotype compiler | agent-8 | integration/provenance/source-audit | `READY_BUT_BLOCKED` | BLOCKED_NO_CANONICAL_R01_R18_LEDGER | canonical R01-R18 ledger absent; do not alias legacy unpadded R1-R10 |
| `R09` | R | Universal parasite receptor | agent-8 | integration/provenance/source-audit | `READY_BUT_BLOCKED` | BLOCKED_NO_CANONICAL_R01_R18_LEDGER | canonical R01-R18 ledger absent; do not alias legacy unpadded R1-R10 |
| `R10` | R | Dual predictor fusion | agent-8 | integration/provenance/source-audit | `READY_BUT_BLOCKED` | BLOCKED_NO_CANONICAL_R01_R18_LEDGER | canonical R01-R18 ledger absent; do not alias legacy unpadded R1-R10 |
| `R11` | R | Dependency-versioned cognition caching | agent-8 | integration/provenance/source-audit | `READY_BUT_BLOCKED` | BLOCKED_NO_CANONICAL_R01_R18_LEDGER | canonical R01-R18 ledger absent; do not alias legacy unpadded R1-R10 |
| `R12` | R | Certified compute skipping | agent-8 | integration/provenance/source-audit | `READY_BUT_BLOCKED` | BLOCKED_NO_CANONICAL_R01_R18_LEDGER | canonical R01-R18 ledger absent; do not alias legacy unpadded R1-R10 |
| `R13` | R | Multi-clock persistent workspace | agent-8 | integration/provenance/source-audit | `READY_BUT_BLOCKED` | BLOCKED_NO_CANONICAL_R01_R18_LEDGER | canonical R01-R18 ledger absent; do not alias legacy unpadded R1-R10 |
| `R14` | R | Adaptive skeleton bodies S0/S1/S2 | agent-8 | integration/provenance/source-audit | `READY_BUT_BLOCKED` | BLOCKED_NO_CANONICAL_R01_R18_LEDGER | canonical R01-R18 ledger absent; do not alias legacy unpadded R1-R10 |
| `R15` | R | Resource-complementarity scheduler | agent-8 | integration/provenance/source-audit | `READY_BUT_BLOCKED` | BLOCKED_NO_CANONICAL_R01_R18_LEDGER | canonical R01-R18 ledger absent; do not alias legacy unpadded R1-R10 |
| `R16` | R | Dense LayerPack + weight-stationary block verification | agent-8 | integration/provenance/source-audit | `READY_BUT_BLOCKED` | BLOCKED_NO_CANONICAL_R01_R18_LEDGER | canonical R01-R18 ledger absent; do not alias legacy unpadded R1-R10 |
| `R17` | R | Latent-trajectory reuse | agent-8 | integration/provenance/source-audit | `READY_BUT_BLOCKED` | BLOCKED_NO_CANONICAL_R01_R18_LEDGER | canonical R01-R18 ledger absent; do not alias legacy unpadded R1-R10 |
| `R18` | R | Elastic Neuralink bus | agent-8 | integration/provenance/source-audit | `READY_BUT_BLOCKED` | BLOCKED_NO_CANONICAL_R01_R18_LEDGER | canonical R01-R18 ledger absent; do not alias legacy unpadded R1-R10 |
| `PFM` | REMORA | Progressive Future Materialization (PFM-A / PFM-B) | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | **PFM-A REJECTED AS DISTINCT MECHANISM; PFM-B DEFERRED** | upstream dependency or authority gate remains open; see source status |
| `REMORA-1` | REMORA | Elastic MTP generation depth | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | DEFERRED behind B0 | upstream dependency or authority gate remains open; see source status |
| `REMORA-10` | REMORA | Dense organ map by value per byte | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-11` | REMORA | REMORA Portion | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-12` | REMORA | REMORA Reclaim | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-13` | REMORA | Computational refrigerator / artifact provenance | agent-8 | integration/provenance | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-14` | REMORA | Value-weighted salvage cache | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-15` | REMORA | Waste Ledger / circular efficiency | agent-8 | integration/provenance | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-16` | REMORA | Tiered Inference Reserve | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-17` | REMORA | Reserve mobilization | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-18` | REMORA | Moving maintenance setpoint | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-19` | REMORA | Uncertainty-adjusted safe surplus | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-2` | REMORA | Elastic verification depth / continuous horizon | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | DEFERRED behind B0 | upstream dependency or authority gate remains open; see source status |
| `REMORA-20` | REMORA | Fast/slow adaptation clocks | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-21` | REMORA | Portable parasitic neural hypervisor | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-22` | REMORA | REMORA Link / host receptor | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-23` | REMORA | REMORA Morph | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-24` | REMORA | REMORA Flow | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-25` | REMORA | REMORA Verify / progressive escalation | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-26` | REMORA | Dependency-versioned cached cognition | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-27` | REMORA | Delta-Certified skipping | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-28` | REMORA | Native hardware-morphic Symbiote | agent-3 | rust-core/ir/control-plane | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-29` | REMORA | Universal learned parasite / Neuralink MTP | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-3` | REMORA | Neuralink/REMORA future packets | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-30` | REMORA | Dense-to-MoE translation | agent-6 | memory/residency/moe-streaming | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-4` | REMORA | Multi-drafter fusion | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-5` | REMORA | PHASE outcome-tree prediction | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-6` | REMORA | RSSO resident skeleton + streamed oracle | agent-5 | compiler/quantization/sidecar | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-7` | REMORA | Layer-stationary speculative wavefront | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-8` | REMORA | Small speculative wavefront tree | agent-2 | language/decode/control | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `REMORA-9` | REMORA | Latent Inertial Drafting | agent-7 | kv/contextfold/context-memory | `READY_BUT_BLOCKED` | DEFERRED | upstream dependency or authority gate remains open; see source status |
| `TBEH` | REMORA | REMORA Tail-Bounded Elastic Horizon | agent-2 | language/decode/control | `NOT_STARTED` | **THEORETICAL / OFFLINE-REPLAY ONLY** | schema validation; CPU/static fixture |

## Assignment authority

- Agent 1: external full-model performance and correctness oracle.
- Agent 2: HAR language, decode/MTP, future packets, PHASE, elastic horizons.
- Agent 3: Rust core, IR, control plane, reference boundary, phenotype compilation; public architecture authority.
- Agent 4: RDNA 4, Vulkan, kernels, GPU sampling, low-level lowering.
- Agent 5: model compiler, quantization, calibration, sidecars, representation lattice.
- Agent 6: NVMe/RAM/VRAM, pages, MoE streaming, transfer scheduler, resource complementarity.
- Agent 7: KV, prefix graph, ContextFold, causal-closure cache, context memory.
- Agent 8: integration, idea registry, language vault, provenance, CI, public-release gate.
