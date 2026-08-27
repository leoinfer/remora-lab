# CODEX MASTER PROMPT — QWEN DENSE → REMORA RESEARCH PROGRAM
## Frozen-host exact-inference investigation, five ideas per batch

You are the main experiment/research agent for the dense-Qwen and REMORA program.

Investigate the complete set of new post-RSSO ideas below **five at a time**, in dependency order, using the real frozen Qwen model and current machine. Do not turn this into an uncontrolled implementation spree. Establish repeatable evidence, falsify cheaply, and modify the hot path only after the relevant batch passes its implementation gate.

# Authoritative state

Hardware:
- AMD Radeon RX 9060 XT 16 GB
- Ryzen 7 3700X
- 32 GB DDR4
- One live GPU agent at a time
- Keep the current normal/max-performance configuration unchanged
- No undervolting, underclocking, reduced power limit, CPU-core disabling, BIOS changes, or hidden ECO mode
- Do not silently enable experimental Vulkan queue flags

Model:
- `[archival master prompt omitted]`
- SHA-256: `3d6ff16be3258f910eac4dcec7142edc7a7100d8400fe363035c8cfedc151164`
- Size: `35,776,484,480 bytes`
- Frozen host: no retraining, fine-tuning, distillation, or weight modification
- Q8_K_XL remains the quality authority

Paths:
- Dense worktree: `[local path omitted]`
- Dense research root: `[local path omitted]`
- Vault: `[local path omitted]`
- Repeatability failure report:
  `[local path omitted]`

Latest valid primary observation:
- MTP off: `1.4800 tok/s`
- MTP on: `2.2030 tok/s`
- MTP acceptance: `117/138`
- Matching final token hash
- Exact 256-token parity passed
- Highest stable observed placement: `NGL=26`
- RSSO decision: `DEFER — INSUFFICIENT EVIDENCE`
- No RSSO hot-path optimization was implemented
- A later repeated performance ladder failed exact parity; those repeated numbers are invalid
- Latest reported vault commit: `f1808d6`

Treat 1.4800 and 2.2030 as valid primary observations, not repeatability-certified baselines.

# Scientific rules

1. Fix repeatability before optimization.
2. Any parity failure invalidates the corresponding performance result.
3. A generic PASS string is insufficient if token parity, state parity, fail-closed counters, or result gating disagree.
4. Never conflate draft positions/s, verifier positions/s, raw target tok/s, accepted drafts, and exact committed target-valid tok/s.
5. Primary performance metric: **exact committed target-valid tok/s**.
6. Every result records wall-clock denominator, token/position denominator, warmup, prompt, token count, context, NGL, MTP config, exact command, git commit, model SHA, output hash, parity, VRAM/GTT/RAM, transfer bytes where available, and power/joules where available.
7. No silent CPU fallback.
8. No stale KV/recurrent state.
9. No staging reuse before its owning fence completes.
10. No result enters ESTABLISHED without raw-log verification.
11. Label STATIC, SIMULATED, MEASURED, ESTABLISHED, FAILED, FALSIFIED, and DEFERRED accurately.
12. Do not infer speedup from reduced bytes alone.
13. Exact modes preserve the frozen host’s authoritative output.

# Vault contract — mandatory after every experiment and batch

Use `[local path omitted]`.

Before writing:
- Acquire `.agent-write-lock`.
- If another agent holds it, create a complete drop under:
  `Inbox/Experiment-Agent-Drops/<timestamp>-codex-remora-dense/`

For every experiment:
- assign an experiment ID;
- record hypothesis and status;
- preserve exact command;
- preserve raw-log path, size, SHA-256, and timestamp;
- preserve code commit, model SHA, and hardware/config fingerprint;
- record parity and fail-closed status;
- record metric definitions, conclusion, and limitations.

Copy small durable artifacts into the vault. Large logs may remain elsewhere only if indexed with absolute path, size, SHA-256, timestamp, command, commit, model hash, experiment ID, and retention status. Nothing may remain only in `/tmp`.

After every batch update:
- Home
- Current-State
- Active-Experiment
- Dashboard
- experiment notes
- architecture notes
- measurement ledger
- claims ledger
- failure ledger
- certificate ledger
- artifact ledger
- chronology
- plans / next actions
- code-path maps

Commit after validation, or create a complete Inbox drop if locked.

Create and maintain:
`REMORA_NEW_IDEA_MASTER_MANIFEST.md`

For every idea record:
- number and name;
- batch;
- falsifiable hypothesis;
- physical bottleneck attacked;
- dense/MoE/hybrid applicability;
- exact/hybrid/approximate;
- frozen-host compatible?;
- training required?;
- dependencies;
- cheapest falsification;
- current evidence;
- implementation status;
- vault links;
- verdict.

Do not begin the next five-idea batch until the current batch has:
1. a written gate verdict;
2. validated artifacts;
3. a vault commit or safe Inbox drop.

Proceed automatically only when the gate allows it. Stop on a correctness blocker, hardware conflict, or architectural falsification requiring an explicit decision.

# Batch 0 — repeatability foundation

This is prerequisite work, not an idea batch.

Goal: turn the primary 1.4800/2.2030 observations into a provenance-complete repeatable baseline or identify why they cannot repeat.

Tasks:
1. Audit `QWEN36_27B_D2_REPEATABILITY_FAILURE.md`.
2. Audit harness reset, prompt reset, KV/recurrent state, sampler, MTP state, callback behavior, result gating, token alignment, hashing, warmup, and timing regions.
3. Run matched interleaved comparisons:
   - A = MTP off
   - B = MTP on
   - minimum order A/B/A/B/A/B
   - identical model, commit, prompt, token count, context, NGL, threads, batches, and environment
   - 256 generated tokens unless shorter diagnostics are required first
4. Require exact parity and matching authoritative output.
5. Report each run and medians: exact committed tok/s, acceptance numerator/denominator, wall time, output hash, VRAM/GTT/RAM, and state/counter deltas.
6. Diagnose all variance or parity failure before proceeding.

Gate B0:
- at least three clean paired A/B comparisons;
- exact output parity;
- trustworthy result gating;
- verified timing denominator;
- no stale state or silent fallback.

Verdicts:
- BASELINE CERTIFIED
- BASELINE CERTIFIED WITH LIMITATIONS
- DEFER — REPEATABILITY BLOCKER
- FALSIFIED — PRIMARY RESULT WAS HARNESS ARTIFACT

Deliver:
- `QWEN36_27B_REPEATABILITY_CERTIFICATE.md`
- `qwen36_27b_repeatability_runs.csv`
- `qwen36_27b_repeatability_summary.json`
- raw-log index
- vault update and commit/drop

Do not touch RSSO/REMORA hot paths until B0 passes.

# Batch 1 — elastic prediction and future control
## Ideas 1–5

1. **Elastic MTP generation depth**
   - Separate maximum trained depth from positions actually computed.
   - Measure marginal draft cost and marginal accepted value per extra MTP stage.
   - Test whether stopping early avoids real work.
   - Prefer prebuilt graph variants; no per-cycle reallocations.

2. **Elastic verification depth and continuous expected horizon**
   - Separate `D_max`, generated `G_t`, verified `V_t`, and accepted `A_t`.
   - A continuous budget such as 1.47 must map to discrete actions using marginal value/confidence, not blind alternation.
   - Execute the next position only when expected accepted value exceeds incremental draft, verify, byte, joule, and opportunity cost.

3. **Neuralink/REMORA future packets**
   - From host MTP/NextN and runtime state predict token survival, semantic boundary, likely correction, required LayerPacks, transfer bytes, verification horizon, latency, and energy.
   - Begin with logged features and a rule-based/offline controller.

4. **Multi-drafter fusion**
   - Compare host MTP, n-gram/local repetition, and later latent-inertia/frozen partial-body drafts.
   - Agreement may control horizon but never establishes correctness.
   - Measure consecutive accepted prefix, cost per accepted token, physical-demand prediction, and exact committed tok/s.

5. **PHASE outcome-tree prediction**
   - While current verification runs, predict accepted-prefix length, bonus/correction token, next draft, and next physical working set.
   - Offline branch budgets B=1,2,4,8.
   - Measure actual-outcome coverage, useful/wasted preparation, memory, fallback, and one-GPU contention.

Batch 1 gate:
- Cost model and offline replay before hot-path code.
- Live prototype only if replay predicts a matched gain after all draft, verify, branch, rollback, memory, and contention costs.

Deliver:
- `REMORA_B1_ELASTIC_PREDICTION.md`
- `elastic_mtp_costs.csv`
- `elastic_horizon_replay.json`
- `future_packet_schema.json`
- `phase_outcome_coverage.csv`
- `REMORA_B1_GATE_REVIEW.md`
- vault update and commit/drop

# Batch 2 — dense weight-stationary execution
## Ideas 6–10

6. **RSSO resident skeleton + streamed oracle**
   - Frozen model only.
   - Report resident bytes, executed original operations, active tensor bytes, draft speed, teacher-forced agreement, and free-running accepted prefixes.
   - Never describe a partial graph as an independently trained smaller model.

7. **Layer-stationary speculative wavefront**
   - Change the execution unit from one token to a causal candidate block.
   - Load/prepare LayerPack i once and process every candidate position while it is resident.
   - Measure weight bytes per block and per exact committed token, activation/state overhead, target positions/s, and exact committed tok/s.

8. **Small speculative wavefront tree**
   - Offline compare linear drafts with small branch trees sharing prefixes.
   - Measure probability coverage, extra compute, memory, target cost, and exact committed-token economics.
   - No heuristic pruning may be called exact.

9. **Latent Inertial Drafting**
   - Teacher-forced/trace-first finite differences, extrapolation, or tiny online low-rank transition fitting without changing host weights.
   - Project predicted anchor state through the output path for drafts.
   - Full target always verifies.

10. **Dense organ map by value per byte**
    - Search scattered operations, not only contiguous early exits.
    - Estimate marginal error reduction divided by latency + weighted joules + weighted bytes.
    - Build nested S0/S1/S2 bodies while preserving mandatory stateful dependencies.

Batch 2 gate:
- Require a valid block-verification path or strong construction evidence;
- measured/simulated weight reuse;
- useful drafter economics;
- exact rollback/state design;
- all-overhead break-even result.

Deliver:
- `REMORA_B2_DENSE_WAVEFRONT.md`
- `dense_organ_map.csv`
- `latent_inertia_results.csv`
- `wavefront_linear_vs_tree.json`
- `rss0_s1_s2_candidates.json`
- `REMORA_B2_GATE_REVIEW.md`
- vault update and commit/drop

# Batch 3 — circular inference / zero unrecoverable waste
## Ideas 11–15

11. **REMORA Portion**
    - Model draft depth, branch width, verification depth, and prefetch as asymmetric overage/underage risk with salvage value.
    - Measure stall/extra-sweep/cold-load cost versus unused compute/bytes/energy/memory cost.

12. **REMORA Reclaim**
    - Classify every surplus artifact as exactly reusable, conditionally reusable, informational only, or truly unrecoverable.

13. **Computational refrigerator / artifact provenance**
    - Record causal-prefix ID, host-state version, input hash, module, shape, precision, location, creation cost, reuse probability, expiry, and validation rule.
    - No reuse without a validity rule.

14. **Value-weighted salvage cache**
    - Compare immediate eviction, LRU, and expected avoided-cost-per-byte admission/eviction.
    - Estimate:
      `V_i = p_reuse * reload_or_recompute_cost - hold_cost - opportunity_cost - validation_cost`

15. **Waste Ledger and circular efficiency**
    - Track useful authority work, accepted/rejected speculation, salvaged work, useful/unused/retained prefetch, avoided reloads/recompute, rollback, expiry, and unrecoverable joules.
    - Future avoided work must be observed, not guessed.

Batch 3 gate:
- Measured avoided future cost must exceed production + holding + validation + eviction cost.
- Exact parity intact.
- VRAM opportunity cost included.
- Improve exact tok/s, joules/token, or both.

Deliver:
- `REMORA_B3_CIRCULAR_INFERENCE.md`
- `artifact_validity_schema.json`
- `salvage_policy_comparison.csv`
- `waste_ledger_schema.json`
- `circular_efficiency_runs.csv`
- `REMORA_B3_GATE_REVIEW.md`
- vault update and commit/drop

# Batch 4 — metabolic reserve and homeostasis
## Ideas 16–20

16. **Tiered Inference Reserve**
    - Hot VRAM, warm RAM, cold/session metadata, and policy-learning “compost.”
    - Measure decay and useful lifetime by artifact class.

17. **Reserve mobilization**
    - Actively spend reserve during correction, uncertainty spikes, congestion, cold demand, semantic boundaries, or compatible future branches.
    - Measure prevented sweeps, reloads, stalls, and recomputation.

18. **Moving computational-maintenance setpoint**
    - Estimate current baseline work required for exact output at desired speed.
    - Inputs include complexity, context growth, MTP acceptance, rejection, required body depth, residency, hardware throughput, thermal state, and reserve.
    - Use live evidence, not prompt labels.

19. **Uncertainty-adjusted safe surplus**
    - Provision:
      `Q_t = estimated_maintenance + safe_surplus - usable_reserve`
    - Increase surplus when uncertainty/deficit cost is high or reserve low.
    - Decrease it under memory pressure, low salvage, full reserve, or target interference.

20. **Fast and slow adaptation clocks**
    - Fast: token/block confidence, acceptance, queues, transfer, reserve use.
    - Slow: task class, context growth, stable hotsets, thermal equilibrium, hardware model, long-run acceptance.
    - Distinguish temporary spikes from a new baseline.

Batch 4 gate:
Compare matched exact policies:
1. Lean
2. Fixed surplus
3. Reclaim
4. Full homeostatic controller

Measure exact tok/s, joules/token, deficit stalls, reserve ROI, unrecoverable waste, memory pressure, and repeatability.

Deliver:
- `REMORA_B4_HOMEOSTASIS.md`
- `maintenance_estimator_schema.json`
- `reserve_state_machine.json`
- `homeostasis_policy_comparison.csv`
- `reserve_roi.csv`
- `REMORA_B4_GATE_REVIEW.md`
- vault update and commit/drop

# Batch 5 — REMORA host attachment and hardware phenotype
## Ideas 21–25

21. **Portable parasitic neural hypervisor**
    - V1 uses rule-based/offline control; no universal learned parasite required.
    - Host remains frozen and authoritative.
    - “Takeover” means reduced exposed host work, never silent replacement.

22. **REMORA Link / elastic host receptor**
    - Define ABI for tokens/logits, hidden taps, MTP output, state snapshots, commit/rollback, graph variants, placement, and counters.
    - Allow canonical state plus host-specific residual state.

23. **REMORA Morph**
    - Build measured hardware fingerprint and resident/warm/streamed/cold phenotype from real microbenchmarks.

24. **REMORA Flow**
    - Schedule prediction, movement, and verification using resource vectors for GPU compute, VRAM BW, PCIe, RAM BW, and CPU.
    - Only claim overlap when the critical path measurably shrinks.

25. **REMORA Verify and progressive escalation**
    - Exact fallback, short/long verification, S0→S1→S2→Full, accepted-prefix commit, rejected-suffix discard, and no stale state.
    - Reuse earlier valid work instead of restarting.

Batch 5 gate:
Build an end-to-end exact controller only after prior batches establish repeatable baseline, useful elastic prediction, useful block/wavefront economics, positive reclaim/reserve economics, and trustworthy state management.

Deliver:
- `REMORA_B5_HOST_HYPERVISOR.md`
- `remora_host_abi.json`
- `remora_hardware_fingerprint.json`
- `remora_phenotype_plan.json`
- `remora_flow_resource_map.csv`
- `REMORA_B5_END_TO_END_GATE.md`
- vault update and commit/drop

# Batch 6 — stronger long-range ideas, design/trace first
## Ideas 26–30

Do not train anything without explicit authorization.

26. **Dependency-versioned cached cognition**
    - Exact reuse requires identical authoritative deterministic inputs.
    - Approximate reuse is draft-only.

27. **Delta-Certified skipping**
    - Investigate rigorous bounds:
      `||z_full - z_cheap||_∞ ≤ ε`
      and top-one margin `γ > 2ε`.
    - Never call empirical calibration a certificate.

28. **Native hardware-morphic Symbiote**
    - Design shared genotype, machine phenotype, and dynamic coarse organs.
    - Record what frozen-host evidence implies about adapters, continued elastic training, or native pretraining.

29. **Universal learned parasite / Neuralink MTP**
    - Design one controller across frozen hosts with host-specific receptors, canonical future-state prediction, and token/semantic/hardware future packets.
    - Separate training-free V1, receptor training, parasite training, and native-model training.

30. **Dense-to-MoE translation**
    - LayerPack→ExpertPack
    - dense wavefront→expert-major multi-position wavefront
    - known layer order→predicted expert union
    - organ map→shared-core/expert value map
    - reserve→persistent expert/route reserve
    - elastic MTP→elastic future expert-union depth
    - No live DeepSeek work while Qwen owns the GPU.

Deliver:
- `REMORA_B6_LONG_RANGE_ARCHITECTURE.md`
- `dependency_reuse_trace_results.csv`
- `delta_certificate_feasibility.md`
- `native_symbiote_training_ladder.md`
- `universal_parasite_design.md`
- `dense_to_moe_transfer_matrix.csv`
- `REMORA_B6_FINAL_SYNTHESIS.md`
- vault update and commit/drop

# Lifecycle for every idea

1. Restate the falsifiable hypothesis.
2. Identify the lower-bound term attacked: compute, VRAM BW, host transfer, synchronization, serial dependency, repeated traversal, speculation waste, energy, or memory opportunity.
3. Audit local code and old branches first. The Q3/older-quant paths are a goldmine.
4. Define exact metrics and denominators.
5. Use the cheapest falsification: static trace, teacher-forced replay, offline simulator, microbenchmark, then live exact prototype last.
6. Define correctness gates before performance work.
7. Use held-out workloads: conversation, explanation, code, repetitive code, math, factual, JSON, long-context retrieval where feasible, and abrupt topic/complexity changes.
8. Preserve the max-performance hardware configuration.
9. Archive all evidence.
10. Issue one verdict: PROCEED, PROCEED WITH REVISED DESIGN, DEFER, or FALSIFIED.
11. Update the manifest and vault.

# Final program outputs

Create:
- `REMORA_COMPLETE_RESEARCH_REPORT.md`
- `REMORA_COMPLETE_CLAIMS_LEDGER.md`
- `REMORA_COMPLETE_FAILURE_LEDGER.md`
- `REMORA_COMPLETE_EXPERIMENT_INDEX.csv`
- `REMORA_DENSE_BEST_CERTIFICATE.md`
- `REMORA_QWEN_TO_DEEPSEEK_TRANSFER_PLAN.md`
- updated `REMORA_NEW_IDEA_MASTER_MANIFEST.md`

Clearly separate:
- Established
- Measured but not established
- Simulated
- Inferred
- Target
- Falsified
- Deferred

# First action now

Begin with **Batch 0 only**.

1. Read the dense foundation docs and repeatability failure report.
2. Verify the exact code commit and commands behind 1.4800/2.2030.
3. Diagnose the parity failure.
4. Produce the interleaved repeatability certificate.
5. Archive and commit everything to the vault.
6. Print a concise checkpoint:
   - baseline verdict;
   - exact per-run numbers;
   - parity;
   - root cause;
   - vault commit/drop;
   - whether Batch 1 is authorized.

Do not skip the vault contract.
