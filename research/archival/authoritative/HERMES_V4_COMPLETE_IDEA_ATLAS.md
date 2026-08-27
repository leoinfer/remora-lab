# HERMES-V4 Complete Idea Atlas

**Authorial origin:** `leoinfer`, with technical refinement and systems formalization developed jointly in conversation.

**Date consolidated:** 2026-08-01

**Scope:** DeepSeek-V4-Flash-0731, hardware-adaptive out-of-core Mixture-of-Experts inference on consumer hardware, especially an AMD Radeon RX 9060 XT 16 GiB system with a Ryzen 7 3700X, 32 GiB DDR4-3200, and NVMe expert streaming.

---

## How to read this document

This document preserves the complete research direction as it developed: the core mechanisms, the user-originated analogies, and the later technical refinements that turned those analogies into testable systems hypotheses.

For each idea, the document separates:

1. **Your original idea — faithful reconstruction.** This preserves the intended meaning and terminology from the conversation and project materials. It is not presented as a verbatim quote unless quotation marks are used. Some original messages were short or conversational, so a perfectly word-for-word transcript is not always available.
2. **Refined technical formulation.** This converts the intuition into an implementable mechanism.
3. **Why it may matter.** The bottleneck or opportunity it addresses.
4. **Minimal experiment.** The cheapest test that could support or kill the hypothesis.
5. **Success and failure criteria.** What would count as evidence, and what would falsify it.
6. **Dependencies and interactions.** Where it belongs in the larger architecture.
7. **Honesty boundary.** What must not be claimed without measurement.

The original discussion often referred to “21 ideas.” Once decomposed, those ideas contain more than 21 distinct engineering mechanisms. This atlas therefore keeps the original families but also separates embedded sub-ideas that deserve their own experiments.

---

# Part I — Project thesis and non-negotiable principles

## 1. The consumer-hardware frontier-model challenge

### Your original idea — faithful reconstruction

Run an enormous frontier-class MoE model, specifically DeepSeek-V4-Flash, on ordinary consumer hardware and push it far beyond the obvious out-of-core baseline. The point is not merely to make the model technically load. The goal is to discover a new inference architecture that treats slow storage, limited VRAM, routing uncertainty, speculation, energy, and recovery as one coordinated system.

The ambition was framed around approximately **30 accepted tokens per second in real chat**, not merely a route-stable warm-cache microbenchmark. That number was a north star, not a promised result.

### Refined technical formulation

The problem is an online constrained scheduling problem over a heterogeneous memory and compute hierarchy:

- VRAM
- host RAM
- NVMe
- optional LAN nodes
- CPU copy engines
- Vulkan transfer and compute queues
- route prediction
- speculative token proposals
- exact or hybrid verification

The runtime should minimize the total cost of a **verified accepted token block**, rather than greedily minimizing the next operation.

A useful top-level objective is:

```text
J(a | s) = E[accepted_verified_tokens
             + alpha * ability_value
             + gamma * future_state_value]
           - lambda * joules
           - mu * latency
           - nu * exposed_bytes
           - xi * wasted_work
           - beta * correctness_risk
           - kappa * future_fatigue
```

subject to memory, queue, thermal, power, quality, exactness, and recovery-reserve constraints.

### Why it may matter

The model activates only a small fraction of its total parameters, yet the active expert payload remains enormous. For the current configuration:

```text
43 routed layers
× 6 experts
× 3 matrices
× 4,456,448 bytes
= 3,449,290,752 bytes
= 3,289.5 MiB
= about 3.2124 GiB logical expert payload per token
```

The central problem is therefore not just arithmetic throughput. It is reducing, reusing, hiding, or avoiding physical expert traffic while maintaining an honest correctness label.

### Minimal experiment

Build a trustworthy stage-by-stage latency and byte ledger for one token and one short generation:

- route computation
- expert cache lookup
- NVMe read
- RAM copy
- staging
- GPU upload
- kernel execution
- synchronization
- verification
- rejection and repair

### Success criterion

The project has a valid foundation when every claimed optimization can be attributed to a measured stage and every result has a correctness class.

### Failure criterion

The 30 tok/s target may be physically impossible on the present system. A rigorous negative result is still valuable if it identifies the limiting roofline and the minimum hardware or representation change required.

### Honesty boundary

Never equate:

- resident/route-stable speed with arbitrary chat;
- logical bytes with exposed physical bytes;
- close logits with exact output;
- isolated kernel speedups with end-to-end generation speedups;
- theoretical speedup products with measured performance.

---

## 2. Full Q8 as the source of truth; exact, hybrid, and approximate modes

### Your original idea — faithful reconstruction

Keep the full Q8 model as the canonical source of truth. Lower-bit paths, skipped experts, route predictions, or speculative blocks can be used aggressively, but they must not quietly redefine what “correct” means.

### Refined technical formulation

Every runtime result must be labeled as one of:

- **Exact / target-equivalent:** canonical target behavior is preserved by a sound acceptance mechanism.
- **Numerically close:** logits are similar, but exact target equivalence is not established.
- **Hybrid:** an approximate path proposes work and a stronger path verifies or repairs it.
- **Approximate:** the final output may differ from canonical Q8.
- **Simulator-only:** no real model execution claim.
- **Unvalidated:** instrumentation or correctness is incomplete.

For deterministic greedy decoding, exactness requires either:

1. matching the canonical autoregressive token sequence across meaningful tests, or
2. a sound zero-false-accept certificate.

For stochastic sampling, target-equivalent claims require valid speculative rejection sampling, not simple agreement heuristics.

### Why it may matter

This rule prevents the project from “winning” by silently changing the model. It also enables honest product modes such as:

- Exact/Eco
- Exact/Balanced
- Exact/Turbo
- Hybrid/Eco
- Hybrid/Balanced
- Hybrid/Turbo

### Minimal experiment

Run canonical Q8, compact full-Q8 transport, and a deliberately degraded low-bit path on the same prompts. Confirm that the evidence system distinguishes all three.

### Success criterion

No run can be promoted as exact without passing the corresponding exactness gate. Timing validity and correctness validity are recorded separately.

### Failure criterion

A single generic `VERIFIED` label that conflates transport, timing, numerical closeness, and target equivalence is unacceptable.

---

# Part II — Core compute and data-movement mechanisms

## 3. Resident compact expert skeleton

### Your original idea — faithful reconstruction

Keep a tiny low-bit “skeleton” of the entire expert system resident in fast memory. Let the skeleton provide a cheap first approximation, while the full Q8 expert payload remains available for correction, widening, or exact verification.

The intuition was that a compact whole-model representation might be more useful than repeatedly pulling full experts from NVMe.

### Refined technical formulation

Store a resident 2-bit, 3-bit, 4-bit, FP8, or otherwise compact representation of routed expert matrices. The skeleton should support one or more of:

- cheap approximate expert output;
- confidence estimation;
- route prediction assistance;
- early rejection of bad speculative positions;
- progressive refinement;
- generation of residual requests for Q8 tiles.

The skeleton is not assumed to replace Q8. Its primary value may be to reduce how often full Q8 is needed.

A refined representation could be:

```text
Q8 expert output = compact skeleton output + exact residual correction
```

or:

```text
cheap skeleton proposes
→ confidence/risk test
→ only uncertain positions request stronger data
```

### Why it may matter

A resident representation attacks the dominant movement problem directly. It can also create a stable low-cost path that Route Scout, DSpark, and the controller can invoke without storage latency.

### Minimal experiment — E1

Test skeleton precision and depth:

```text
precision: 2, 3, 4 bits
routed-layer depth: 4, 12, 24, 43 layers
```

For each configuration measure:

- logit RMS
- cosine similarity
- KL divergence
- top-1 and top-k agreement
- generated-token identity
- first divergence layer/token
- route divergence
- task-level ability retention
- bytes moved
- runtime
- energy

### Expected outcome

The likely result is not that 2-bit or 3-bit replaces all 43 Q8 layers. A more plausible useful region is:

- 4-bit across a limited number of layers;
- 2-bit or 3-bit for coarse proposals, route scouting, or screening;
- mixed precision, where a small subset of sensitive layers receives more bits.

The most important possible discovery is a nonlinear error cliff: a compact path may remain stable for several layers and then abruptly trigger route and token divergence.

### Success criterion

At least one compact configuration produces enough quality or confidence signal to save real Q8 bytes or verification work.

### Failure criterion

If even 4-bit, shallow-depth configurations collapse immediately, the whole-model skeleton should be redesigned or parked.

### Dependencies

Requires:

- verified compact transport;
- reliable full-Q8 control;
- numerical and task-quality harness;
- precise layer-selection controls.

### Honesty boundary

A low RMS value is not automatically useful. It must translate into accepted tokens, avoided bytes, or reliable confidence decisions.

---

## 4. Progressive Energy-Aware MoE Skipper: P1 → P2 → P4 → P6

### Your original idea — faithful reconstruction

Do not always evaluate all six selected experts immediately. Begin with fewer experts and widen only when the current position requires more confidence or quality. Preserve the work already completed so widening does not restart from zero.

### Refined technical formulation

Define nested paths:

- **P1:** top-1 expert
- **P2:** top-2 experts
- **P4:** top-4 experts
- **P6:** canonical top-6 Q8 path

Incremental widening should obey:

```text
y_P2 = y_P1 + w2 * E2(x)
y_P4 = y_P2 + w3 * E3(x) + w4 * E4(x)
y_P6 = y_P4 + w5 * E5(x) + w6 * E6(x)
```

Previously computed expert outputs must be reused. The path controller chooses the lowest-cost level that satisfies the current confidence, quality, or verification requirement.

A path cost model:

```text
C(P_k | s) = immediate_energy
           + lambda * latency
           + p_miss * repair_cost
           + quality_risk
           + future_resource_cost
```

A resident P2 can be cheaper than a cold NVMe P1, so bit width alone cannot determine the action.

### Why it may matter

P1 has roughly one-sixth of the logical top-6 payload. If a large fraction of positions can be handled or screened with P1/P2, exposed traffic could fall dramatically.

### Minimal experiment

For controlled prompts and recorded routes:

1. compute P1, P2, P4, P6 outputs;
2. measure incremental latency and bytes at every widening step;
3. evaluate confidence signals that predict when widening is needed;
4. verify that skipped experts are genuinely not read, staged, uploaded, or computed;
5. measure accepted-token quality and repair cost.

### Success criterion

The progressive policy beats fixed P6 on one of:

- accepted tokens per joule;
- accepted tokens per exposed byte;
- accepted throughput;
- latency under an explicit quality constraint.

### Failure criterion

If most positions immediately widen to P6, or if widening requires recomputing prior work, the mechanism has little value.

### Dependencies

The compact skeleton, route margins, stable slots, and exact evidence harness all improve this idea.

### Honesty boundary

Counting fewer expert FLOPs is not a result unless the corresponding storage, copy, upload, and synchronization work is actually avoided.

---

## 5. Canonical Q8 residual tiles and selective correction

### Your original idea — faithful reconstruction

Use the compact representation as a base, then fetch only the exact information needed to correct it rather than reloading a complete full expert every time.

### Refined technical formulation

Factor each canonical expert into:

```text
W_Q8 = W_skeleton + Delta_W_exact
```

Possible implementations:

- exact residual tiles;
- error-ranked blocks;
- row/column corrections;
- high-energy channel corrections;
- selective projection correction;
- full-expert fallback when residual sparsity is insufficient.

The important property is that correction must be exact where an exact claim is made, and the representation must preserve recoverability of canonical Q8 bytes.

### Why it may matter

The 30 tok/s cold-storage roofline requires exposed bytes per accepted token to fall into the tens-of-MiB range, not merely from 3.2 GiB to 1–2 GiB. Selective residual traffic is one of the few mechanisms with enough theoretical leverage.

### Minimal experiment

For a small set of layers and experts:

- quantize a compact base;
- compute exact residuals;
- rank residual blocks by contribution to output error;
- fetch increasing fractions of residual tiles;
- measure output recovery curve versus bytes.

### Success criterion

A steep recovery curve: most useful accuracy is restored with a small fraction of full Q8 traffic.

### Failure criterion

If exact residual information is dense and requires nearly the complete expert, the representation offers little I/O benefit.

### Dependencies

Requires ExpertPack metadata, kernel-native residual application, and a robust error/quality evaluator.

---

## 6. Route Scout — the MoE branch predictor

### Your original idea — faithful reconstruction

Predict which experts future tokens and layers are likely to need, like a branch predictor. Use the prediction to prefetch data or prepare slots before the true router finishes.

### Refined technical formulation

Predict per future position and routed layer:

- likely top-1 expert;
- likely top-2/top-4 widening set;
- top-6 union recall;
- router margin and entropy;
- confidence;
- likely source location;
- cross-position sharing;
- repair value if wrong.

Candidate signals:

- previous-token route overlap;
- recent route history;
- expert coactivation statistics;
- hidden-state or compact-router features;
- DSpark lookahead;
- current expert residency;
- router margin trends.

The predictor’s objective is not classification accuracy alone. A wrong prediction may still be valuable if the expert is soon reused or remains useful after draft rejection.

### Why it may matter

NVMe and upload latency can be hidden only if the runtime knows what to request early enough. Route Scout creates that lead time.

### Minimal experiment

Start in shadow mode. For lookahead distances 1, 2, 4, 8, and 16 positions, record:

- top-1 accuracy;
- true top-6 union recall;
- bytes prefetched;
- useful and wasted bytes;
- arrival before demand;
- hidden latency;
- post-rejection reuse;
- recovery cost.

### Success criterion

Prediction reduces the critical path or accepted-token cost. A high route-accuracy number alone does not qualify.

### Failure criterion

If prediction arrives too late, creates excessive false traffic, or evicts more valuable experts, it should remain disabled.

### Dependencies

Needs telemetry, stable slots, an asynchronous pipeline, and preferably DSpark future-token hints.

---

## 7. MARC — Margin-Aware Routing Calibration

### Naming note

Within this HERMES-V4 discussion, **MARC** refers to **Margin-Aware Routing Calibration**. This should not be confused with the older broader MARC project, “Modular Architecture with Routing and Control.” The two share a routing philosophy but are distinct concepts.

### Your original idea — faithful reconstruction

Use router confidence and margin, not merely selected expert IDs, to decide how aggressively to approximate, prefetch, skip, or widen.

### Refined technical formulation

MARC consumes:

- router top-1/top-2 margin;
- entropy of the selected distribution;
- Route Scout confidence;
- expert residency and source;
- predicted load time;
- task sensitivity;
- historical widening need;
- verification risk;
- energy and queue state.

It outputs one or more of:

- exact load now;
- prefetch only;
- P1/P2/P4/P6 path;
- defer/cancel;
- safe fallback;
- approximate commit only in explicitly approximate mode.

Begin with calibrated deterministic thresholds, then compare against learned policies.

### Why it may matter

High-margin routing decisions may tolerate narrower paths, while near-ties may require immediate widening. This lets the runtime spend expensive Q8 work where it has the highest marginal value.

### Minimal experiment

Bucket tokens by router margin and measure:

- probability P1/P2 changes the final token;
- logit and route divergence;
- repair probability;
- bytes required for safe widening.

Build reliability curves rather than trusting raw margins.

### Success criterion

MARC beats static path selection under the same quality/correctness constraint.

### Failure criterion

If router margin is poorly calibrated or task-dependent, it must not be used as a sole acceptance signal.

---

## 8. DSpark / MTP future-token canvas

### Your original idea — faithful reconstruction

Use DeepSeek’s speculative module not only to propose future tokens, but also to reveal future route demand. Treat the proposed sequence as a “canvas” of future positions that can be refined, grouped, prefetched, and verified together.

### Refined technical formulation

Desired flow:

1. propose 2–16 future tokens;
2. estimate confidence/stability;
3. predict routes for all positions;
4. build the expert union by layer;
5. batch positions by expert;
6. verify the longest correct causal prefix;
7. salvage useful route, cache, or prefetch work after rejection.

Later positions cannot be committed across an earlier rejection. However, their data movement may still retain value.

The diffusion-style “canvas” analogy is scheduling inspiration only:

- freeze stable positions;
- refine uncertain positions;
- locally widen P1/P2/P4/P6;
- use the canonical target to certify the accepted prefix.

### Current practical constraint

The current 162 GB Unsloth-derived GGUF appears to omit the official `mtp.*` tensors. The official module reportedly includes roughly 4,705 tensors, three speculative MoE layers, Markov weights, a confidence head, and about 10.8 GB of payload. Converter and runtime support must therefore be restored before the full idea can be tested.

### Why it may matter

Longer accepted spans transform the economics. One target verification can amortize expert loads, synchronization, and kernel launches across multiple accepted tokens.

### Minimal experiment

Before full DSpark integration, use teacher-forced future routes to simulate canvases of 2, 4, 8, 12, and 16 positions. Measure:

- unique expert union per layer;
- logical requests versus physical loads;
- acceptance-weighted reuse;
- block cost;
- hypothetical bytes per accepted token.

### Success criterion

The combined proposal-plus-verification path increases accepted throughput after counting draft cost, target verification, rejection, and repair.

### Failure criterion

Low acceptance, rapidly growing expert unions, or sequential “batched” verification that performs k target passes would kill the expected benefit.

### Honesty boundary

Do not import published DSpark speedups as though they transfer to this model, quantization, backend, or hardware.

---

## 9. Expert-major multi-position batching

### Your original idea — faithful reconstruction

Instead of processing token 1, then token 2, then token 3, load an expert once and process every future position that needs it together.

### Refined technical formulation

For a block of positions:

1. collect selected experts by layer;
2. compute the unique expert union;
3. group all requesting positions by expert;
4. load each unique expert once;
5. run one batched expert operation;
6. scatter outputs back to their causal positions.

Define reuse:

```text
R_reuse = logical expert uses / unique physical expert loads
```

### Why it may matter

It converts small, memory-dominated operations into larger, healthier GEMMs and amortizes one expert load across multiple positions.

### Minimal experiment

Use recorded or teacher-forced routes before DSpark exists. For block lengths 2, 4, 8, 12, 16, measure:

- union size;
- repeated expert fraction;
- GEMM shapes;
- occupancy;
- physical loads;
- bytes per position;
- kernel time;
- scatter overhead;
- rare-expert tail latency.

### Success criterion

A real reduction in physical loads and milliseconds per position compared with sequential execution.

### Failure criterion

If unions approach the full expert set too quickly or batching overhead dominates, use smaller adaptive blocks.

### Dependencies

Requires a valid causal/KV schedule, stable slot identities, and true multi-position compact verification.

---

## 10. Persistent expert atlas and stable GPU slots

### Your original idea — faithful reconstruction

Do not constantly treat every expert as a fresh anonymous payload. Maintain a persistent “atlas” that knows where experts live and preserves useful placement across tokens, layers, speculative blocks, and rejected drafts.

### Refined technical formulation

Maintain per-layer state for:

- expert → GPU slot mapping;
- slot → resident expert;
- RAM cache ownership;
- age and reuse score;
- in-flight read/upload state;
- predicted future demand;
- salvage value;
- eviction cost.

Stable slot identity reduces remapping, descriptor churn, copies, and verification ambiguity. It also lets rejected speculative work remain useful.

### Why it may matter

Measured consecutive route overlap is modest, but longer-range and cross-position reuse may be much larger. A stable atlas captures value that token-local caches miss.

### Minimal experiment

Compare:

- no persistent slots;
- LRU;
- LFU;
- Belady trace oracle;
- reuse/value-aware policy.

Measure hit rate, exposed bytes, remap work, upload count, and accepted-token cost.

### Success criterion

Lower exposed traffic or tail latency without cache monopolization.

### Failure criterion

A stable mapping that reduces flexibility, creates fragmentation, or preserves low-value experts should be rejected.

### Important implementation lesson

Slot maps are per layer, not one global map. Verifiers and simulators must reproduce the runtime’s real per-layer state evolution.

---

## 11. ExpertPack — route-aware lossless physical aerodynamics

### Your original idea — faithful reconstruction

Repack routed experts into a format designed for how the runtime actually reads them, like making the data aerodynamically clean. Preserve the original information, but remove fragmentation and unnecessary handling.

### Refined technical formulation

Create a reversible indexed format where each expert is stored contiguously:

```text
[gate | up | down]
```

with metadata for:

- layer;
- expert ID;
- file offset;
- stored and decoded size;
- quantization;
- alignment;
- checksum;
- staging layout;
- GPU layout;
- preferred stable slot;
- coactivation statistics;
- residual-tile map.

Investigate:

- `preadv`;
- `io_uring`;
- sorted extents;
- cancelable requests;
- hot packs;
- P1-first physical ordering;
- coactivation clustering;
- direct reads into kernel-native staging;
- exact reconstruction of original quantized bytes.

### Why it may matter

Even without reducing logical bytes, ExpertPack may reduce:

- read amplification;
- fragmented I/O;
- system calls;
- CPU packing;
- staging copies;
- upload count;
- descriptor and dispatch overhead;
- cache churn.

### Minimal experiment

Pack a few layers. Replay identical expert routes against GGUF and ExpertPack. Compare:

- extents per token;
- cold and warm read time;
- CPU time;
- staging time;
- upload time;
- reversible hash identity;
- end-to-end logits.

### Success criterion

A measurable I/O or packing improvement under identical routes, with exact reversibility.

### Failure criterion

If the original GGUF layout is already sufficiently contiguous or the repacking cost/storage overhead outweighs gains, park it.

### Honesty boundary

ExpertPack alone is not a 30 tok/s mechanism. It is an enabling physical-layout improvement.

---

## 12. RDNA4-native kernel path

### Your original idea — faithful reconstruction

Once the winning architecture is known, make the execution path native to the actual AMD GPU rather than relying on generic compromises.

### Refined technical formulation

Potential targets:

- Wave32-aware work decomposition;
- fused dequantization + matmul;
- shader-native expert packing;
- persistent descriptors;
- persistent expert slots;
- expert-major multi-position MMID kernels;
- fewer dispatches and fences;
- overlapped transfer and compute;
- minimized intermediate tensors;
- efficient incremental P1→P2→P4→P6 widening.

Separate:

1. canonical, lossless Q8 execution;
2. compact lower-bit skeleton execution.

### Why it may matter

Traffic reduction should come first, but after architecture-level savings, kernels can become the next bottleneck. Larger expert-major batches may finally expose shapes where custom RDNA4 kernels matter.

### Minimal experiment

Benchmark isolated kernel shapes that correspond to the real winning path, not arbitrary GEMMs. Compare:

- generic baseline;
- forced existing path;
- custom prepacked path;
- end-to-end token effect.

### Success criterion

Repeated same-command A/B/A/B gains with numerical validation and no shifted bottleneck that erases the end-to-end benefit.

### Failure criterion

A large isolated kernel multiplier with negligible accepted-token improvement is not enough.

### Honesty boundary

An external report of a “7×” kernel/config speedup is a research lead until reproduced on the exact model, hardware, path, and correctness harness.

---

## 13. Asynchronous three-lane pipeline

### Your original idea — faithful reconstruction

Run drafting, data movement, and verification as separate lanes so that one stage can work while another waits. The runtime should behave like a pipeline rather than a serial loop.

### Refined technical formulation

Functional stages:

1. **Draft / Route Scout lane**
2. **Data movement lane**
3. **Verification / repair / commit lane**

Ideal cycle:

```text
T_cycle ≈ max(T_draft, T_IO, T_verify) + T_merge
```

rather than:

```text
T_cycle = T_draft + T_IO + T_verify
```

### Why it may matter

Even when bytes cannot be eliminated, some latency can be hidden behind useful compute.

### Minimal experiment

Build a timestamped timeline for one block showing:

- route available;
- read issued/completed;
- staging ready;
- upload submitted/completed;
- kernel start/end;
- verification;
- commit.

### Success criterion

Measured overlap reduces critical-path time without corrupting staging, violating causality, or exhausting memory.

### Failure criterion

Apparent overlap that merely moves work into queues or increases p95 latency is not useful.

### Key implementation caveat

Staging slices must be submission-owned or fence-protected. A slice cannot be reused while pending GPU work still references it. CPU worker completion must also be correctly ordered, especially when non-temporal stores are used.

---

# Part III — Scheduling, control, and traffic metaphors made concrete

## 14. Global telemetry “satellite”

### Your original idea — faithful reconstruction

Give the runtime a satellite view of the whole system. It should see traffic, congestion, accidents, roadworks, tailwinds, queues, energy, and future demand rather than making blind local decisions.

### Refined technical formulation

Track:

- GPU compute/copy queues;
- NVMe queue depth and latency;
- RAM/VRAM pressure;
- per-layer slot maps;
- in-flight reads and transfers;
- route margins;
- DSpark confidence;
- accepted span lengths;
- widening rates;
- useful/wasted prefetch;
- temperature, clocks, throttling;
- energy;
- KV growth;
- policy state;
- correctness class.

“Accidents” are transient events:

- P1 miss;
- stalled read;
- rejected prefix;
- eviction;
- blocked copy.

“Roadworks” are persistent conditions:

- KV pressure;
- background storage load;
- thermal slowdown;
- bad shader path;
- repeatedly broad unions.

### Why it may matter

Every adaptive idea depends on accurate state. A controller without telemetry optimizes an imagined machine.

### Minimal experiment

Implement compact binary hot-path events plus background aggregation. Measure instrumentation overhead in off, summary, sampled, and full modes.

### Success criterion

The telemetry predicts at least one future stall or explains one tail-latency event while adding negligible overhead in normal mode.

### Failure criterion

Verbose synchronous logging in the hot path is itself a performance bug.

---

## 15. Dynamic MoE GPS / verified-token navigator

### Your original idea — faithful reconstruction

Treat inference like live GPS navigation. The cheapest next road is not always the best route to the destination. The runtime should replan as traffic and hardware state change.

### Refined technical formulation

Model the runtime as a dynamic graph or receding-horizon controller.

State includes:

- token/block position;
- layer;
- draft and verification status;
- resident experts;
- in-flight transfers;
- slots;
- path width;
- queue state;
- thermal/fatigue state;
- energy budget;
- correctness mode.

Actions include:

- run resident path;
- fetch from a source;
- widen;
- wait or cancel;
- batch;
- change lane;
- commit prefix;
- recover;
- shrink canvas.

Path cost:

```text
c(edge) = latency
        + lambda * energy
        + mu * exposed_bytes
        + nu * wasted_work
        + xi * future_fatigue
        + quality/risk penalty
```

The controller repeatedly:

1. plans;
2. executes the first action;
3. ingests telemetry;
4. replans.

### Why it may matter

A locally cheap action can create expensive future repair or destroy valuable residency. Receding-horizon planning can account for future reuse and recovery.

### Minimal experiment

Compare on trace replay:

- static rules;
- greedy next-step policy;
- simple receding horizon;
- oracle with future trace knowledge.

### Success criterion

The live planner captures a meaningful fraction of oracle savings with low overhead.

### Failure criterion

If planning overhead approaches the savings, use simpler rules.

---

## 16. Safe, Balanced, and Autobahn risk lanes

### Your original idea — faithful reconstruction

Give inference different driving lanes. A safe lane should be conservative; a balanced lane should take measured risks; an Autobahn lane should exploit favorable conditions aggressively.

### Refined technical formulation

**Safe**

- small canvas;
- conservative P2/P4;
- strong residency preference;
- high recovery reserve;
- strict verification.

**Balanced**

- medium canvas;
- P1 with planned P2 recovery;
- moderate lookahead;
- normal batching.

**Autobahn**

- largest currently profitable canvas;
- aggressive lookahead;
- high parallelism;
- exploit tailwind.

Autobahn constraint:

```text
expected_miss_cost
+ future_fatigue
+ queue_risk
+ quality_risk
<= risk_budget
```

Always reserve an emergency lane:

- VRAM slots;
- staging capacity;
- cancelable reads;
- GPU budget;
- verification bandwidth.

### Why it may matter

A single fixed policy cannot be optimal across cold starts, warm route-local runs, thermal saturation, or high uncertainty.

### Minimal experiment

Replay identical workloads through fixed Safe, Balanced, and Autobahn policies. Measure throughput, tails, wasted work, and recovery success.

### Success criterion

The lane selector improves the latency/energy frontier and can retreat safely from aggressive mode.

### Failure criterion

“Autobahn” must not become an excuse to ignore correctness or reserve capacity.

---

## 17. Motorway merges, on-ramps, and ramp metering

### Your original idea — faithful reconstruction

New work should not flood the system merely because parallelism is available. Control how speculative and recovery work enters the pipeline, like motorway ramp metering.

### Refined technical formulation

Incoming work includes:

- completed expert read;
- P2/P4 recovery;
- full-Q8 fallback;
- new speculative block;
- KV commit;
- newly resident expert.

Priority rule:

```text
canonical verification/recovery
> high-confidence reusable prefetch
> low-confidence speculation
```

Ramp constraint:

```text
new speculative volume
<= predicted downstream free capacity
```

Low-priority speculative work must yield to high-value recovery. Zipper-merge fairness prevents starvation.

### Why it may matter

Uncontrolled concurrency creates queue explosions, cache churn, and p95 collapse. More parallel work can make the system slower.

### Minimal experiment

Inject increasing speculative volume and compare:

- unlimited admission;
- fixed cap;
- telemetry-based ramp metering.

### Success criterion

Higher sustained accepted throughput and better tails without starving speculation.

### Failure criterion

If ramp control reacts too slowly, it may oscillate. Add hysteresis and predicted queue debt.

---

## 18. Broadband / multi-source expert fabric

### Your original idea — faithful reconstruction

Treat expert data like broadband traffic that can come from multiple sources: VRAM, RAM, NVMe, another machine, or remote storage. Choose the source that will deliver the expert soonest and most economically.

### Refined technical formulation

Possible sources:

- VRAM;
- local RAM;
- local NVMe;
- second NVMe;
- LAN RAM;
- LAN GPU;
- remote depot for noncritical prefetch.

Arrival estimate:

```text
T_arrival(source) = startup_latency
                  + bytes / effective_rate
                  + queue_delay
                  + decode/upload_cost
```

Selection:

```text
source* = argmin(T_arrival
                 + lambda * energy
                 + opportunity_cost
                 + p_failure * repair_cost)
```

### Why it may matter

A second compute box or RAM-rich node could act as an expert depot. The future two-DGX-Spark dream fits naturally here: one machine need not duplicate every role.

### Minimal experiment

Start locally: choose among VRAM, RAM, and NVMe using measured arrival time. Only then test a LAN RAM depot.

### Success criterion

A second source reduces real accepted-token critical-path cost.

### Failure criterion

Aggregate bandwidth that does not reduce the critical path is not a win. Ordinary internet should not enter the token-critical path by default.

---

## 19. Tailwind, headwind, and the throughput sweet spot

### Your original idea — faithful reconstruction

The system sometimes has a tailwind: experts are resident, routes repeat, queues are empty, speculation is accepted, and shaders are warm. Other times it has a headwind. The runtime should sense this and adjust speed rather than using a fixed throttle.

### Refined technical formulation

Tailwind score may include:

- residency;
- reuse;
- prefetch hits;
- hidden transfer;
- DSpark acceptance;
- warm shaders;
- empty queues;
- nearly completed reads;
- salvageable rejected work.

Example:

```text
W = a*resident
  + b*reuse
  + c*prefetch_hit
  + d*acceptance
  + e*hidden_transfer
  - f*waste
  - g*queue_pressure
```

Use W to adjust:

- canvas size;
- lane;
- prefetch depth;
- path width;
- concurrency.

Find the throughput knee using marginal cost:

```text
MC(v) = delta_joules / delta_accepted_tokens_per_second
```

### Why it may matter

Peak burst throughput may be energetically or thermally unsustainable. The best operating point may lie below maximum instantaneous speed.

### Minimal experiment

Sweep concurrency and canvas size after thermal stabilization. Plot accepted tok/s, joules/token, p95, and queue debt.

### Success criterion

Identify stable Eco, Balanced, Turbo, and Sustained regimes with repeatable boundaries.

### Failure criterion

Do not invent a universal cubic or biological law. The curve must be measured on the real hardware.

---

## 20. Fatigue, recovery, and RIR

### Your original idea — faithful reconstruction

Borrow from training/bodybuilding: systems accumulate fatigue, need recovery, and should preserve “reps in reserve.” Do not push every resource to failure on every token.

### Refined technical formulation

Resource-specific fatigue:

```text
F_r(t + dt) = F_r(t) * exp(-dt / tau_r) + load_r(t)
```

Fatigue can mean:

- queue debt;
- thermal pressure;
- cache churn;
- fragmentation;
- slot pressure;
- false-prefetch debt;
- policy instability;
- synchronization backlog.

RIR means reserve headroom in:

- VRAM;
- I/O queues;
- thermal budget;
- rollback capacity;
- latency;
- recovery slots.

Policy:

- low fatigue → larger blocks and more speculation;
- medium fatigue → balanced operation;
- high fatigue → active-recovery blocks and reduced admission.

### Why it may matter

Short benchmark peaks can hide eventual collapse. A fatigue-aware controller seeks sustainable accepted throughput.

### Minimal experiment

Run long generations with and without fatigue state. Compare throughput drift, temperatures, queue oscillation, cache churn, and tail latency.

### Success criterion

Fewer collapses and better sustained efficiency.

### Failure criterion

Do not force artificial “rest” when no measured resource is stressed.

---

## 21. Compound versus isolation

### Your original idea — faithful reconstruction

Some work should be compounded together because batching and reuse create returns. Other errors should be isolated and repaired locally rather than forcing the whole block through a heavy path.

### Refined technical formulation

Compound operations:

- expert-major block GEMM;
- fused kernels;
- compound prefetch;
- multi-position verification;
- shared recovery.

Isolated operations:

- one expert;
- one position;
- one local widening;
- one failed layer.

A speculative return measure:

```text
SFR = (expected accepted tokens + future reuse)
      / (latency + lambda*energy + mu*waste + nu*future_fatigue)
```

### Why it may matter

Over-batching increases union size and repair blast radius. Under-batching loses amortization. This idea defines the boundary.

### Minimal experiment

For identical route traces, compare global block repair with local position/layer repair.

### Success criterion

The controller finds a block size where shared work outweighs union and rejection cost.

### Failure criterion

A giant block that produces impressive reuse but poor accepted-token economics should be rejected.

---

## 22. Macro resource allocation: carbs, protein, and fat

### Your original idea — faithful reconstruction

Use a bodybuilding/macronutrient analogy for runtime budgets:

- carbs = productive fast-path work;
- protein = verification and correction;
- fat = reserve and headroom.

### Refined technical formulation

Let:

```text
Budget = C + P + F
```

where:

- **C:** speculative/draft throughput work;
- **P:** exact verification, correction, and fallback;
- **F:** reserved memory, bandwidth, power, and recovery capacity.

Maximize C only after minimum safe P and F are reserved.

### Why it may matter

An aggressive runtime can consume all resources on speculative work and leave no capacity to verify or recover. This abstraction makes reserve allocation explicit.

### Minimal experiment

Sweep reserve percentages and measure accepted throughput, failure recovery, and queue collapse.

### Success criterion

A nonzero reserve improves sustained accepted tokens despite slightly reducing peak speculative work.

### Failure criterion

No literal dietary percentages should be imposed. The analogy exists to define resource classes, not physiology.

---

## 23. Water-purification cascade

### Your original idea — faithful reconstruction

Treat inference as progressively purifying uncertainty. Cheap stages remove obvious uncertainty; expensive stages are used only when the result remains contaminated or risky.

### Refined technical formulation

Uncertainty vector:

```text
u = [token uncertainty,
     router entropy,
     expert-miss risk,
     target divergence,
     verification risk]
```

Stages:

- coarse screening;
- coagulation/sedimentation = group expert needs;
- sand filter = P1/P2;
- activated carbon = selective P4;
- reverse osmosis = full P6/Q8;
- UV/chlorination = final certification.

Transition:

```text
u_(j+1) = M_j * u_j + epsilon_j
```

Choose the cheapest chain satisfying the task threshold:

```text
minimize sum(energy_j + lambda * time_j)
subject to final_uncertainty <= threshold
```

### Why it may matter

It provides a concrete staged stopping policy. Stable positions stop early; uncertain positions receive deeper treatment.

### Minimal experiment

Calibrate how each stage reduces each component of uncertainty and how often stages can be skipped.

### Success criterion

The cascade reduces average work while escalating correctly on hard positions.

### Failure criterion

If uncertainty signals are uncalibrated, early stopping may create silent quality loss.

---

## 24. Sunscreen / protection-budget allocation

### Your original idea — faithful reconstruction

Treat future speculative positions like exposed skin. Some positions face more risk, remain exposed longer, or are more causally important. Allocate verification “protection” where it reduces the most risk.

### Refined technical formulation

For speculative position i:

```text
D_i = intensity_i * exposure_time_i * sensitivity_i
```

Protection x_i gives residual risk:

```text
R_i(x_i) = D_i * exp(-k_i * x_i)
```

Optimize productive speculative surface subject to:

```text
sum(energy_i) <= budget
sum(residual_risk_i) <= risk_budget
critical_prefix_risk <= threshold
```

Protection ROI:

```text
-delta_risk / delta_energy
```

Map P1/P2/P4/P6 to increasing protection. “Reapplication” means widening when telemetry changes.

### Why it may matter

Early speculative positions deserve stronger protection because one weak prefix point invalidates all later commits.

### Minimal experiment

Compare uniform verification across positions with risk-weighted verification under the same quality constraint.

### Success criterion

More accepted tokens or lower energy at the same causal-prefix failure rate.

### Failure criterion

Do not let later high-confidence positions distract from a weak early prefix.

---

# Part IV — Learning, arbitration, and adaptive structure

## 25. Behaviorism / consequence-driven learning

### Your original idea — faithful reconstruction

Let the runtime learn from consequences, not from elegant internal stories. Reward actions that produce verified accepted tokens cheaply and punish waste, repair, quality loss, and future resource debt.

### Refined technical formulation

Actions:

- canvas size;
- path width;
- lane;
- prefetch;
- source;
- priority;
- slot policy;
- recovery timing.

Reward:

```text
r = accepted_verified_tokens
  - lambda * energy
  - mu * latency
  - nu * wasted_bytes
  - xi * repair_cost
  - kappa * quality_penalty
  - omega * future_fatigue
```

Avoid reward hacking:

- route accuracy alone is not enough;
- tok/s alone is not enough;
- hit rate alone is not enough;
- local KPIs must improve global verified-token economics.

Deployment progression:

1. fixed rules;
2. contextual bandit in shadow mode;
3. bounded online authority;
4. rollback on drift.

### Why it may matter

The best policy may depend on hardware state, task type, context, and recent outcomes in ways that fixed thresholds cannot capture.

### Minimal experiment

Train on trace replay and evaluate on held-out prompts and thermal/storage states. Compare against frozen rules.

### Success criterion

Better held-out accepted-token cost with bounded regressions and immediate fallback.

### Failure criterion

Do not begin unrestricted online RL on an unverified runtime.

---

## 26. Freud: Id, Superego, and Ego

### Your original idea — faithful reconstruction

Use a Freud-inspired software metaphor:

- Id wants aggressive speed and speculation;
- Superego enforces constraints;
- Ego chooses what is realistic now.

### Refined technical formulation

```text
candidate_actions = Id_proposer(state)
feasible_actions = Superego_filter(candidate_actions, state)
selected_action = Ego_argmax(risk_adjusted_value)
```

**Id** proposes:

- larger canvases;
- narrower paths;
- more prefetch;
- more concurrency.

**Superego** enforces:

- correctness;
- energy;
- memory;
- recovery reserve;
- viability;
- quality constraints.

**Ego** uses the GPS/value model to choose among feasible actions.

### Why it may matter

This is a clean software architecture: proposal, hard constraint filtering, and realistic arbitration.

### Minimal experiment

Ablate the constraint filter and show which unsafe actions it blocks; ablate the proposer and measure lost opportunity.

### Success criterion

The system avoids unsafe aggressive choices without collapsing into permanent conservatism.

### Failure criterion

Do not extend the metaphor into claims about psychology or consciousness.

---

## 27. Investment, capital allocation, compound interest, and salvage value

### Your original idea — faithful reconstruction

Treat runtime resources as capital. Every load, prefetch, slot assignment, and speculative action is an investment with immediate return, future dividends, risk, fees, opportunity cost, liquidity, sunk cost, and salvage value.

### Refined technical formulation

Capital includes:

- joules;
- milliseconds;
- bandwidth;
- VRAM slots;
- RAM;
- GPU occupancy;
- thermal headroom.

Action value:

```text
Value(action) = immediate_value
              + gamma * expected_future_reuse
              + salvage_value_after_rejection
              - load_cost
              - opportunity_cost
              - beta * risk
```

### Why it may matter

A prefetch that misses the immediate token may still be profitable if the expert remains resident and is reused. Conversely, a high hit-rate policy can be bad if it monopolizes scarce slots.

### Minimal experiment

Compare:

- latency-only greedy;
- byte-only greedy;
- immediate-hit policy;
- future-value/salvage-aware policy.

### Success criterion

The capital-aware policy improves total verified-token value, not a private subsystem KPI.

### Failure criterion

No subsystem may optimize its own hit rate, occupancy, or route score while worsening global cost.

---

## 28. Inference-IQ / ability retention per joule

### Your original idea — faithful reconstruction

Measure not only whether tokens match, but how much useful model ability remains under cheaper paths. Create an “Inference IQ” or ability-per-joule concept for the model, while being explicit that this is not human IQ.

### Refined technical formulation

Create a controlled task battery covering:

- reasoning;
- math;
- code;
- instruction following;
- planning;
- factual reliability;
- long-context sensitivity;
- near-tie logits.

An IRT-like model:

```text
P(correct_i | theta) = sigmoid(a_i * (theta - b_i))
```

Optional standardized score:

```text
IQ_inf = 100 + 15 * theta
```

This must always be labeled a model-specific standardized inference score.

Ability retention:

```text
R_ability(P_k) = (theta_Pk - theta_baseline)
                 / (theta_Q8 - theta_baseline)
```

Efficiency:

```text
eta(P_k) = R_ability(P_k) / energy(P_k)
```

Marginal ability per joule:

```text
E[theta_(k+1) - theta_k]
/ E[energy_(k+1) - energy_k]
```

### Why it may matter

Logit RMS can move while capability remains intact, or remain small while a critical task fails. Ability-level measurement gives the controller a more meaningful quality objective.

### Minimal experiment

First prove the battery distinguishes canonical Q8 from a deliberately degraded path. Then compare P1/P2/P4, low-bit skeletons, and hybrid verification.

### Success criterion

Stable, reproducible quality-efficiency curves that detect real degradation.

### Failure criterion

Do not present this as a psychometric measurement of human intelligence.

---

## 29. Maturana / autopoiesis / viability governor

### Your original idea — faithful reconstruction

A system should preserve the conditions that allow it to keep operating and recovering. Use Maturana/autopoiesis as a control principle, not as a claim that the runtime is alive.

### Refined technical formulation

Define a viability region:

```text
V = {state:
     temperature < T_max,
     SSD_queue < Q_max,
     free_recovery_slots >= R_min,
     error_risk < epsilon,
     latency_debt < L_max,
     policy_drift < D_max}
```

Aggressive actions are allowed only when the predicted next state remains in V.

The governor maintains:

- recovery capacity;
- bounded queues;
- thermal sustainability;
- correct telemetry;
- stable policy;
- cache health;
- verification availability.

It may:

- shrink canvas;
- force lane change;
- pause learning;
- drain queues;
- rebuild reserve;
- revert policy;
- force exact mode;
- recalibrate.

### Why it may matter

A throughput-maximizing controller can drive the runtime into a state where it cannot verify, recover, or remain thermally stable.

### Minimal experiment

Inject overload, bad predictions, queue saturation, thermal pressure, and cache churn. Compare governor on/off.

### Success criterion

The system preserves correctness and recovery instead of maximizing short-term speed until collapse.

### Failure criterion

Do not anthropomorphize the mechanism as life or consciousness.

---

## 30. Wolff’s law / Inference Mechanostat

### Your original idea — faithful reconstruction

Borrow Wolff’s law: structures adapt over time to repeated load. The inference system should slowly remodel its persistent structure according to verified demand, rather than only reacting token by token.

### Refined technical formulation

Compute a slow structural score:

```text
structural_value = (verified accepted-token value
                    + future reuse)
                   / (latency
                      + energy
                      + exposed bytes
                      + monopolization penalty)
```

Adapt slowly:

- stable VRAM slot allocation;
- RAM cache capacity by layer;
- ExpertPack co-location;
- prefetch priority;
- staging and queue capacity;
- kernel specialization;
- mixed-precision skeleton allocation.

Use:

- hysteresis;
- minimum sample counts;
- holdout validation;
- rollbackable profiles;
- anti-monopolization penalties.

### Why it may matter

The optimal persistent structure may depend on long-run workload patterns that a token-level controller cannot infer reliably.

### Minimal experiment

Compare long-run adaptation against static placement, LRU, and LFU on held-out workloads.

### Success criterion

Improved long-run accepted-token cost without overfitting to one prompt family.

### Failure criterion

Rapid structural changes are not mechanostat behavior; they are instability. Remodeling must be slow and evidence-based.

---

## 31. Neuro-inspired modular inference control plane

### Your original idea — faithful reconstruction

Model the control system as specialized brain-like modules with different responsibilities and timescales. The point is modular control architecture, not biological imitation.

### Refined technical formulation

Possible mapping:

- **Thalamus:** event/work dispatcher;
- **Amygdala:** fast risk interrupt;
- **Prefrontal system:** slower planner;
- **Basal ganglia:** action selection;
- **Hippocampus:** episodic trace retrieval;
- **Cerebellum:** timing prediction and error correction;
- **Hypothalamus:** power, thermal, and queue homeostasis;
- **Brainstem:** canonical Q8 emergency path;
- **Glial layer:** cleanup, compaction, maintenance;
- **Shared state bus:** cross-module communication.

Fast reflexes must remain separate from slower planning and learning.

### Why it may matter

A monolithic controller becomes hard to debug and impossible to ablate. Functional modules make authority, latency, and failure boundaries explicit.

### Minimal experiment

Implement the smallest modular split:

1. fast safety interrupt;
2. normal rule-based selector;
3. slow planner;
4. telemetry state bus.

Ablate each module.

### Success criterion

Each module has a measurable responsibility and removing it creates a predictable degradation.

### Failure criterion

A module that merely renames an existing function adds no value.

---

## 32. Startup autotuner

### Your original idea — faithful reconstruction

Let the runtime measure the actual machine at startup and choose settings for this exact GPU, driver, storage, model, and context instead of relying on universal defaults.

### Refined technical formulation

Modes:

- off;
- quick;
- full.

Measure:

- cold/warm NVMe;
- queue depths;
- contiguous/random reads;
- RAM copy;
- pinned memory;
- upload;
- overlap;
- P1/P2/P4/P6 kernels;
- block/batch sizes;
- slot counts;
- synchronization;
- power/energy;
- thermal stabilization;
- repair and cancellation cost.

Profile key includes:

- GPU;
- driver;
- Vulkan runtime;
- CPU/RAM;
- storage;
- model hash;
- quantization;
- shader hash;
- context/KV state;
- ExpertPack version;
- staging/slot configuration.

### Why it may matter

The best queue depth, staging size, and kernel can change after a driver update or with a different model/context.

### Minimal experiment

Tune a small action space, cache the profile, restart, and verify that the selected settings reproduce their gain.

### Success criterion

The profile invalidates itself when relevant identity changes and consistently improves the chosen objective.

### Failure criterion

An autotuner has little value before there are real alternative mechanisms to tune.

---

## 33. Empirical certificates and the Explorer/Verifier workflow

### Your original idea — faithful reconstruction

Treat every strong systems claim as requiring a certificate: a compact, replayable evidence package showing exactly what was tested, what passed, and what remains uncertain.

### Refined technical formulation

Evidence levels:

- **L0:** idea only;
- **L1:** code path exists;
- **L2:** marker/path evidence;
- **L3:** correctness evidence;
- **L4:** repeated performance evidence;
- **L5:** ablation/revert evidence;
- **L6:** independently replayable evidence.

Every certificate should include:

- source commit and diff hash;
- executable/library hashes;
- model/index hashes;
- hardware and software environment;
- exact command;
- prompt/seed/tokens;
- cache and thermal state;
- raw logs;
- parser version;
- correctness class;
- timing validity;
- fallback status;
- known limitations.

An **Explorer** may run broad diagnostics and generate hypotheses. A **Verifier** must be narrow, deterministic, and hostile to false positives.

### Why it may matter

The project has already demonstrated why this matters: a transport verifier using the wrong slot-map model produced apparent mismatches. The evidence system itself must be tested and falsifiable.

### Minimal experiment

Inject known failures:

- wrong expert IDs;
- wrong slot order;
- missing copy;
- stale payload;
- invalid marker;
- parser misattribution.

Confirm the certificate rejects each one.

### Success criterion

A certificate can be independently replayed and distinguishes mechanism validity, numerical validity, and end-to-end inference validity.

### Failure criterion

The analogy to formal proof certificates is procedural, not logical. An empirical certificate is not a mathematical proof.

---

## 34. Reasoning-distilled inference controller

### Your original idea — faithful reconstruction

Use an expensive, intelligent offline controller to reason through many possible hardware actions, then distill those decisions into a small, fast runtime policy.

### Refined technical formulation

An offline oracle searches over:

- P1/P2/P4/P6;
- canvas length;
- prefetch plan;
- cache/slot decisions;
- kernel choice;
- queue/synchronization policy;
- risk lane;
- recovery plan.

Store structured traces:

- state;
- candidate actions;
- predicted outcomes;
- chosen action;
- actual outcome;
- reason codes;
- counterfactual failures.

Train a bounded student policy. Do not depend on unrestricted natural-language chain-of-thought. Use structured rationales and measurable state/action targets.

Deployment:

1. offline evaluation;
2. shadow mode;
3. bounded authority;
4. conservative fallback;
5. rollbackable profile;
6. slow adaptation only after stability.

### Why it may matter

A sophisticated planner may be too expensive to run every token, but its decisions can supervise a small controller.

### Minimal experiment

Choose one bounded decision, such as canvas length or P1/P2/P4/P6 selection. Generate oracle labels on traces and compare student, rule baseline, and oracle.

### Success criterion

The student captures most oracle value with negligible runtime overhead and bounded regressions.

### Failure criterion

Do not train the controller until telemetry, actions, rewards, and correctness labels are stable. Otherwise it learns bugs and noise.

---

# Part V — Experimental programs and cross-cutting hypotheses

## 35. E1 — compact-skeleton fidelity experiment

### Purpose

Determine whether a low-bit resident expert skeleton has a useful operating region.

### Matrix

```text
bits: 2, 3, 4
compact routed-layer depth: 4, 12, 24, 43
control: full-Q8 compact transport
```

### Measurements

- logit RMS;
- cosine and KL;
- top-1/top-k agreement;
- generated-token identity;
- first divergence;
- route divergence;
- ability retention;
- bytes and time;
- fallback and path markers.

### Current placement

E1 begins only after the full-Q8 compact control and verifier are trustworthy. Debugging the control is not yet the low-bit experiment itself.

### Predicted qualitative outcome

- 4-bit shallow configurations are the strongest candidate;
- 3-bit may work for proposals or a subset of layers;
- 2-bit is more likely a coarse screening signal;
- full-depth low-bit replacement is unlikely;
- layer sensitivity and mixed precision may matter more than a global bit width.

### Scientific value of a negative result

If full-depth skeletons fail but shallow or selective layers work, the result still supports progressive certification and mixed precision.

---

## 36. E2 — attention scaling and the non-MoE floor

### Your original idea — faithful reconstruction

Do not assume expert traffic is the only bottleneck. Measure how attention and other resident work scale with context, because even perfect expert elimination cannot beat the non-MoE floor.

### Refined technical formulation

Fit attention time versus position/context:

```text
T_attention(n) ≈ intercept + slope * n
```

or a more appropriate measured model if nonlinear effects appear.

Measure:

- attention ms by position;
- KV-cache effects;
- context length;
- resident compute floor;
- synchronization floor.

### Why it may matter

The route-stable resident baseline around 222.5 ms/token is already far above the 33.3 ms target for 30 tok/s. Long-span block verification or more radical compute reuse is required even if storage traffic vanishes.

### Minimal experiment

Run the same binary and configuration at controlled context lengths with no concurrent builds or E1 jobs. Fit slope and intercept, then repeat.

### Success criterion

A credible non-MoE floor model that informs the maximum value of storage optimizations.

### Failure criterion

No timing result is valid if compilation, another benchmark, or background model loading contaminates the run.

---

## 37. Hardware/configuration lead as a separate experimental family

### Your original idea — faithful reconstruction

There may be large gains hidden in kernel selection, driver configuration, queue choice, power mode, CPU pinning, memory settings, or backend flags. Explore them, but treat external anecdotes as leads rather than facts.

### Refined technical formulation

Keep a controlled configuration matrix:

- Vulkan queue configuration;
- shader/kernel variants;
- CPU affinity and thread count;
- staging size;
- queue depth;
- memory profile;
- power mode;
- driver/runtime version.

Use same-command A/B/A/B or baseline/patch/revert/patch.

### Why it may matter

The cheapest gain is often a path-selection bug or generic default. A correct kernel/config choice can outperform months of architectural work.

### Minimal experiment

Sweep one variable at a time while recording hashes and exact environment.

### Success criterion

Repeated end-to-end gain with equivalent correctness and no hidden workload difference.

### Failure criterion

Do not preserve a dramatic number that cannot be reproduced under the same command and state.

---

## 38. Accepted-token roofline and exposed-byte target

### Your original idea — faithful reconstruction

Work backward from the target speed to the physical traffic budget. The runtime cannot negotiate with storage bandwidth.

### Refined technical formulation

Storage roofline:

```text
accepted_tokens_per_second
<= storage_bandwidth / exposed_bytes_per_accepted_token
```

At approximately 2.53 GiB/s cold storage bandwidth:

- 30 tok/s permits about 86 MiB exposed per accepted token;
- 40 tok/s permits about 65 MiB.

At approximately 7.2 GiB/s warm bandwidth:

- 30 tok/s permits about 246 MiB;
- 40 tok/s permits about 184 MiB.

These are rough arithmetic ceilings before compute and synchronization.

### Why it may matter

This immediately shows that ordinary cache improvements from 3.2 GiB to 1.8 GiB/token are valuable but nowhere near sufficient for 30 tok/s.

### Minimal experiment

For every mechanism, report exposed bytes per accepted token and position it against the roofline.

### Success criterion

The architecture demonstrates a credible route toward tens or low hundreds of MiB per accepted token through reuse, compact representation, long spans, or residual correction.

### Failure criterion

A proposed mechanism that cannot possibly cross the byte roofline should not be marketed as a 30 tok/s path.

---

# Part VI — How the ideas combine

## 39. The integrated HERMES-V4 architecture

The full architecture is not 39 unrelated projects. It is a set of interacting layers:

### Representation layer

- compact resident expert skeleton;
- canonical Q8 source of truth;
- residual tiles;
- ExpertPack;
- RDNA4-native layout.

### Prediction and speculation layer

- Route Scout;
- MARC;
- DSpark/future-token canvas;
- confidence and route-union estimation.

### Execution layer

- P1/P2/P4/P6 widening;
- expert-major batching;
- persistent expert atlas;
- asynchronous lanes;
- multi-source fabric.

### Control layer

- telemetry satellite;
- dynamic GPS;
- risk lanes;
- ramp metering;
- tailwind/fatigue state;
- capital allocation;
- purification and protection budgets.

### Safety and learning layer

- Id/Superego/Ego arbitration;
- viability governor;
- behaviorist learning;
- mechanostat remodeling;
- reasoning-distilled controller;
- empirical certificates.

### Unified operating principle

> Select the cheapest recoverable path to a verified accepted token block, while preserving enough reserve to correct mistakes and remain inside the system’s viable operating region.

---

# Part VII — Recommended execution order

This is the value-first order developed for Pi Agent.

## Phase 0 — trustworthy foundation

1. Correct transport, staging, slot mapping, and verifier logic.
2. Separate timing validity from correctness validity.
3. Obtain a clean full-Q8 compact control.
4. Freeze hashes, commands, and certificates.

## Phase 1 — measurement and quality

5. Telemetry and empirical certificates.
6. Ability-retention / Inference-IQ battery.
7. E2 non-MoE floor measurement.

## Phase 2 — highest-value architecture hypothesis

8. Compact skeleton E1.
9. Progressive P1/P2/P4/P6 widening.
10. Mixed-precision and layer-sensitivity analysis.

## Phase 3 — physical movement and reuse

11. Persistent expert atlas.
12. ExpertPack microprototype.
13. Expert-major teacher-forced batching.

## Phase 4 — speculation and prediction

14. DSpark/MTP restoration and true block verification.
15. Route Scout in shadow mode.
16. MARC thresholds and confidence calibration.

## Phase 5 — hardware optimization

17. RDNA4-native kernels for the winning shapes.
18. Startup autotuner.
19. Controlled kernel/config sweeps.

## Phase 6 — controller integration

20. Capital allocation and purification/protection policies.
21. Behaviorist bounded adaptation.
22. Fatigue, GPS, risk lanes, and ramp metering.
23. Id/Superego/Ego modular arbitration.
24. Neuro-inspired control modules where ablations justify them.
25. Maturana viability governor.

## Phase 7 — slow learning and scale-out

26. Inference Mechanostat.
27. Reasoning-distilled controller.
28. LAN/multi-source expert fabric.

---

# Part VIII — Global research rules

1. **Do not multiply speculative speedups.** Integrate them in one accounting model.
2. **Only one load-bearing experiment at a time.** Do not contaminate timing with builds or parallel benchmarks.
3. **Preserve failed runs.** Negative certificates are evidence.
4. **Use small falsification tests first.** One layer and one token before a 43-layer generation.
5. **Require ablation.** Never publish one giant integrated number without attribution.
6. **Separate exact and hybrid modes.** Every output prints its correctness class.
7. **Measure accepted-token economics.** Raw route accuracy, cache hit rate, or kernel throughput are intermediate metrics.
8. **Treat analogies as design lenses.** Traffic, bodybuilding, purification, sunscreen, Freud, Maturana, and neuroscience must map to measurable mechanisms.
9. **Never overclaim novelty.** Classify each mechanism as direct prior art, close analogy, integration opportunity, plausible new contribution, or unsupported.
10. **The first goal is not 30 tok/s.** The first goal is a trustworthy apparatus capable of accepting or killing the hypotheses.

---

# Part IX — What would count as a major result

## Strong success

- large exposed-byte reduction;
- target-equivalent behavior where claimed;
- sustained real-chat throughput increase;
- stable p95/p99 latency;
- lower joules per accepted token;
- clear ablations;
- low controller overhead;
- reproducible certificates.

## Partial but valuable success

- a realistic simulator;
- a useful low-bit layer subset;
- ExpertPack I/O gain;
- expert-major kernel gain;
- Route Scout hides real I/O;
- a controller improves energy or tails even below 30 tok/s;
- a sound physical roofline explaining the limit.

## Valuable falsification

- expert unions grow too quickly;
- DSpark acceptance is too low;
- compact paths lose too much ability;
- verification dominates;
- controller overhead is too high;
- exactness requires nearly complete Q8 work;
- the hardware floor prevents the target.

A rigorous negative result is not failure. It tells the field which attractive idea does not survive contact with the actual machine.

---

# Appendix A — Compact index of all idea families

1. Consumer-hardware frontier-model challenge
2. Full Q8 source of truth and correctness classes
3. Resident compact expert skeleton
4. Progressive Energy-Aware MoE Skipper P1/P2/P4/P6
5. Canonical Q8 residual tiles
6. Route Scout branch predictor
7. MARC margin-aware routing calibration
8. DSpark future-token canvas
9. Expert-major multi-position batching
10. Persistent expert atlas and stable slots
11. ExpertPack / physical aerodynamics
12. RDNA4-native execution
13. Asynchronous three-lane pipeline
14. Global telemetry satellite
15. Dynamic MoE GPS
16. Safe/Balanced/Autobahn risk lanes
17. Motorway merges and ramp metering
18. Broadband / multi-source expert fabric
19. Tailwind/headwind/sweet spot
20. Fatigue/recovery/RIR
21. Compound versus isolation
22. Macro resource allocation / bodybuilding
23. Water-purification cascade
24. Sunscreen protection budget
25. Behaviorism / consequence-driven learning
26. Freud Id/Superego/Ego arbiter
27. Investment/capital/salvage value
28. Inference-IQ / ability per joule
29. Maturana viability governor
30. Wolff’s law / Inference Mechanostat
31. Neuro-inspired modular control plane
32. Startup autotuner
33. Empirical certificates / Explorer-Verifier chain
34. Reasoning-distilled inference controller
35. E1 compact-skeleton fidelity program
36. E2 attention/non-MoE floor program
37. Kernel/configuration lead program
38. Accepted-token roofline and exposed-byte budget
39. Integrated HERMES-V4 architecture

---

# Appendix B — Provenance and limits

This consolidation is based primarily on:

- the HERMES-V4 Pi Agent master prompt;
- the canonical 21-idea execution-order prompt;
- subsequent conversation refinements covering Wolff’s law, the neuro-inspired control plane, empirical certificates, kernel/config leads, and the reasoning-distilled controller;
- the current DeepSeek-V4 hardware-aware research context and debugging history.

The “Your original idea” sections are faithful reconstructions of intent. They are not guaranteed word-for-word quotations because not every original conversational message survives as a directly addressable transcript fragment in the current artifact context. The technical refinements are explicitly separated so future readers can distinguish the originating intuition from the systems-engineering interpretation.
