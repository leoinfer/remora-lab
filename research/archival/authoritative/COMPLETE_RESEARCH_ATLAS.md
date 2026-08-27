# the project Complete Research Atlas

**Purpose:** authoritative inventory of the project's HERMES-V4 mechanisms plus the broader Qwen, MARC, compression, kernel, memory, and autonomous-experiment programs. This is an inventory and dependency record, not a claim that every item is implemented.

**Authoritative HERMES source read in full:** `archival/HERMES_V4_COMPLETE_IDEA_ATLAS.md` (2,491 lines, SHA-256 `7b092e2d9d850950d9d29b3424a26f71239107ca81d48faa2372aa85dc7369be`). The `.txt` copy has the same hash. Durable inputs also include the Qwen compact audits/port map, MARC-Symbiote schemas and policy, DeepSeek/HERMES feasibility and streaming records, DSpark memory-lookahead records, MoE-Skipper claims/failures, Laguna/REAP reports, and R4I8 checkpoint notes.

**Scope correction:** the autonomous lane below is the local research-experiment harness only.

## Exact inventory counts

The count is explicit so later queue revisions cannot silently merge names:

- **39 HERMES-V4 mechanisms:** `H01`–`H39`, exactly the 39 numbered mechanisms in the complete atlas.
- **26 broader/new named families:** `N01`–`N26`. Names that the project supplied separately remain separate: MARC-X, MARC-OS, and MARC-Synapse are three layers; REAP, Laguna, pruning, and AutoSurgeon are four tracks; coverage-first, precision-first, and heterogeneous are three practical variants; R4I8, R5I8, and R6I8 are three format tracks; cooperative-matrix and GEMV/GEMM are two kernel tracks.
- **Total preserved idea families: 65** under the strict no-merge count used here.
- **Top-level bundles in the user's broader bullet list: 56** (39 HERMES + 17 requested broader bullets); the 26-family `N` layer deliberately decomposes those bundles so separately named MARC layers, pruning tracks, three variants, formats, and kernel families are not silently merged.
- **Decomposed experiments:** 96, enumerated in `COMPLETE_EXPERIMENT_QUEUE.md`.

A family can have several experiments. A shared experiment names every family it covers rather than erasing those names.

## Shared terminology and correctness labels

- **Exact / target-equivalent:** canonical source-of-truth behavior is preserved by a sound acceptance or equivalence procedure.
- **Numerically close:** logits or tensors are close, but target-equivalent output is not established.
- **Hybrid / verified:** a cheaper path proposes work and a stronger authority verifies or repairs it.
- **Approximate:** final output can differ from the authority; quality must be measured separately.
- **Trace-only:** observations are recorded without changing execution.
- **Simulator-only:** no real model execution claim.
- **Unvalidated:** design or code exists, but the required gate has not run.

A speed number never upgrades a correctness class. Logical bytes, cached bytes, uploaded bytes, and exposed physical bytes are separate metrics.

---

# Part I — HERMES-V4: all 39 mechanisms

## H01 — consumer-hardware frontier-model challenge

- **Original terminology / analogy:** Run a frontier-scale sparse MoE, especially DeepSeek-V4-Flash, on an ordinary RX 9060 XT/32-GB/NVMe machine; the goal is a new inference architecture rather than merely loading a model. Approximately 30 accepted tokens/s in real chat is a north-star target, not a promise.
- **Technical interpretation:** An online constrained scheduler over VRAM, RAM, NVMe, CPU copies, Vulkan queues, routing, speculation, verification, energy, and recovery. Optimize verified accepted-token blocks and exposed bytes, not a local kernel KPI.
- **Applicability:** DeepSeek first; Qwen is the cheaper architecture host; both share sparse-MoE transport lessons; future MARC-native models can expose the same control problem.
- **Implementation status:** Active research thesis; not achieved. Qwen Q2 is the immediate load-bearing experiment.
- **Existing evidence:** DeepSeek route-stable resident and streaming baselines, storage/PCIe rooflines, and Qwen NGL=20 control establish that movement and scheduling dominate arithmetic.
- **Smallest falsifier:** Produce a clean stage ledger for one token and show that the proposed action changes its binding stage; a 30-t/s claim without accepted-token and byte accounting is invalid.
- **Dependencies:** H02, H10, H13, H14, H15, H33, H38, and a clean model-specific control.
- **Speed/quality mechanism and correctness:** Reduce exposed movement and amortize work while retaining an explicit exact, hybrid, or approximate label. The challenge itself is a measurement objective, not a correctness shortcut.
- **Compatibility:** Complementary with every mechanism; incompatible only with claims that ignore the non-MoE floor or storage physics.

## H02 — source-of-truth correctness modes

- **Original terminology / analogy:** Full Q8 is the canonical source of truth. Lower-bit paths, skipped experts, and speculation may be aggressive but must not quietly redefine correctness.
- **Technical interpretation:** Every result carries exact/target-equivalent, numerically close, hybrid/verified, approximate, simulator-only, or unvalidated status. Greedy identity and stochastic rejection sampling require their own gates.
- **Applicability:** Both Qwen and DeepSeek; future MARC-native systems need the same authority boundary.
- **Implementation status:** Policy established and used in HERMES certificates; Qwen Q2 will use exact Q6_K_XL source bytes, not re-quantized authority.
- **Existing evidence:** HERMES control certificates, transport failures preserved as invalid, and MoE-Skipper records that quarantine very fast but garbage output.
- **Smallest falsifier:** Run canonical, transport-only, and degraded paths on identical prompts and verify that the reporting system distinguishes all three.
- **Dependencies:** H33 certificates, source hashes, deterministic controls, and model-specific parity harnesses.
- **Speed/quality mechanism and correctness:** Prevents quality laundering; hybrid verification can save work without accepting an unchecked token.
- **Compatibility:** Complementary to exact transport and hybrid H08/H35; mutually exclusive with calling unchecked H03/H04/H23 output exact.

## H03 — compact expert skeleton

- **Original terminology / analogy:** A tiny low-bit skeleton of the entire expert system stays resident as a cheap whole-model first approximation, while full Q8 remains available for correction or verification.
- **Technical interpretation:** A Q2/Q3/Q4/FP8 or other compact representation supplies proposal, confidence, route assistance, progressive refinement, or residual requests. It is not assumed to replace Q8.
- **Applicability:** DeepSeek is the original target; Qwen is a high-value transfer host; both; future MARC-native models can train a native skeleton.
- **Implementation status:** DeepSeek E1 fidelity matrix measured and E1A staging anomaly being closed; Qwen skeleton is not the active Q2 path and must not be confused with exact Q6 compact transport.
- **Existing evidence:** DeepSeek Q4_K preserves greedy identity through shallow depth and Q3_K reaches 24 layers in the recorded matrix; full-depth fidelity degrades and the invalid b4_k43 row was a staging artifact. Qwen geometry supports a compact slot arena, not yet a low-bit model skeleton.
- **Smallest falsifier:** One layer, one position, bits 2/3/4, compare logits, route IDs, token identity, and bytes against the canonical control.
- **Dependencies:** H02, H33, H35, H38; residual tiles H05 and progressive H04 are optional extensions.
- **Speed/quality mechanism and correctness:** Resident approximation removes cold traffic; exact mode requires H08-style verification or a sound certificate. Otherwise it is approximate.
- **Compatibility:** Complementary with H04–H10; a Q2 exact-slot transport is complementary but not evidence for low-bit skeleton fidelity.

## H04 — P1/P2/P4/P6 progressive MoE

- **Original terminology / analogy:** Start with fewer selected experts and widen only when confidence or quality requires it; reuse already-computed work.
- **Technical interpretation:** Nested P1, P2, P4, and canonical P6 paths with incremental sums and a cost/risk controller. A resident P2 can beat a cold P1, so bit count alone cannot choose the path.
- **Applicability:** Both existing MoEs; future MARC-native routers can expose nested paths directly.
- **Implementation status:** Design; no validated runtime widening path. Qwen top-8 makes the direct labels model-specific, so Qwen experiments must define P1/P2/P4/P8 or retain the HERMES terminology as an analogy.
- **Existing evidence:** HERMES theory and Q6 MoE-Skipper show that approximation error compounds across layers; no P1-to-P6 accepted-token result exists.
- **Smallest falsifier:** On recorded routes, compute incremental path outputs and measure whether most positions require the maximum path and whether prior work is actually reused.
- **Dependencies:** H02, H03, H07, H14, H23/H24, and a route-aware verifier.
- **Speed/quality mechanism and correctness:** Saves expert FLOPs and bytes when early paths are sufficient; exact output needs verification or a conservative fallback.
- **Compatibility:** Complementary with H03/H05/H07; mutually exclusive with any implementation that recomputes earlier experts on every widening step.

## H05 — Q8 residual tiles

- **Original terminology / analogy:** Use the compact base, then fetch only the exact correction information needed instead of a complete expert.
- **Technical interpretation:** Store `W_Q8 = W_skeleton + Delta_W_exact` as ranked tiles, channel corrections, or projection corrections, with full-authority fallback when residuals are dense.
- **Applicability:** Both if a nested representation is built; future MARC-native models could train residual-friendly layers.
- **Implementation status:** Design/unvalidated. Ordinary Q3/Q4/Q6 GGUF files are not a nested residual ladder.
- **Existing evidence:** HERMES residual-byte budget and weight-structure audit show expert weights are largely unstructured at the existing grid; this makes the experiment important but not guaranteed.
- **Smallest falsifier:** One layer/expert: quantize a compact base, rank residual tiles by output contribution, and plot output recovery versus exact bytes fetched.
- **Dependencies:** H03, H11, H12, H28, H38, and a tile index.
- **Speed/quality mechanism and correctness:** Selective correction can move tens of MiB rather than a full expert; exactness holds only for fetched exact residuals plus a sound fallback.
- **Compatibility:** Complementary with H03/H04/H08; a representation change is mutually exclusive with the fixed Q6/Q8 control for the same A/B run.

## H06 — Route Scout

- **Original terminology / analogy:** A branch predictor for future expert choices; use likely routes to prefetch and prepare slots before the true router finishes.
- **Technical interpretation:** Predict future IDs, unions, margins, source tier, reuse, and repair value; judge by useful hidden traffic and stall reduction, not classification accuracy.
- **Applicability:** Both Qwen and DeepSeek; future MARC-native systems can expose route features natively.
- **Implementation status:** DeepSeek has a naive transition-table prefetch substrate and trace data; Qwen route history is available; margin-aware and cache-aware Route Scout are not validated.
- **Existing evidence:** DeepSeek strict-holdout transition predictor beats t-1 at short lookahead but miss-conditioned F1 falls; Qwen 24-step trace shows large route unions and churn.
- **Smallest falsifier:** Shadow-predict one next position, measure top-k/union recall, useful bytes, wasted bytes, on-time arrival, and exposed latency against last-route baseline.
- **Dependencies:** H07, H10, H13, H14, H18, H38.
- **Speed/quality mechanism and correctness:** Hides I/O and protects slots; it may never gate correctness and must be cancellable.
- **Compatibility:** Complementary with H08/H10/H15; heuristic prefetch that ignores H14 telemetry or becomes mandatory is incompatible with the research policy.

## H07 — margin-aware routing calibration

- **Original terminology / analogy:** Use router confidence and margin, not only expert IDs, to decide how aggressively to approximate, prefetch, skip, or widen.
- **Technical interpretation:** Calibrate top-1/top-2 margins, entropy, predictor confidence, residency, load cost, task sensitivity, and repair risk into path and authority decisions.
- **Applicability:** Both; Qwen derives margins from top-8 probabilities, DeepSeek uses top-6 traces; future MARC-native routers can train confidence heads.
- **Implementation status:** Proposed; margin fields exist in DeepSeek/Qwen trace designs but no fixed gate has passed.
- **Existing evidence:** Qwen trace tooling emits margins/entropy; HERMES policy explicitly requires calibration rather than trusting raw margins.
- **Smallest falsifier:** Bucket positions by margin and measure P1/P2 disagreement, route divergence, repair probability, and bytes required; compare calibrated thresholds to static width.
- **Dependencies:** H02, H06, H14, H28, H33.
- **Speed/quality mechanism and correctness:** Spends authority on near ties and saves it on stable decisions; the margin is never a proof by itself.
- **Compatibility:** Complementary with H04/H05/H06; conflicts with fixed-width-only policies only as an optional adaptive mode.

## H08 — DSpark/MTP future-token canvas

- **Original terminology / analogy:** Treat proposed future tokens as a canvas: freeze stable positions, refine uncertain ones, group route demand, and verify the longest causal prefix.
- **Technical interpretation:** Propose 2–16 positions, predict route unions, expert-major execute, verify strict rejection/acceptance, and salvage useful rejected work. Future route information is not exact before target execution.
- **Applicability:** DeepSeek is the original DSpark target; Qwen has separate MTP/DSpark artifacts and is the first practical host; both only after model-specific integration.
- **Implementation status:** DeepSeek GGUF currently lacks official `mtp.*` tensors; Qwen-first policy defers MTP; trace-only canvas simulations are available. No integrated HERMES block verifier.
- **Existing evidence:** DSpark audit found official module size/tensor gaps; static union and wrong-token/right-memory analyses show potential but also high union cost. MTP on other 16-GB configurations can be slower when the draft is not resident.
- **Smallest falsifier:** Teacher-forced K=2/4/8/16 route-union simulation with real draft/target traces; then a K=2 exact block with accepted-prefix and draft-cost accounting.
- **Dependencies:** H02, H03, H06, H09, H10, H13, H38; Qwen Q2 is not a prerequisite for the trace-only step but is a high-value residency prerequisite.
- **Speed/quality mechanism and correctness:** Amortizes loads, dispatches, and verification across accepted tokens; exact mode is hybrid/verified, approximate mode is explicitly approximate.
- **Compatibility:** Complementary with H03/H09/H10; sequential one-token verification, missing MTP tensors, and unchecked later positions are blockers.

## H09 — expert-major multi-position batching

- **Original terminology / analogy:** Load an expert once and process every future position that needs it together instead of token-by-token.
- **Technical interpretation:** Form the per-layer expert union, group positions by expert, run grouped MMID/GEMM, scatter outputs, and preserve causal/KV state. Teacher-forced grouping is not automatically free-running verification.
- **Applicability:** Both; Qwen Q2 currently rejects multi-position input; future MARC-native models can train for the shape.
- **Implementation status:** Offline union curves are measured; true graph execution is not validated.
- **Existing evidence:** DeepSeek union curves reduce expert executions per token as K grows but still expose large bytes; Qwen route union grows from 48–112 distinct experts/layer across 24 steps.
- **Smallest falsifier:** Two layers, K=2/4 teacher-forced rows, compare sequential and expert-major outputs bit/numerically and measure actual physical loads and kernel shapes.
- **Dependencies:** H02, H10, H13, H38, KV snapshots, and Qwen Q2 single-position certificate.
- **Speed/quality mechanism and correctness:** Amortizes load and launch cost; exactness depends on the causal schedule and matching accumulation order.
- **Compatibility:** Complementary with H08/H11/H12; cannot be combined with a claim of parallel causal token generation without a state proof.

## H10 — persistent expert atlas and stable slots

- **Original terminology / analogy:** Maintain an atlas of where experts live rather than treating each payload as anonymous; stable slots survive tokens, rejection, and route rank churn.
- **Technical interpretation:** Per-layer original-ID↔slot maps, ages, cache ownership, in-flight state, reuse/salvage value, and eviction policy. Map IDs by identity, not router rank.
- **Applicability:** Both. DeepSeek has validated associative slot-map transport; Qwen Q2 implements a dynamic 10-slot design in an isolated branch.
- **Implementation status:** DeepSeek transport/associative mapping validated; Qwen manager exists but is not graph-integrated or tested.
- **Existing evidence:** HERMES 214/214 upload verification and corrected rank-vs-slot/staging fixes; Qwen simulation shows LRU 8/9/10 hit rates of 24.596/29.388/32.721% and an oracle 6.755-GiB one-time set.
- **Smallest falsifier:** One-layer one-slot thrash oracle with repeated ID permutations; require byte-identical output and zero stale-slot reads.
- **Dependencies:** H02, H13, H14, H33, source authority.
- **Speed/quality mechanism and correctness:** Avoids re-upload on rank churn and supports salvage; stale or repurposed data is a hard correctness failure.
- **Compatibility:** Complementary with H06/H08/H09/H11; positional-slot-only mappings are superseded by associative maps.

## H11 — ExpertPack

- **Original terminology / analogy:** Make expert data physically aerodynamic: contiguous, indexed, aligned, reversible, and arranged for the runtime's actual access pattern.
- **Technical interpretation:** A lossless `[gate|up|down]` indexed pack with offsets, checksums, quantization, staging/GPU layout, coactivation order, and optional pread/preadv/io_uring support.
- **Applicability:** Both; Qwen's current authority is a single GGUF and can use tensor-span/pack indexes; DeepSeek has an expert index and a repack WIP.
- **Implementation status:** DeepSeek repack functions exist in a stash/WIP, not a clean completed result; Qwen Q2 starts from CPU/mmap tensor slices and defers packing.
- **Existing evidence:** DeepSeek index lookup is validated; 882 GGUF extents reduce fragmentation concern but Btrfs compressed reads have amplification; no end-to-end ExpertPack gain yet.
- **Smallest falsifier:** Pack two layers, replay identical routes against GGUF and pack, require reversible hashes and compare cold/warm read, CPU, staging, upload, and logits.
- **Dependencies:** H02, H10, H18, H33, storage measurements.
- **Speed/quality mechanism and correctness:** Reduces read amplification, syscalls, copies, and descriptor churn without changing model values.
- **Compatibility:** Complementary with all exact transport; independent packs are mutually exclusive with the unmodified-file control only during an A/B, not architecturally.

## H12 — RDNA4-native execution

- **Original terminology / analogy:** Make the winning path native to the actual AMD GPU instead of relying on generic compromises.
- **Technical interpretation:** Wave32/64 choice, fused dequant+matmul, persistent descriptors, expert-major MMID, fewer dispatches/fences, and shape-specific kernels for compact and authority forms.
- **Applicability:** Qwen and DeepSeek on RDNA4; future MARC-native models can target the same hardware interface.
- **Implementation status:** Hardware capability is audited; custom HERMES/Qwen kernels are not complete. Native Qwen Q6_K/Q8_0 `MUL_MAT_ID` registration exists.
- **Existing evidence:** RDNA4 exposes cooperative matrix and integer dot capabilities; DeepSeek IQ3 MMID is bandwidth/instruction limited and does not currently use coopmat; R4I8 native dispatch is structurally working but quality is low.
- **Smallest falsifier:** Benchmark the exact selected MMID shape with generic versus custom kernel under identical bytes and end-to-end decode; require repeated A/B/A/B gain.
- **Dependencies:** H09, H10, H38, H37, Qwen Q2 certificate.
- **Speed/quality mechanism and correctness:** Raises useful VRAM bandwidth and reduces launch/dequant overhead; tensor-level parity is mandatory.
- **Compatibility:** Complementary with transport and batching; a cooperative-matrix rewrite is not assumed beneficial merely because hardware advertises it.

## H13 — asynchronous three-lane pipeline

- **Original terminology / analogy:** Draft/Route Scout, data movement, and verification/repair should run as lanes, not a serial loop.
- **Technical interpretation:** Overlap draft/route generation, read/stage/upload, and canonical verification/commit with fences, ownership, causal ordering, and recovery reserve.
- **Applicability:** Both; DeepSeek has deferred copy/staging substrate, Qwen has imported hooks but no integrated graph.
- **Implementation status:** Partial transport substrate; no complete end-to-end three-lane pipeline.
- **Existing evidence:** HERMES deferred H2D and staging fixes are validated; separate graphics/compute queue overlap hung on the audited driver, so cross-queue overlap is risky.
- **Smallest falsifier:** Timestamp one two-position block and show actual overlap and reduced critical path without staging reuse or p95 regression.
- **Dependencies:** H10, H14, H17, H18, H33; Q2 fence/epoch rules.
- **Speed/quality mechanism and correctness:** Hides unavoidable latency; staging slices remain owned until a fence/epoch boundary.
- **Compatibility:** Complementary with H06/H08/H09; `RADV_EXPERIMENTAL=transfer_queue` and `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` remain disabled in Qwen control.

## H14 — global telemetry satellite

- **Original terminology / analogy:** A satellite view sees traffic, queues, accidents, roadworks, energy, and future demand instead of making blind local choices.
- **Technical interpretation:** Record routes/margins, slots, cache tiers, staging high-water, pending transfers, bytes/time by tier, queue state, acceptance, thermal/power, KV, policy, and correctness.
- **Applicability:** Both; DeepSeek already has partial counters and Qwen tracer can add process/route signals; future MARC-native models need a typed state bus.
- **Implementation status:** Partial and mostly trace-only. Qwen hardware fields are simulated until Q2 exposes live residency.
- **Existing evidence:** HERMES `DSV4_STATS`/trace/certificates and Qwen `qwen_trace` route/entropy output; Qwen Q2 expected markers include `QWEN_Q2_STATS` and fail counters.
- **Smallest falsifier:** Turn on summary telemetry and predict/explain one stall while measuring negligible overhead against no-trace control.
- **Dependencies:** H02, H10, H13, Qwen tracer/Q2 manager.
- **Speed/quality mechanism and correctness:** Enables attribution and safe backpressure; synchronous verbose logging is prohibited.
- **Compatibility:** Foundational and complementary; telemetry must not silently alter graph scheduling.

## H15 — dynamic MoE GPS

- **Original terminology / analogy:** Live GPS replans the cheapest route as traffic and conditions change.
- **Technical interpretation:** Receding-horizon controller over route state, residency, queues, path width, energy, thermal state, and correctness risk; execute one action, ingest telemetry, replan.
- **Applicability:** Both; future MARC-OS is a general implementation.
- **Implementation status:** Design/simulator target; no live authority.
- **Existing evidence:** HERMES feasibility model and Qwen residency simulator provide the state/cost ingredients, but not a live planner.
- **Smallest falsifier:** Replay the same traces through static, greedy, receding-horizon, and oracle policies; show planner savings exceed planner overhead.
- **Dependencies:** H14, H10, H04/H06/H08, H20, H29.
- **Speed/quality mechanism and correctness:** Accounts for future reuse and repair rather than one locally cheap action; hard constraints remain authoritative.
- **Compatibility:** Complementary with H16/H17/H27; conflicts with an unbounded learned controller.

## H16 — Safe/Balanced/Autobahn lanes

- **Original terminology / analogy:** Safe driving is conservative, Balanced takes measured risks, Autobahn exploits favorable conditions while retaining an emergency lane.
- **Technical interpretation:** Profiles of canvas length, path width, residency preference, prefetch depth, recovery reserve, and risk budget.
- **Applicability:** Both; future MARC-OS can expose user/system policy.
- **Implementation status:** Named policy design; no controlled lane comparison.
- **Existing evidence:** HERMES feasibility already distinguishes cold/warm and route-stable regimes; no lane selector has been validated.
- **Smallest falsifier:** Replay identical workloads under three fixed profiles and measure accepted throughput, tails, waste, and recovery.
- **Dependencies:** H14/H15/H20/H29/H33.
- **Speed/quality mechanism and correctness:** Trades burst throughput for sustained safe operation; Autobahn never bypasses H02.
- **Compatibility:** Complementary with H17/H19; lane policies are mutually exclusive choices per run, not merged speed multipliers.

## H17 — motorway merges and ramp metering

- **Original terminology / analogy:** Admission control prevents speculative and recovery traffic from flooding a motorway.
- **Technical interpretation:** Prioritize canonical recovery/verification over reusable prefetch and low-confidence speculation; cap new work by downstream capacity and use fairness/hysteresis.
- **Applicability:** Both.
- **Implementation status:** Design; no runtime admission controller.
- **Existing evidence:** DSpark static analysis identifies cache pollution and saturated queues; HERMES pipeline has no live ramp controller.
- **Smallest falsifier:** Inject increasing speculative volume and compare unlimited, fixed-cap, and telemetry-metered admission under p95 and accepted-token metrics.
- **Dependencies:** H13/H14/H15/H16/H29.
- **Speed/quality mechanism and correctness:** Prevents queue explosion and protects recovery bandwidth.
- **Compatibility:** Complementary with all speculative paths; conflicts with unlimited concurrency.

## H18 — broadband/multi-source expert fabric

- **Original terminology / analogy:** Expert data is broadband traffic from VRAM, RAM, NVMe, LAN RAM, or another GPU; choose the source that arrives soonest and economically.
- **Technical interpretation:** Measure source arrival time, queue delay, upload/decode cost, energy, opportunity cost, and failure repair risk.
- **Applicability:** Both; local tiers first, future MARC-native multi-node systems later.
- **Implementation status:** Local VRAM/RAM/NVMe exists; LAN/remote fabric is unvalidated and not token-critical.
- **Existing evidence:** DeepSeek and Qwen budgets quantify local tiers; no LAN endpoint has been tested.
- **Smallest falsifier:** Choose between resident, RAM, and NVMe using live arrival estimates and demonstrate lower critical-path cost; only then test a local LAN depot.
- **Dependencies:** H10/H14/H15, H11, H38.
- **Speed/quality mechanism and correctness:** Reduces wait or energy by source selection; remote failures must fall back without changing authority.
- **Compatibility:** Complementary with H13/H17; internet-in-the-token-loop is out of scope.

## H19 — tailwind/headwind/sweet spot

- **Original terminology / analogy:** Residence, reuse, accepted speculation, warm shaders, and empty queues create tailwind; churn, misses, and thermal pressure create headwind.
- **Technical interpretation:** A measured score controls canvas, path width, prefetch, and concurrency; find marginal joules per accepted token/s rather than inventing a universal law.
- **Applicability:** Both.
- **Implementation status:** Design; regime measurements exist but no controller.
- **Existing evidence:** DeepSeek warm/cold and Qwen control differences; Gemma/Qwen records are not interchangeable model evidence.
- **Smallest falsifier:** Sweep concurrency/canvas after thermal stabilization; plot accepted tok/s, joules/token, p95, and queue debt.
- **Dependencies:** H14, H20, H28, H29.
- **Speed/quality mechanism and correctness:** Selects sustainable operating points and prevents peak-only claims.
- **Compatibility:** Complementary with H16/H17/H20; no fixed universal sweet spot is assumed.

## H20 — fatigue/recovery/RIR

- **Original terminology / analogy:** Like training, queues, thermal headroom, cache churn, and rollback capacity accumulate fatigue; retain reps in reserve.
- **Technical interpretation:** Exponentially decaying resource-specific fatigue with reserve thresholds and active recovery policies.
- **Applicability:** Both; future MARC-OS control plane.
- **Implementation status:** Design only.
- **Existing evidence:** Long-run HERMES/Q6 records show variance, thermal/queue concerns, and need for sustained rather than burst metrics; no fatigue ablation.
- **Smallest falsifier:** Long-generation soak with fatigue on/off under induced queue/thermal/cache pressure; compare drift, tails, and recovery.
- **Dependencies:** H14, H17, H19, H29.
- **Speed/quality mechanism and correctness:** Prevents self-induced collapse and preserves verification reserve.
- **Compatibility:** Complementary with H16/H17; artificial rests without measured load are not a mechanism.

## H21 — compound versus isolation

- **Original terminology / analogy:** Compound useful work through batching/reuse, but isolate local errors and repair only the failed position/layer.
- **Technical interpretation:** Choose block size and repair scope using accepted-token value, union growth, scatter cost, and rollback blast radius.
- **Applicability:** Both.
- **Implementation status:** Design; union curves and cascade failure evidence make the trade real.
- **Existing evidence:** Expert-major union curves show reuse; MoE-Skipper multi-layer cascades show error compounding; no unified policy.
- **Smallest falsifier:** Replay identical routes with global block repair versus local layer/position repair and measure accepted economics.
- **Dependencies:** H08/H09/H13/H24/H33.
- **Speed/quality mechanism and correctness:** Balances amortization against causal error blast radius.
- **Compatibility:** Complementary with H09; giant blocks and global repair can be mutually exclusive with low-latency exact mode.

## H22 — macro resource allocation/bodybuilding

- **Original terminology / analogy:** Carbs are productive speculative work, protein is verification/correction, and fat is reserve/headroom.
- **Technical interpretation:** Allocate `Budget = C + P + F`; reserve minimum verification and recovery capacity before maximizing speculative work.
- **Applicability:** Both and future MARC-OS.
- **Implementation status:** Design.
- **Existing evidence:** DSpark memory analysis shows speculation can saturate bandwidth; HERMES reserve rules are specified but not swept.
- **Smallest falsifier:** Sweep reserve percentages and measure accepted throughput, recovery success, and collapse frequency.
- **Dependencies:** H14/H17/H20/H29.
- **Speed/quality mechanism and correctness:** Avoids consuming all resources on work that cannot be certified.
- **Compatibility:** Complementary with H16/H23/H24; percentages are empirical, not literal physiology.

## H23 — water-purification cascade

- **Original terminology / analogy:** Cheap filters remove obvious uncertainty; expensive reverse-osmosis/UV-like stages are reserved for contaminated or risky positions.
- **Technical interpretation:** A calibrated uncertainty vector passes through coarse screening, P1/P2, P4, full P6/Q8, and final certification with an explicit stopping threshold.
- **Applicability:** Both; future MARC-native models can build stages into the architecture.
- **Implementation status:** Design.
- **Existing evidence:** Margin, entropy, path-width, and verification concepts exist independently; no cascade calibration.
- **Smallest falsifier:** Measure uncertainty reduction and false early-stop rate at each stage on one layer and held-out prompts.
- **Dependencies:** H04, H07, H14, H28, H33.
- **Speed/quality mechanism and correctness:** Stops easy positions early while escalating hard ones; unsafe early stopping demotes to approximate.
- **Compatibility:** Complementary with H04/H24; it is not the same as MoE-Skipper's learned layer substitution.

## H24 — sunscreen protection budget

- **Original terminology / analogy:** Speculative positions have different exposure and sensitivity; protect causal-prefix positions more strongly and reapply protection when conditions change.
- **Technical interpretation:** Allocate verification/widening budget by intensity × exposure time × semantic sensitivity, with residual risk constraints.
- **Applicability:** Both; future MARC-native semantic sensitivity can improve it.
- **Implementation status:** Design.
- **Existing evidence:** HERMES protection-mask design names numbers, names, code, and boundaries; no risk-weighted comparison.
- **Smallest falsifier:** Uniform versus risk-weighted verification at equal energy/quality budget, using causal-prefix failure rate.
- **Dependencies:** H07/H08/H14/H28/H33.
- **Speed/quality mechanism and correctness:** Protects early positions whose failure invalidates later acceptance.
- **Compatibility:** Complementary with H08/H23; cannot weaken the earliest prefix gate to save later work.

## H25 — behaviorism/consequence-driven learning

- **Original terminology / analogy:** Reward what produces verified accepted tokens cheaply; punish wasted bytes, repairs, quality loss, and future debt rather than trusting internal stories.
- **Technical interpretation:** Bounded policy learning over canvas, width, source, slots, and priority with a reward based on accepted-token economics.
- **Applicability:** Both; future MARC-native control.
- **Implementation status:** Design; no unrestricted online learning.
- **Existing evidence:** MARC proxy budget routing demonstrates a measurable control loop but uses keyword rules; HERMES policy requires shadow mode first.
- **Smallest falsifier:** Trace-replay contextual bandit versus frozen rules on held-out hardware states, with rollback on regressions.
- **Dependencies:** H14, H28, H29, H33, stable reward telemetry.
- **Speed/quality mechanism and correctness:** Learns state-dependent policies without allowing a private KPI to replace global accepted-token value.
- **Compatibility:** Complementary after fixed policies; mutually exclusive with early unrestricted RL.

## H26 — Id/Superego/Ego arbiter

- **Original terminology / analogy:** Id proposes aggressive speed; Superego enforces correctness, resource, and viability constraints; Ego selects the realistic feasible action.
- **Technical interpretation:** Proposal generator → hard constraint filter → risk-adjusted selector. It is a software architecture, not psychology.
- **Applicability:** Both; future MARC-OS.
- **Implementation status:** Design.
- **Existing evidence:** HERMES safety/viability rules provide the Superego boundary; no explicit three-module ablation.
- **Smallest falsifier:** Log candidate actions, blocked unsafe actions, selected actions, and ablate proposer/filter separately.
- **Dependencies:** H14/H15/H16/H17/H29.
- **Speed/quality mechanism and correctness:** Retains aggressive options without permitting infeasible or uncertified output.
- **Compatibility:** Complementary with H25/H31/H34; an opaque learner cannot replace the hard filter.

## H27 — investment/capital/salvage value

- **Original terminology / analogy:** Loads, prefetches, slots, and speculation are investments with immediate return, future dividends, risk, fees, opportunity cost, liquidity, and salvage value.
- **Technical interpretation:** Score actions by immediate value + future reuse + post-rejection salvage − load/opportunity/risk cost.
- **Applicability:** Both.
- **Implementation status:** Design; rejected-work accounting is specified but not measured in live Q2.
- **Existing evidence:** DSpark wrong-token/right-memory analysis and HERMES atlas design support the premise; no capital-policy A/B.
- **Smallest falsifier:** Replay latency-only, byte-only, hit-only, and salvage-aware policies on identical traces.
- **Dependencies:** H08/H10/H14/H21/H33.
- **Speed/quality mechanism and correctness:** Retains useful work after rejection without monopolizing scarce slots.
- **Compatibility:** Complementary with H06/H08/H17; a hit-rate-only cache policy is an explicit competing baseline.

## H28 — inference-IQ/ability per joule

- **Original terminology / analogy:** Measure useful retained ability and ability-per-joule, explicitly not human IQ.
- **Technical interpretation:** A repeatable battery across reasoning, math, code, planning, instruction, factuality, long context, and near ties; report retention and marginal ability per joule.
- **Applicability:** Both; future MARC-native model comparison.
- **Implementation status:** Proposed; HERMES Order-2 battery is queued, Qwen quality work has direct logit/task evidence but no unified score.
- **Existing evidence:** MoE-Skipper KL/top-token gates show why tensor error alone is insufficient; Gemma/Qwen quality files are model-specific and not a shared battery.
- **Smallest falsifier:** Verify that the battery distinguishes canonical Q8/Q6 from a deliberately degraded path, then compare P1/P2/P4 and pruned variants.
- **Dependencies:** H02, H33, fixed prompts/seeds, energy/timing measurement.
- **Speed/quality mechanism and correctness:** Gives the controller a capability objective rather than only RMS/KL; never upgrades approximate output to exact.
- **Compatibility:** Complementary with H03/H04/N10–N16; incompatible with reporting an IQ-like number as human psychometrics.

## H29 — Maturana viability governor

- **Original terminology / analogy:** Preserve the conditions that let the system keep operating and recovering; viability is a control principle, not a claim of life or consciousness.
- **Technical interpretation:** Define a viable region for temperature, queues, free recovery slots, error risk, latency debt, policy drift, and telemetry validity; shrink/cancel/revert when predicted state exits it.
- **Applicability:** Both; future MARC-OS.
- **Implementation status:** Design only.
- **Existing evidence:** Staging-capacity invalidation and driver queue hang show real non-viable states; no governor ablation.
- **Smallest falsifier:** Inject queue saturation, stale slots, thermal pressure, bad predictions, and cache churn with governor on/off.
- **Dependencies:** H14/H17/H20/H25/H26/H33.
- **Speed/quality mechanism and correctness:** Preserves recovery and correctness instead of maximizing short-term throughput until collapse.
- **Compatibility:** Complementary with every adaptive policy; it is the hard boundary that cannot be learned away.

## H30 — Wolff’s law/Inference Mechanostat

- **Original terminology / analogy:** Repeated load remodels structure over time; persistent slots, cache tiers, packs, and kernels should slowly adapt to verified demand.
- **Technical interpretation:** EMA structural value based on accepted-token value, reuse, bytes, energy, and monopolization penalty, with hysteresis, sample counts, holdouts, and rollback.
- **Applicability:** Both; future MARC-native systems can learn module structure.
- **Implementation status:** Design/simulator target.
- **Existing evidence:** Stable HERMES per-layer maps, LFU/LRU comparisons, and route-history statistics provide measurable state; no slow remodeling run.
- **Smallest falsifier:** Long trace replay comparing static, LRU, LFU, and slow value-aware remodeling on held-out workloads.
- **Dependencies:** H10/H14/H15/H27/H33.
- **Speed/quality mechanism and correctness:** Improves long-run placement while avoiding prompt-local overfit.
- **Compatibility:** Complementary with H10/H11; rapid per-token churn is not a mechanostat.

## H31 — neuro-inspired modular control plane

- **Original terminology / analogy:** Specialized modules at different timescales resemble functional brain regions: fast risk interrupt, slower planner, action selection, trace memory, timing correction, homeostasis, and emergency path.
- **Technical interpretation:** Explicit dispatcher, risk interrupt, planner, selector, episodic trace, timing predictor, homeostasis, canonical emergency path, and maintenance modules on a shared typed bus.
- **Applicability:** Both; future MARC-native control plane.
- **Implementation status:** Static design; no biological claim and no runtime module set.
- **Existing evidence:** Existing telemetry/certificate/cache components already imply the boundaries; modular ablation has not been done.
- **Smallest falsifier:** Implement only safety interrupt, rule selector, slow planner, and bus; ablate each and measure responsibility/latency.
- **Dependencies:** H14/H15/H26/H29/H33.
- **Speed/quality mechanism and correctness:** Keeps fast safety decisions out of a slow learned planner and makes authority auditable.
- **Compatibility:** Complementary with H25/H30/H34; module names that add no distinct behavior are rejected.

## H32 — startup autotuner

- **Original terminology / analogy:** Measure this exact machine/model/driver/storage at startup instead of trusting universal defaults.
- **Technical interpretation:** Quick/full profiles for reads, copies, uploads, kernels, queue depths, slots, synchronization, energy, and thermal behavior, keyed by hardware/model/shader/config hashes.
- **Applicability:** Both; future MARC-native runtime.
- **Implementation status:** Design; hardware probe data exists but no production selector.
- **Existing evidence:** RDNA4 audits show dramatic sensitivity to queue, clock, staging, GTT, and configuration choices; Qwen control flags are intentionally frozen.
- **Smallest falsifier:** Tune a bounded action space, restart, verify profile invalidation on identity change, and reproduce the selected gain.
- **Dependencies:** H14, H37, H38, stable microbenchmarks.
- **Speed/quality mechanism and correctness:** Avoids bad defaults and selects the measured operating point; does not train model quality.
- **Compatibility:** Complementary with H19/H37; never hides a workload change inside a benchmark.

## H33 — Explorer-Verifier certificates

- **Original terminology / analogy:** Explorers search broadly; verifiers are narrow, deterministic, hostile to false positives, and emit replayable certificates.
- **Technical interpretation:** Evidence levels L0–L6, hashes, commands, environment, prompts/seeds, raw logs, parser version, correctness/timing validity, fallback, and limitations.
- **Applicability:** Both and future MARC-native.
- **Implementation status:** DeepSeek transport certificates are strong; Qwen Q2 one-layer, eight-token, fence/thrash, and full-core placement certificates pass; speed and wider residency work remain gated.
- **Existing evidence:** HERMES upload verifier 214/214, control RMS/greedy checks, preserved invalid b4_k43 staging artifact, Qwen source/slot/ID/parity certificates, and required zero-error markers/counters.
- **Smallest falsifier:** Inject wrong IDs, slot order, missing copies, stale payload, parser errors, and invalid markers; require rejection.
- **Dependencies:** H02 and every load-bearing experiment.
- **Speed/quality mechanism and correctness:** Does not directly speed inference; it prevents invalid optimization from surviving.
- **Compatibility:** Mandatory and complementary; any result without a certificate remains unvalidated.

## H34 — reasoning-distilled inference controller

- **Original terminology / analogy:** An expensive offline controller reasons through hardware actions; a small fast policy is distilled from structured decisions.
- **Technical interpretation:** Oracle/state/action/reason-code traces for width, horizon, prefetch, slots, kernel, lanes, and recovery; train bounded student policies in shadow mode.
- **Applicability:** Both; future MARC-native controller.
- **Implementation status:** Design only.
- **Existing evidence:** H15/H25 simulators and oracle traces provide a possible teacher; no student policy trained.
- **Smallest falsifier:** Distill one bounded decision, such as horizon or P1/P2 choice, and compare student/rules/oracle with negligible overhead.
- **Dependencies:** H14/H15/H25/H29/H33 and stable labels.
- **Speed/quality mechanism and correctness:** Captures planning value without running an expensive planner per token; hard fallbacks remain.
- **Compatibility:** Complementary after fixed policies; training before telemetry/correctness stability is prohibited.

## H35 — E1 compact-skeleton program

- **Original terminology / analogy:** E1 determines whether a resident compact skeleton has a useful fidelity region across bits and routed-layer depth.
- **Technical interpretation:** Matrix of 2/3/4-bit representations and 4/12/24/43-layer depth with logits, route/token identity, ability, bytes, timing, and fallback metrics.
- **Applicability:** DeepSeek evidence first; Qwen later with its own topology; both.
- **Implementation status:** DeepSeek matrix completed to a Gate-3 GO with a b4_k43 staging anomaly classified/being corrected; Qwen E1 not started.
- **Existing evidence:** Q3/Q4 shallow identity and full-depth degradation; golden offline Q4 reconstruction better than Q3; Qwen trace/geometry only.
- **Smallest falsifier:** One ordinary Qwen layer and one token through the same matrix, with Q2 exact control alongside it.
- **Dependencies:** H02/H03/H33 and clean staging capacity.
- **Speed/quality mechanism and correctness:** Tests whether resident approximation can remove expert movement; results are numerical/approximate until verified.
- **Compatibility:** Complementary with H04/H05/H08; not interchangeable with exact Q2.

## H36 — E2 non-MoE floor program

- **Original terminology / analogy:** Measure the non-MoE floor so expert elimination is not mistaken for the whole target.
- **Technical interpretation:** Fit attention/KV/resident compute time versus context/position and separate graph, synchronization, and expert terms.
- **Applicability:** Both; Qwen's recurrent/delta paths make its topology-specific floor important.
- **Implementation status:** DeepSeek E2 measured with low confidence because expert-I/O variance contaminated the fit; a clean route-stable follow-up is queued.
- **Existing evidence:** DeepSeek E2 reports about 0.43 ms/KV token slope but 556-ms fit RMS and 2,398-ms constant; HERMES feasibility warns that attention assumptions decide the 30-t/s envelope.
- **Smallest falsifier:** Route-stable resident sweep at controlled contexts with no concurrent builds/background work.
- **Dependencies:** H02, H14, H38, model-specific control.
- **Speed/quality mechanism and correctness:** Establishes the ceiling after traffic is reduced; no output approximation is implied.
- **Compatibility:** Complementary with all traffic reductions; it bounds their value.

## H37 — kernel/configuration lead program

- **Original terminology / analogy:** Configuration, driver, power, CPU affinity, queue, and kernel choices may hide cheap gains; anecdotes are leads, not facts.
- **Technical interpretation:** Controlled one-variable A/B/A/B matrix with build/environment/model hashes and matched correctness.
- **Applicability:** Both; RDNA4-specific but method general.
- **Implementation status:** Active measurement discipline; many DeepSeek hardware findings exist, Qwen control intentionally keeps risky queue flags off.
- **Existing evidence:** GPU clock, GTT/ReBAR, H2D, queue, staging, and `RADV_PERFTEST` sensitivity; Qwen NGL=20 control and queue baseline.
- **Smallest falsifier:** Repeat a candidate gain after revert under the same command and state; otherwise discard.
- **Dependencies:** H02/H33/H38 and H32.
- **Speed/quality mechanism and correctness:** Removes overhead/default path losses without changing model values.
- **Compatibility:** Complementary with H12/H13; mutually exclusive with uncontrolled configuration changes during a timing series.

## H38 — accepted-token roofline/exposed-byte budget

- **Original terminology / analogy:** Work backward from accepted speed to physical traffic; storage bandwidth cannot be negotiated away.
- **Technical interpretation:** `accepted_tokens/s <= bandwidth / exposed_bytes_per_accepted_token`, with separate SSD, H2D, VRAM, compute, and acceptance terms.
- **Applicability:** Both; future MARC-native.
- **Implementation status:** Established analytical gate under stated hardware assumptions.
- **Existing evidence:** DeepSeek 30-t/s budget is roughly 92 MiB SSD and 213 MiB H2D per useful token at measured rates; Qwen Q2 simulator reports 584.9–655.4 MiB/token for naive LRU and 288.2 MiB/token for a route-history oracle.
- **Smallest falsifier:** For each proposed mechanism, measure physical bytes per accepted token below/above the roofline; a claimed 30-t/s path that misses it is falsified.
- **Dependencies:** H02, H10, H14, H19, H33.
- **Speed/quality mechanism and correctness:** Forces focus onto reuse, compact representation, residuals, and accepted spans; it is not a quality certificate.
- **Compatibility:** Complementary with every mechanism; incompatible with multiplying speculative speedups without a shared byte ledger.

## H39 — integrated HERMES-V4 architecture

- **Original terminology / analogy:** The 39 mechanisms form one architecture: representation, prediction, execution, control, safety, and learning layers governed by the cheapest recoverable path to a verified accepted block.
- **Technical interpretation:** Q8 authority + resident skeleton/residuals + Route Scout/MARC + expert-major verification + persistent atlas + asynchronous lanes + telemetry/GPS/viability + bounded learning/certificates.
- **Applicability:** DeepSeek target, Qwen-first porting host, both, and future MARC-native models.
- **Implementation status:** Architecture/design only; no integrated end-to-end HERMES-V4 result.
- **Existing evidence:** Individual HERMES transport, certificate, fidelity, union, and roofline components; integration blockers remain.
- **Smallest falsifier:** Compose only two independently certified components (for example Qwen Q2 one-layer transport plus core residency) and compare accepted-token economics to control; stop on any correctness failure.
- **Dependencies:** All load-bearing H01–H38 gates, especially H02/H33/H35/H36/H38.
- **Speed/quality mechanism and correctness:** Integration may compound gains, but only the measured composite earns a claim; exact mode remains authority-driven.
- **Compatibility:** The architecture is a composition contract, not permission to run all mechanisms simultaneously. It contains mutually exclusive exact/approximate and representation choices.

---

# Part II — Broader and newer named families

## N01 — Qwen compact expert transport

- **Original terminology / analogy:** Port the proven compact-expert transport architecture to Qwen without importing DeepSeek dimensions or assumptions.
- **Technical interpretation:** Keep canonical Q6_K/Q8_0 tensors CPU/mmap-authoritative; allocate persistent 10-slot/layer device arenas; map original top-8 IDs to local slots; rewrite only MMID IDs; use native Vulkan `MUL_MAT_ID`; fence before epoch reset; fail closed.
- **Applicability:** Qwen primary; DeepSeek transport is the reference; future MARC-native sparse models can reuse the invariants.
- **Implementation status:** **Q2 exact layer/persistence and full-core placement gates PASS, branch `qwen-compact-q2`; speed remains gated.** Layer 21 one-token/eight-token certificates and the full-core residency report are written; no speed claim exists.
- **Existing evidence:** Qwen has 40 layers, 256 experts, top-8, 2,834,432 B normal expert triplets, 3,342,336 B at exceptional layer 1, and 1.060638 GiB for all-layer top-8+2 slots. Qwen route parity passed; layer-21 source/slot bytes and control outputs are exact; changed-slot persistence saved 28.12496% physical upload bytes over upload-all; full-core A/B reduced decode-stage VRAM by 12.04 GiB with flat GTT and equal greedy tokens.
- **Smallest falsifier:** The one-layer, eight-token, one-replacement-slot fence, and placement falsifiers all passed; exceptional/multi-layer expansion remains a separate gated experiment.
- **Dependencies:** H02/H10/H13/H33, Qwen `qwen35moe.cpp`/graph integration, actual staging geometry.
- **Speed/quality mechanism and correctness:** Persistent slots may free roughly 12 GiB versus NGL=20's 12.839844-GiB routed residency, enabling full-core residency; Q2 itself is intended exact transport, not a low-bit quality change.
- **Compatibility:** Complementary with N02/N03/H10/H38; Q2 is mutually exclusive with full routed-expert residency for the selected layer during the A/B control.

## N02 — Qwen full-core GPU residency

- **Original terminology / analogy:** Use the VRAM released by compact expert transport to keep the dense/shared model core on Vulkan rather than spending VRAM on all 256 routed experts.
- **Technical interpretation:** Device-local dense/shared core (~2.494250 GiB file payload) plus runtime buffers and compact slots, while all routed authority remains CPU/mmap.
- **Applicability:** Qwen primary; concept transfers to DeepSeek only after its different core/MLA budget; future MARC-native.
- **Implementation status:** **Placement/residency A/B PASS; speed remains unvalidated.** Full-core Q2 places the dense/shared core on Vulkan while routed authority remains host/mmap.
- **Existing evidence:** NGL=20 measured Vulkan buffer 14,385.22 MiB with 12.839844 GiB routed subset; full-core Q2 measured 3,112,374,272 decode-stage VRAM bytes versus 16,036,802,560 for NGL=20, with decode-stage GTT effectively unchanged (`344,510,464` versus `344,514,560`). Qwen compact top-8+2 is 1.060638 GiB; the selected ten-slot arena is 28,344,320 bytes. GTT growth is not a meaningful substitute for device-local residency.
- **Smallest falsifier:** After Q2 certificate, load dense/shared core + one-layer slots and compare VRAM/GTT, graph placement, and decode to NGL=20 control.
- **Dependencies:** N01, Qwen model-loader overrides, H37, H38.
- **Speed/quality mechanism and correctness:** More resident core compute and fewer whole-block offload penalties; exact output should remain unchanged.
- **Compatibility:** Complementary with N01 and H10; conflicts with keeping all routed experts for the same layer in device-local memory.

## N03 — Qwen-first experimentation

- **Original terminology / analogy:** Use fast local Qwen iteration first; DeepSeek is an expensive gated confirmation, not a place to search every hypothesis.
- **Technical interpretation:** Qwen instrumentation, trace, shadow policies, twin, and Q2 are architecture evidence; DeepSeek transfer requires a fixed Qwen pass gate and separate calibration.
- **Applicability:** Qwen primary; DeepSeek confirmation only; both topology adapters are explicit.
- **Implementation status:** Active policy. Qwen baseline, topology audit, route parity, geometry, trace plan, one-layer Q2 exactness, eight-token persistence, fence/thrash, and full-core placement are complete; speed and larger P1 work remain gated.
- **Existing evidence:** Locked Qwen-first policy, 40/256/top-8/2048 topology, Qwen tracer, P1 gates G1–G7, Qwen route parity, and the Q2 certificates in `QWEN_Q2_ONE_TOKEN_CERTIFICATE.md` and `QWEN_Q2_EIGHT_TOKEN_PERSISTENCE.md`.
- **Smallest falsifier:** One trace/no-trace parity pair with identical token and route IDs; a trace-induced difference invalidates the instrumentation lane.
- **Dependencies:** H02/H33 and N01.
- **Speed/quality mechanism and correctness:** Reduces iteration cost and protects DeepSeek from premature experiments; Qwen results never become DeepSeek performance claims.
- **Compatibility:** Complementary with all Qwen programs; incompatible with DeepSeek-first broad searches before the Qwen gate.

## N04 — semantic fingerprints

- **Original terminology / analogy:** A compact fingerprint says what the host is doing—objective, reasoning phase, entities, constraints, memory references, style, uncertainty, and phase lifetime—rather than relying on keywords.
- **Technical interpretation:** Typed tuple of explicit route IDs/margins/entropy/hidden deltas and learned latents, with field-specific cadence and expiry; F10 interrupt/refresh condition is explicit.
- **Applicability:** Qwen first; DeepSeek transfer after Qwen PASS; future MARC-Synapse native.
- **Implementation status:** Static schema complete; Qwen tracer partially implemented; no P1 gate result; historical keyword/hash MARC fingerprint is a baseline, not a semantic success.
- **Existing evidence:** Qwen signal map and schema; DeepSeek E1A route families persist over short phrase positions, but long phase stability is unknown. MARC V0 audit confirms its old fingerprint was keyword/hash only.
- **Smallest falsifier:** Qwen P1 B4 structured fingerprint must beat B0/B1 AMI and boundary/hotset gates on held-out prompts with no future leakage.
- **Dependencies:** N03, Qwen trace parity, H02, H14.
- **Speed/quality mechanism and correctness:** Predicts phase changes and memory/precision needs; it may guide compute/residency but never correctness alone.
- **Compatibility:** Complementary with N05/N09/H06/H07; keyword-only and final-hidden-only alternatives are baselines, not merged semantics.

## N05 — hardware fingerprints

- **Original terminology / analogy:** The execution body's hardware conditions are part of its identity: resident IDs, slot ages, cache tier, staging, queues, precision forms, and memory pressure.
- **Technical interpretation:** Typed snapshot fields H1–H11, authoritative at decision time; stale fields after an epoch are unknown, not valid.
- **Applicability:** DeepSeek live HERMES; Qwen simulated until Q2 exposes live slots; future MARC-native.
- **Implementation status:** DeepSeek schema/static design complete; Qwen fields are twin-only in the P1 policy; Q2 will add real slot/staging counters.
- **Existing evidence:** DeepSeek HERMES store/cache/staging counters; Qwen Q2 manager's fail counters and expected diagnostic markers.
- **Smallest falsifier:** Predict a Q2 upload or stale-slot event from a snapshot, then compare against the actual one-layer trace; stale state must not produce an applied action.
- **Dependencies:** N01, H10, H14, N04 for joint binding.
- **Speed/quality mechanism and correctness:** Avoids decisions based on an imagined cache/queue; no direct quality approximation.
- **Compatibility:** Complementary with N04/N09/H15; semantic-first then hardware-optimization is explicitly rejected.

## N06 — MARC-X

- **Original terminology / analogy:** Apply Modular Architecture with Routing and Control to existing MoE models through observation: hotsets, expert logging, residency, prefetch, substitution, and latency/quality drift.
- **Technical interpretation:** Practical layer for Qwen/DeepSeek runtime interventions over existing router surfaces, not a new trained model.
- **Applicability:** Qwen and DeepSeek; future MARC-native is not its main target.
- **Implementation status:** Historical probing and routing vocabulary exist; direct live residency controller is not complete. HERMES/Qwen Q2 are the concrete successor substrate.
- **Existing evidence:** MARC overview/audit, Qwen probing work, HERMES route/slot/cache traces.
- **Smallest falsifier:** Shadow a hotset/prefetch or bounded substitution policy on a real MoE trace and require improvement over exact control with no correctness regression.
- **Dependencies:** N03/N04/N05, H06/H10/H14/H33.
- **Speed/quality mechanism and correctness:** Reduces movement or selects cheaper paths; substitutions are approximate/verified until proven.
- **Compatibility:** Complementary with HERMES; should not be silently conflated with H07's Margin-Aware Routing Calibration.

## N07 — MARC-OS

- **Original terminology / analogy:** A general operating system for conditional compute: choose the cheapest path under a quality-risk budget.
- **Technical interpretation:** Hardware-aware policy across precision, active modules/layers, context, prefetch, verification, escalation, and model choice.
- **Applicability:** Both existing models and future MARC-native.
- **Implementation status:** Historical Qwen budget proxy shows a working policy loop; the old fingerprint is keyword-based and the model-internal runtime is not complete.
- **Existing evidence:** MARC V0.7/V0.8 proxy: concise_256 around 6.85 s versus default thinking 1,024 around 52.97 s with different quality/judge semantics; this is prompt-budget evidence, not MoE transport evidence.
- **Smallest falsifier:** On a held-out Qwen workload, compare fixed budget policy, hardware-aware policy, and oracle under matched quality and resource limits.
- **Dependencies:** N04/N05, H15/H25/H29/H34, H28 quality battery.
- **Speed/quality mechanism and correctness:** Chooses less work while keeping a measured risk bound; no exact claim from token-budget reduction alone.
- **Compatibility:** Complementary with N06/N09; keyword-only routing must remain a baseline.

## N08 — MARC-Synapse

- **Original terminology / analogy:** Train a modular model from scratch: shared core, semantic router, module bank, associative memory, verification/escalation, and hardware-aware residency.
- **Technical interpretation:** A future architecture in which the prompt assembles a temporary model from specialized modules.
- **Applicability:** Future MARC-native primarily; not a direct Qwen/DeepSeek runtime patch.
- **Implementation status:** Historical V0 only; toy equal-active-FLOP modular FFN lost to monolithic baseline (ppl 3.88 vs 2.68), and router collapsed about 92% of prompts into two modules.
- **Existing evidence:** MARC-Synapse audit and overview; negative result is preserved.
- **Smallest falsifier:** A controlled equal-active-compute model with a non-keyword semantic router, held-out task mixture, and residency-aware module cost must beat a matched dense baseline on quality-per-joule or it remains a failed architecture at that scale.
- **Dependencies:** N04/N05/N09, associative memory N25, H28.
- **Speed/quality mechanism and correctness:** Conditional module activation and escalation; correctness is model-quality/verified, not exact equivalence to Qwen/DeepSeek.
- **Compatibility:** Long-term complementary context; mutually exclusive with claiming the toy V0 already proves the full vision.

## N09 — MARC-Symbiote temporary execution body

- **Original terminology / analogy:** A frozen authoritative host temporarily assembles an execution body from explicit primitives and bounded learned latents, then interrupts/refreshes it when semantic meaning, uncertainty, or requirements change.
- **Technical interpretation:** Joint semantic×hardware binder selects KEEP_SLOT, PREFETCH_EXPERT_UNION, GROUP_BY_EXPERT, USE_ANCHOR, APPLY_REFINEMENT, REQUEST_AUTHORITY, REFRESH_HOST, or HALT. It is not speculative token drafting.
- **Applicability:** DeepSeek host in the original design; Qwen-first trace/twin gate; future MARC-native.
- **Implementation status:** Static design only; no graph modification, training, executor, DSpark restore, or speed claim permitted.
- **Existing evidence:** Symbiote architecture, semantic/hardware schemas, P0/P1/P2 phasing, and the explicit distinction from speculative decoding.
- **Smallest falsifier:** Qwen P1 phase/hotset gate, then CPU-only binder twin; failure after two signal redesigns stops expansion.
- **Dependencies:** N04/N05, H02/H10/H14/H29/H33, Qwen-first gate.
- **Speed/quality mechanism and correctness:** Keeps useful residency/precision warm during a semantic phase; host remains authoritative, so body output is trace/simulator/verified/approximate as labeled.
- **Compatibility:** Complementary with H10/H06/H05; explicitly not a replacement for H08 DSpark.

## N10 — REAP

- **Original terminology / analogy:** Remove low-value experts based on observed routing/saliency so capacity can be spent on survivors.
- **Technical interpretation:** Per-layer expert token counts, routing-weight sums, retained-ID manifests, and router/tensor rewriting at checkpoint/conversion time.
- **Applicability:** Laguna and Qwen-like sparse MoEs; DeepSeek only with a separate checkpoint/converter path; future MARC-native training.
- **Implementation status:** Laguna integration design and Qwen quality-floor calibration artifacts exist; full model quality gate is not complete.
- **Existing evidence:** Laguna 256-expert/47-layer inventory; synthetic layer-1 pilot Spearman 0.9818 and top-50 overlap 98%, explicitly limited by synthetic data; virtual pruning code.
- **Smallest falsifier:** Real held-out calibration, prune one layer at 10–60%, run logits/task quality and route validity against random-pruning control.
- **Dependencies:** H02/H28, N12, N13, variant experiments.
- **Speed/quality mechanism and correctness:** Fewer experts reduce file/compute/memory, but router rewrite changes the model and is approximate unless retrained/validated.
- **Compatibility:** Complementary with N11/N12/N14–N16; mutually exclusive with an unchanged-model exact claim.

## N11 — Laguna

- **Original terminology / analogy:** Laguna-S compression/speculative framework and its resident-128/low-bit model variants.
- **Technical interpretation:** A large MoE compression/conversion/runtime path with fused expert tensors, speculative draft support, and configurable context/memory budget.
- **Applicability:** Laguna target models; method informs Qwen/DeepSeek compression but is not a drop-in Q2 transport.
- **Implementation status:** Archived/complete experiment family; outputs on HDD and engine fork required to resume.
- **Existing evidence:** 67-G Laguna R4I8 and 28-G R2I8-resident128 outputs; 235-GB Laguna-S-2.1 inventory; 256K budget shows Q4 KV and 40–60% pruning are necessary for 48-GB targets.
- **Smallest falsifier:** Restore one archived variant, verify converter/runtime hashes, and compare exact/logit/task quality and memory against an uncompressed control.
- **Dependencies:** N10/N12/N14–N16, N19, H28.
- **Speed/quality mechanism and correctness:** Compresses/prunes weight and context budgets; output is variant-specific approximate unless direct equivalence is shown.
- **Compatibility:** Complementary as a model-production lane; not combined with the fixed Qwen Q6 control in one claim.

## N12 — model pruning

- **Original terminology / analogy:** Surgically remove experts/parameters that are dispensable under measured calibration while preserving a usable model.
- **Technical interpretation:** Layer-specific retained IDs, router row/bias rewriting, expert tensor slicing, streaming conversion, and quality gates.
- **Applicability:** Laguna/Qwen-like models; DeepSeek future; future MARC-native can train for structured sparsity.
- **Implementation status:** Virtual-pruning/converter support exists; no complete three-model quality frontier.
- **Existing evidence:** REAP-Laguna integration audit explains checkpoint-level surgery and says different layers can retain different counts; real end-to-end validation remains open.
- **Smallest falsifier:** Compare REAP-ranked and random pruning at equal retained bytes on held-out prompts, including selected-ID validity and downstream logits.
- **Dependencies:** N10/N11/H02/H28.
- **Speed/quality mechanism and correctness:** Reduces resident/streamed bytes and expert work; changes model semantics and is approximate until validated.
- **Compatibility:** Complementary with quantization; mutually exclusive with unchanged authority for the pruned model.

## N13 — AutoSurgeon

- **Original terminology / analogy:** Preserve the project's name **AutoSurgeon** for automated, evidence-driven model surgery across pruning, quantization, routing, layer selection, and conversion.
- **Technical interpretation:** A manifest-producing orchestrator that consumes calibration, saliency, hardware budget, and quality gates, then emits reversible model variants and certificates; it must not silently mutate authority.
- **Applicability:** Qwen/Laguna first; DeepSeek later; future MARC-native.
- **Implementation status:** **Unlocated/unvalidated in the durable project files ingested here.** The exact AutoSurgeon artifact was not found, so this entry preserves the requested name without inventing implementation details. REAP/Laguna scripts are related evidence, not proof of AutoSurgeon.
- **Existing evidence:** Laguna conversion/pruning scripts and three-strategy code provide component primitives; no AutoSurgeon API, manifest, or result was located.
- **Smallest falsifier:** Define a dry-run manifest for one Qwen/Laguna layer, reproduce the chosen bytes and quality from the manifest, and verify rollback to the source hash.
- **Dependencies:** N10–N12, N14–N16, H02/H28/H33.
- **Speed/quality mechanism and correctness:** Automates the quality/size frontier; certificates and source immutability are required because surgery changes model behavior.
- **Compatibility:** Complementary with pruning/quantization; not a synonym for REAP or Laguna and not allowed to replace a human interpretation of unresolved variant goals.

## N14 — three practical compressed model variants: coverage-first

- **Original terminology / analogy:** Keep all 256 experts at a very low average precision so coverage is preserved.
- **Technical interpretation:** The `three_strategies.py` target is approximately 2.36 bpw for all routed experts, using improved asymmetric/per-channel quantization.
- **Applicability:** Laguna target first; candidate for Qwen/DeepSeek only after native quantizer and quality gates.
- **Implementation status:** Script/design; no reported complete model-level result.
- **Existing evidence:** Script defines the byte-equal comparison and layer samples; no end-to-end quality certificate.
- **Smallest falsifier:** One real layer and held-out activations/logits versus precision-first and heterogeneous at exactly equal routed bytes.
- **Dependencies:** N13, N19–N21, H28, native kernels.
- **Speed/quality mechanism and correctness:** Maximizes route coverage and may reduce missing-expert penalties; approximate representation.
- **Compatibility:** A competing variant to N15/N16 at fixed byte budget, not silently averaged with them.

## N15 — three practical compressed model variants: precision-first

- **Original terminology / analogy:** Keep only the most valuable experts at high precision rather than all experts at low precision.
- **Technical interpretation:** The target script retains roughly 67 experts at approximately Q6/Q9-style precision under the same routed byte budget, with router pruning.
- **Applicability:** Laguna first; Qwen/DeepSeek future.
- **Implementation status:** Script/design; no complete quality frontier.
- **Existing evidence:** `three_strategies.py` specifies the retained-saliency experiment; REAP ranking stability is only pilot/synthetic evidence.
- **Smallest falsifier:** Equal-byte held-out route/quality test against N14 and N16, including a prompt class that activates low-ranked experts.
- **Dependencies:** N10/N12/N13, H02/H28.
- **Speed/quality mechanism and correctness:** Higher survivor fidelity and fewer active storage objects; pruned router changes the model.
- **Compatibility:** Competes with N14/N16 at fixed bytes; can be combined with H10 but not an unchanged-model exact claim.

## N16 — three practical compressed model variants: heterogeneous

- **Original terminology / analogy:** Spend precision unevenly: approximately 40 Q6 experts, 80 R4I8 experts, 80 2.5-bit experts, and 56 pruned in the target script.
- **Technical interpretation:** Layer/expert saliency selects several quantization forms and removes the tail under one routed byte budget.
- **Applicability:** Laguna first; Qwen/DeepSeek future.
- **Implementation status:** Script/design; no complete model-level result.
- **Existing evidence:** `three_strategies.py` and R4I8 code provide the proposed composition; quality is unmeasured.
- **Smallest falsifier:** Equal-byte per-layer and end-to-end quality/latency comparison against N14/N15, with rare-expert stress prompts.
- **Dependencies:** N10–N13, N19–N21, N22/N23, H28.
- **Speed/quality mechanism and correctness:** Allocates bits where sensitivity is high; approximate and model-specific.
- **Compatibility:** Competing byte-budget variant; can coexist with H04/H07 but not be presented as Q6 authority.

## N17 — MoE-Skipper cascade and correction systems

- **Original terminology / analogy:** Replace selected MoE work with learned low-rank gate/down predictors, trained in execution order so downstream predictors see cascade-disturbed inputs; repair the error wall with residual/cascade correction and exact tails.
- **Technical interpretation:** Per-layer affine normalization plus rank-64 gate/down predictors, layer policies, cascade training, exact-tail cutoffs, and direct logit gates.
- **Applicability:** Qwen3.6-35B-A3B primary; other sparse MoEs with architecture-specific retraining; not a direct DeepSeek exact transport path.
- **Implementation status:** Quality-valid approximate Qwen L25/L30 F16 path has a fixed long-prefill gain (+17.62%, 112.95 vs 96.03 tok/s) but is context/batch dependent; three-layer and broad all-layer paths fail or regress.
- **Existing evidence:** Cascade research: L20/L25/L30 quality-valid at KL ~0.04–0.047 and +5.6% in one table; claims ledger records the stronger F16 long-workload result, negative generalization, and all-layer KL 9.27/0% same-top invalidity.
- **Smallest falsifier:** One layer direct-logit gate, then two-layer cascade on an independent corpus before any speed run.
- **Dependencies:** H02/H28/H33, Qwen baseline, exact supervision.
- **Speed/quality mechanism and correctness:** Removes selected MoE compute in approximate prefill; no exact generation claim and cascade error must be measured.
- **Compatibility:** Complementary as an approximate branch to Q2/exact paths; mutually exclusive with exact Q6 output claims and not equivalent to H04 progressive expert widening.

## N18 — DSpark/MTP restoration and custom runtime

- **Original terminology / analogy:** Restore the missing future-token module and build the custom runtime needed for block proposal, route/memory lookahead, and strict verification.
- **Technical interpretation:** Converter tensor support, model/draft format, confidence/Markov heads, expert predictor, byte-budget horizon, prefetch scheduler, and causal block verifier.
- **Applicability:** Qwen has local DSpark/MTP artifacts; DeepSeek official DSpark is the original target but the current Unsloth-derived GGUF has no `mtp.*`; both need separate adapters.
- **Implementation status:** Audit/design; DeepSeek restoration blocked by missing tensors/converter/runtime; Qwen MTP is deliberately disabled during first gates.
- **Existing evidence:** Official DSpark audit reports about 4,705 tensors and 10.8 GB; static predictor holdout corrected prior leakage (transition F1 0.356 vs 0.292 baseline); current Qwen-first policy gates MTP later.
- **Smallest falsifier:** Restore only tensor inventory/conversion and run a K=1 parity check; any missing tensor or target-state mismatch blocks the runtime.
- **Dependencies:** H02/H08/H09/H13/H38, N03, Q2/full-core for residency economics.
- **Speed/quality mechanism and correctness:** Accepted spans amortize work; strict rejection/repair required, and draft cost/bytes cannot be hidden.
- **Compatibility:** Complementary with H06/H08; not combined with unsupported MTP claims or MTP/DSpark being enabled in Q2 baseline.

## N19 — R4I8

- **Original terminology / analogy:** Round-4-in-8: 4-bit values stored in 8-bit containers with a specialized cooperative-matrix-friendly format, plus distillation.
- **Technical interpretation:** Custom GGML type and Vulkan shader with block scale/nibble layout, direct Qwen model conversion, and optional heterogeneous auxiliary models.
- **Applicability:** Qwen primary; Laguna artifacts; future models only with format support.
- **Implementation status:** Structural format path works after fixing block-byte order; model output is varied but quality at about 4.5 bpw is low/gibberish in the recorded checkpoint.
- **Existing evidence:** R4I8 checkpoint confirms CPU/Vulkan dispatch and `total=3296` cooperative-matrix dispatches; open BOS metadata and quality issues remain.
- **Smallest falsifier:** Byte-order/unit test, CPU/Vulkan logits comparison, then held-out quality and task battery against Q6/Q8.
- **Dependencies:** N22/N23, H02/H28/H33.
- **Speed/quality mechanism and correctness:** Lower bytes and potentially coopmat-friendly arithmetic; format parity is not model quality.
- **Compatibility:** Alternative model representation to Qwen Q6 control; can inform N14/N16 but cannot be silently mixed into Q2.

## N20 — R5I8

- **Original terminology / analogy:** Preserve the distinct R5I8 name for a future round-5-in-8 format rather than assuming R4I8 generalizes.
- **Technical interpretation:** A prospective higher-fidelity 5-bit-in-8 representation with its own scale/layout, converter, shader, and byte/quality point.
- **Applicability:** Future Qwen/Laguna/DeepSeek model variants; not currently model-proven.
- **Implementation status:** No durable implementation or result was located; research placeholder requiring the project's exact format definition.
- **Existing evidence:** R4I8 infrastructure supplies a neighboring format experiment only; it is not evidence for R5I8.
- **Smallest falsifier:** Define a block format and compare reconstruction error, bytes, shader viability, and logits on one layer before conversion work.
- **Dependencies:** N19, N22/N23, H28/H33.
- **Speed/quality mechanism and correctness:** Expected fidelity/byte tradeoff is unknown; approximate until validated.
- **Compatibility:** Competes with N19/N21 as a representation; not a Q6 control.

## N21 — R6I8

- **Original terminology / analogy:** Preserve the distinct R6I8 name for a still-higher-fidelity 6-bit-in-8 format.
- **Technical interpretation:** Prospective 6-bit-in-8 layout, likely a quality/control point between Q6-style values and R4I8, with custom converter/shader.
- **Applicability:** Future Qwen/Laguna/DeepSeek variants.
- **Implementation status:** No durable implementation or result located; placeholder, not an implied claim.
- **Existing evidence:** Qwen Q6_K_XL and R4I8 provide endpoints/neighboring formats only.
- **Smallest falsifier:** One-block format specification and CPU reconstruction/VRAM shader parity test; stop if overhead erases the intended point.
- **Dependencies:** N19/N20, N22/N23, H28/H33.
- **Speed/quality mechanism and correctness:** Potentially better quality at more bytes; exactness only if it is the authority itself, otherwise approximate.
- **Compatibility:** Alternative representation; do not combine with Q6 control in one unchanged-model claim.

## N22 — RDNA4 cooperative-matrix kernels

- **Original terminology / analogy:** Use RDNA4-native cooperative matrix hardware rather than generic shader paths.
- **Technical interpretation:** 16×16×16 fp16/bf16/int8/fp8 cooperative matrices, layout/packing, dispatch, and dequant integration for R4I8 or other supported forms.
- **Applicability:** Qwen/R4I8 and future low-bit paths; DeepSeek IQ3 currently does not use coopmat in its MMID path.
- **Implementation status:** Capability audited; R4I8 dispatch works structurally; HERMES IQ3/Qwen Q6 coopmat path is not validated.
- **Existing evidence:** RDNA4 probe exposes coopmat; R4I8 fixed checkpoint reports native dispatch; hardware audit says DeepSeek IQ3 MMID is dequant/FP dot, not coopmat.
- **Smallest falsifier:** Same tensor/bytes benchmark with coopmat and existing path, CPU/Vulkan parity, and end-to-end accepted-token measurement.
- **Dependencies:** N19–N21 or a supported quant type, N23, H33/H37.
- **Speed/quality mechanism and correctness:** May improve packing/dequant/occupancy; hardware capability alone does not imply a gain.
- **Compatibility:** Complementary to formats that fit; independent A/B kernel choice, not a reason to enable risky queue flags.

## N23 — RDNA4 GEMV/GEMM kernels

- **Original terminology / analogy:** Tune the actual GEMV/GEMM shapes—especially single-token MoE MMID and expert-major batches—for RDNA4.
- **Technical interpretation:** Wave choice, MMID tiling, fused dequant, launch batching, and separate GEMV versus GEMM paths for Q6_K/Q8_0/R4I8.
- **Applicability:** Qwen and DeepSeek; future MARC-native.
- **Implementation status:** Existing generic Q6/Q8 `MUL_MAT_ID` pipelines are present; custom shape work is unvalidated.
- **Existing evidence:** Qwen audit confirms Q6_K/Q8_0 ID pipelines; DeepSeek audit shows MMID is bandwidth/instruction limited; R4I8 has custom path.
- **Smallest falsifier:** Microbenchmark the exact Qwen `[2048,512,10]` and `[512,2048,10]` shapes plus end-to-end one-layer timing against generic path.
- **Dependencies:** N01, N22, H09/H12/H37.
- **Speed/quality mechanism and correctness:** Improves useful compute and launch overhead after bytes are controlled; parity required.
- **Compatibility:** Complementary with Q2 and H09; kernel speedups cannot be multiplied independently of transport.

## N24 — context-only streaming

- **Original terminology / analogy:** Separate context/KV streaming from parameter/expert streaming: keep model parameters/core resident while moving or compressing only the context state needed for the current phase.
- **Technical interpretation:** A context-memory lane with exact recent KV, compressed segment checkpoints, selective retrieval, and exact fallback; it is distinct from routed expert weight I/O.
- **Applicability:** DeepSeek context/MLA first; Qwen recurrent/delta/full-attention mix later; future MARC-native.
- **Implementation status:** Research-only extension; no runtime modification.
- **Existing evidence:** HERMES IDEA 40/dual hierarchical memory and context budget analyses; clean E2 attention measurement is still needed.
- **Smallest falsifier:** On a model exposing memory states, retain recent context exactly, compress older segments, retrieve a small subset, and compare recall/logits/latency/fallback rate.
- **Dependencies:** H36, N25, H02/H28/H33.
- **Speed/quality mechanism and correctness:** Reduces KV/context memory and attention exposure; exact fallback and task-quality gates required.
- **Compatibility:** Complementary with expert transport; must not be described as reducing expert-weight bytes.

## N25 — associative memory

- **Original terminology / analogy:** A semantic/module bank and context memory retrieve relevant facts/states rather than replaying all history; HERMES's expert atlas is a separate physical associative map.
- **Technical interpretation:** Explicit memory keys/references, stable constraints, segment checkpoints, sparse retrieval, and invalidation/refresh rules.
- **Applicability:** Future MARC-Synapse and Symbiote; context-only experiments on Qwen/DeepSeek.
- **Implementation status:** Historical MARC-Synapse design and HERMES context-memory proposal; no production memory executor.
- **Existing evidence:** MARC-Synapse architecture includes associative memory; IDEA 40 cites growing-memory/sparse selective caching as direct prior art for context memory, not expert streaming.
- **Smallest falsifier:** CPU twin with an unchanged source ledger: retrieve top-k memory entries for held-out phases and measure recall, stale retrieval, bytes, and fallback.
- **Dependencies:** N04/N09/N24/H02/H33.
- **Speed/quality mechanism and correctness:** Compresses/reduces context reads and preserves stable facts; stale memory must trigger host refresh.
- **Compatibility:** Complementary with N24 and semantic fingerprints; not a replacement for H10 slot atlas or Q2 expert authority.

## N26 — autonomous local self-experimentation

- **Original terminology / analogy:** A bounded local loop should propose hypotheses, prepare experiments, run only authorized small tests, judge evidence, preserve failures, and update the queue without silently changing the program.
- **Technical interpretation:** Versioned experiment specs, dependency-aware scheduler, hardware lock, finite budgets, source/result hashes, Explorer/Verifier separation, rollback, and explicit UNKNOWN/FAILED outcomes.
- **Applicability:** All current Qwen/DeepSeek/compression programs; future MARC-native research infrastructure.
- **Implementation status:** Method/policy family; pieces exist in HERMES certificates, Qwen-first policy, and trace/twin plans, but no unified research daemon.
- **Existing evidence:** One-load-bearing-experiment rule, `temporary benchmark launcher` lock discipline, Qwen fixed gate, preserved negative MoE-Skipper records, and Q2 fail-closed counters.
- **Smallest falsifier:** Run a CPU-only queue slice with one planned experiment, an injected failure, and a verifier; require no promotion and a reproducible failure record.
- **Dependencies:** H02/H14/H29/H33 and the complete graph/queue.
- **Speed/quality mechanism and correctness:** Increases information gained per local resource while preventing experiment contamination; it does not authorize autonomous model/runtime changes.
- **Compatibility:** Complementary with all families; prohibited from launching all experiments simultaneously or bypassing the project's unresolved interpretation decisions.

---

# Part III — Durable evidence ledger and unresolved interpretation

## Current hardware/model facts that constrain the atlas

- **Qwen control:** `[model payload omitted]`, 31,843,777,504 bytes, `qwen35moe`, 40 layers, 256 experts/layer, top-8, embedding 2048, Q6_K/Q8_0 mixed routed tensors. NGL=20 is the control; its measured Vulkan allocation is 14,385.22 MiB. Queue experiments are disabled.
- **Qwen expert geometry:** normal gate/up Q6_K and down Q8_0 are 2,834,432 B/expert; layer 1 is all-Q8_0 at 3,342,336 B/expert. Ten slots across all layers are 1,086.094 MiB/1.060638 GiB before runtime padding.
- **Qwen route trace:** 24 single-token generation steps, 7,680 logical requests, 2,545 distinct layer/expert pairs, 48–112 distinct experts per layer, mean 63.625. Qwen A/B route parity passed. Simulator LRU 8/9/10 uploads 655.430/613.760/584.907 MiB/token; route-history oracle 288.218 MiB/token. These are simulation sensitivities, not upload benchmarks.
- **DeepSeek/HERMES:** transport certificate and control are strong; E1/E1A skeleton fidelity and E2 floor remain model-specific; full Q8 streaming is storage-bound and 30 accepted tok/s is conditional/no-go on current evidence.
- **MARC:** old V0 keyword/hash fingerprint and prompt-budget proxy are historical baselines. The new Symbiote design is static and explicitly not speculative drafting.
- **MoE-Skipper:** fast invalid broad skipping is preserved as negative evidence; the quality-valid F16 L25/L30 result is workload-specific approximate prefill evidence, not exact generation.
- **Compression:** REAP/Laguna/R4I8 are model-production alternatives, not Qwen Q2 authority. R4I8 byte order is fixed, but quality remains low in the recorded Qwen output.

## Contradictions and duplicate-looking mechanisms that need the project's interpretation

1. **MARC collision:** H07 is **Margin-Aware Routing Calibration**; N06–N09 are the project's **Modular Architecture with Routing and Control** lineage. Keep both names and never use the acronym alone.
2. **Q2 compact versus compact skeleton:** N01 copies exact Q6_K/Q8_0 source slices into persistent slots; H03 changes representation to low-bit. They are complementary, not the same experiment.
3. **P1/P2/P4/P6 versus MoE-Skipper:** H04 is runtime nested expert widening; N17 is learned layer-output substitution with cascade error. They must have separate quality gates.
4. **Qwen full-core versus Q2 source authority:** N02 needs routed experts off device-local memory; N01 keeps CPU/mmap authority and selected slots. A full all-expert resident control is a competing placement, not an implementation detail.
5. **DSpark versus MTP:** N18 preserves both names. Official DeepSeek `mtp.*` absence, Qwen MTP artifacts, and unrelated MTP runs are separate evidence.
6. **R4I8/R5I8/R6I8:** only R4I8 has a structural implementation record. R5I8 and R6I8 are named future formats, not implied results.
7. **Expert atlas versus associative memory:** H10 maps physical expert IDs to slots; N25 stores semantic/context references. Shared vocabulary does not make them one mechanism.
8. **Context-only streaming versus expert streaming:** N24 moves/compresses context state; H06/H10/H11/N01 move expert weights. Their byte ledgers must remain separate.
9. **AutoSurgeon ambiguity:** the exact durable artifact was not located. N13 is preserved as a named placeholder/manifest program, while REAP/Laguna scripts are evidence only.
10. **Three compressed variants:** N14/N15/N16 are competing fixed-budget hypotheses, not a single “compressed model” result.
11. **Cooperative matrix versus GEMV/GEMM:** N22 is hardware/instruction form; N23 is shape/kernel scheduling. Hardware availability does not imply an end-to-end gain.
12. **Exact versus approximate integration:** H39 can compose mechanisms only after each component's correctness class and byte accounting are retained; no multiplicative speedup arithmetic is allowed.

## Queue rule

Only the completed Qwen Q2 ordinary-layer line owns the recorded GPU evidence. All other families are either static preparation, trace/simulation, or explicitly gated after Q2/Qwen correctness. The next experiment queue, dependency graph, and omission report are in the companion files.
