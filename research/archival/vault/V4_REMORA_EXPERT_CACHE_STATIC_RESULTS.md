# V4 REMORA Expert-Cache — Static Results (CPU/RAM/SSD only)

**Status:** STATIC + CPU-ONLY SIMULATION. GPU untouched (see §0). No training, no long
inference, no VRAM allocation, no model bytes loaded. All artifacts in
`results/v4_expert_cache_static/` (this directory's sibling), produced by
`sim_cache.py` (stdlib-only Python, `nice -n 10`, single thread, peak RSS ≪ 1 GiB).

---

## 0. No-GPU confirmation (this session)

| check | result |
|---|---|
| GPU devices exposed | AMD via `[device path omitted]`, `renderD128`; ROCM_PATH set globally; **no NVIDIA** |
| VRAM occupancy (sysfs, read-only) | **9.7 / 17.1 GB in use** — training owns the GPU; never touched |
| GPU env guards for every workload | `CUDA_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" GGML_VK_VISIBLE_DEVICES=""` (GGML_VK_VISIBLE_DEVICES is the real llama.cpp Vulkan filter, verified in `ggml/src/ggml-vulkan/ggml-vulkan.cpp:7176`) |
| GPU libraries initialized | none (stdlib Python only; no llama.cpp/Vulkan/ROCm processes started) |
| Active training run | PID 604896 `v12_warm_precision.py` observed (read-only); not killed, not env-modified; all heavy work `nice -n 10 ionice -c 3`, ≤1 core |
| RAM headroom | 22 GiB available before work; sim peak RSS < 0.3 GiB (≪ 8 GiB budget) |
| Disk | 42 GB free on /home; artifacts total ~250 KB; no model copies, no temp files |

## 1. Measured inputs (prior artifacts, re-verified this session)

- Model: DeepSeek-V4-Flash-0731 UD, 43 layers, 256 experts, 6/token, experts MXFP4
  (17 B/32 el), 4,456,448 B per projection, **12.75 MiB per expert**, bank 137.06 GiB.
- Routing traces: `results/hermes-v4/traces/*.tr` (6 prompts × 256–512 tokens) +
  `traces/dsv4-16token-trace.jsonl` (17) + `traces/dsv4-first-trace.jsonl` (3) =
  **2,836 tokens, 731,688 expert accesses, 11,008 identities**.
- DSEI v2 index `traces/expert_index_0731_v2.bin` (33,024 spans) — used for the SSD
  access analysis (file offsets are real).
- Bandwidth constants (prior measured; labels retained): cold scatter 1.22 GiB/s,
  nominal 2.9 GB/s, warm QD8 7.20 GiB/s, DRAM gather 6.8 GB/s, H2D GTT 6.4 GB/s.
- Measured decode attribution (resident route-stable): 222.5 ms/token = copy 36.3 +
  cpu 19.3 + vk 47.5 + sync 87.8 + cb 0.5 + build 38.5 ms; sched-loop overhead
  310–475 ms/decode (era measurement).

## 2. Derived outputs (this session — all artifacts in `results/v4_expert_cache_static/`)

### 2.1 Locality metrics (2,836 tokens — larger and stricter than the prior 17-token study)

| metric (mean over 43 layers) | value | prior (17 tok) | reading |
|---|---|---|---|
| unique experts/layer | **237.4 / 256** | 28–77 | at 512-token scale nearly every expert appears; prior "small working set" was a warmup artifact |
| entropy | **6.65 bits** (max 8) | — | near-uniform marginal distribution |
| next-token overlap (same layer) | **1.68 / 6** | 2.2 / 6 | real signal, weaker than the prior trace suggested |
| next-token Jaccard / F1@6 | 0.186 / **0.309** | 0.254 / 0.368 | same direction, lower magnitude |
| median reuse distance | 3.7 tokens (mean 1.4) | — | half the experts recur within ~4 tokens |
| streak length | mean 3.0, **max 256** | — | some experts fire nearly every token in a layer |
| burstiness (dispersion index) | ~3,500 | — | extremely bursty: heavy concentration in short windows |
| top-6 / top-12 / top-48 cumulative | 23.6% / 35.7% / **70.0%** | 31–64% (top-6) | long tail is real: 6 slots/layer covers <25% of accesses |
| cross-layer set overlap (ΔL=1..42) | 0.022–0.028 | ≈0.02 | **cross-layer prediction re-confirmed falsified** |

**Verdict:** the routing is dominated by per-layer temporal burstiness with a very long
tail. Neither "small static working set" nor "predictable cross-layer" models hold.
Cache policy must be temporal (recency-heavy), per-layer, and tolerant of the tail.

### 2.2 Working-set curves (HOT slots/layer; `working_set_curves.csv`)

| HOT slots/layer | LRU hot-hit | freq-oracle hot-hit | SSD GiB/token (no WARM) |
|---|---|---|---|
| 1 | 0.3% | 6.2% | 3.29 |
| 6 | 20.1% | 23.6% | 2.64 |
| 12 | 39.7% | 35.7% | 1.98 |
| 24 | 56.7% | 51.3% | 1.42 |
| 48 | 79.2% | 70.0% | 0.69 |
| 64 | **87.7%** | 77.7% | 0.40 |

- LRU beats the static frequency oracle beyond 12 slots/layer (the tail is time-varying).
- **64 slots/layer (43 × 64 × 12.75 MiB = 35 GiB) reaches 88% hot hits** — far beyond the
  VRAM budget; the HOT tier cannot carry the tail alone on 16 GiB VRAM.
- 6 slots/layer (the current implementation) leaves **~80% of accesses to the WARM/RAM
  path (2.42 GiB/token of RAM gather at 4 GiB WARM)** — this is the real current cost.

### 2.3 RAM-tier sweep (S=6 HOT; `ram_tier_sweep.csv`)

| WARM | experts held | cold miss | SSD MiB/token |
|---|---|---|---|
| 256 MB | 20 | 74.4% | 2,448 |
| 1 GB | 80 | 56.1% | 1,847 |
| 2 GB | 160 | 29.9% | 982 |
| **4 GB** | 321 | **0.28%** | **9.1** |
| 8 GB | 642 | 0.28% | 9.1 |

- **A 4 GiB WARM tier already collapses SSD traffic to ~0** on these traces (the whole
  ~240-experts/layer tail fits in 321 global slots over 512-token prompts). WARM > 4 GiB
  buys nothing at the byte level.
- The SSD bottleneck is a **small-WARM** problem (≤2 GiB); the RAM-gather / H2D path
  (2.42 GiB/token at S=6) is the binding cost in the WARM regime.

### 2.4 Policy comparison (S=6 HOT, 4 GiB WARM; `policy_comparison.csv`)

| policy | hot | warm | cold | SSD MiB/t | RAM GiB/t |
|---|---|---|---|---|---|
| A none | 0% | 0% | 100% | 3,290 | 0 |
| B LRU | 19.9% | 75.2% | 4.8%* | 158* | 2.42 |
| C freq-aware | 12.5% | 82.7% | 4.8%* | 158* | 2.66 |
| D short-term reuse | 13.2% | 82.0% | 4.8%* | 158* | 2.64 |
| E LRU+prefetch | 19.9% | 75.2% | 4.8%* | 158* | 2.42 |
| F hybrid | **22.0%** | 73.2% | 4.8%* | 158* | **2.35** |

\* cold rate in the equal-weighted mean is dominated by the 20-token cold-start corpora;
count-weighted cold = **0.28%** (all policies identical at the SSD boundary at 4 GiB).

- **LRU wins the HOT tier** on this traffic; frequency/reuse eviction *hurts* (bursty
  access). Hybrid (+prediction) is marginally better than LRU (22.0% vs 19.9% hot) and
  slightly reduces RAM gather. Policy differences are second-order at 4 GiB WARM.
- **Decision A (is LRU worth testing?): yes — it is the correct baseline; the more
  interesting comparison is WARM size and HOT slot count, not eviction policy.**

### 2.5 Prefetch economics (`prefetch_sweep.csv`, `prefetch_eval.csv`)

Predictor precision (per-layer, 6 predicted): P1 copy-last 29.0%, P2 argmax-successor
**50.1%** (the demo's), P3 transition-mass top-6 41.4%, P4 popularity 31.4%. Coverage of
the next token's set: P3 36.7% best. Exact-set matches: ≤3.9%.

| WARM | window | SSD Δ (LRU vs +prefetch) | pf precision |
|---|---|---|---|
| 256 MB | full 43 | **−31.2%** (worse) | 41.0% |
| 256 MB | 8 layers | −5.4% (worse) | 47.1% |
| 2 GB | 8 layers | −3.9% (worse) | 52.4% |
| 2 GB | 16 layers | −6.6% (worse) | 52.7% |
| 4 GB | any | no-op (already ~0 cold) | — |

**Prefetch into WARM does not pay at the byte level on these traces at any tested WARM
size**: the extra reads (11–60/token × 12.75 MiB) exceed the cold misses saved. Its only
defensible value is **latency hiding** (converting a stall into background I/O), which a
byte-level sim cannot adjudicate — that is exactly what the A/B runtime experiment must
measure (stall time, not bytes). Prefetch precision (~50%) and lead-time constraints
(§2.7) make it a second-order optimization, not the first lever.

### 2.6 SSD access pattern (`ssd_access_analysis.json`)

- 774 preads/token (6 experts × 3 projections × 43 layers), each 4.25 MiB.
- Adjacent-expert coalescing: only **2.36%** of pairs are adjacent IDs → coalescing is
  negligible (the DSEI spans are contiguous per expert but the *selected sets* rarely
  contain neighbors).
- **Sorting the request batch by file offset cuts the seek-cost proxy by 89.4%** →
  group-by-offset batching (the existing QD8 pool already does unordered parallel reads;
  ordering the batch by offset is a free improvement candidate).
- Shard spread: requests hit shards 1–4 (histogram 647k/681k/681k/187k).

### 2.7 Latency model (`latency_model.json`)

T_load = T_IO + T_copy (+ T_scheduler separately):

| batch | cold scatter | warm QD8 |
|---|---|---|
| 1 expert | 12.0 ms | 3.6 ms |
| 6 experts (one layer) | 72.2 ms | 21.4 ms |
| 43 layers (full token) | 3,104 ms | 919 ms |

Required lead time to hide a 6-expert cold load (72 ms) at the 222.5 ms/token measured
decode: ≥1/3 token → only layers ≥ ~14 can be hidden by same-token-start prefetch;
earlier layers need predictions ≥1 token ahead (F1@6 0.31) or acceptance that they stall.
At warm QD8 (21 ms/layer) the requirement drops to ~1/10 token (layers ≥ ~4).

### 2.8 Speculation unions (`union_analysis.csv`; cross-checked against hermes F(k))

| k | union experts | union GiB | F = union/(k·258) | per-token prefetch burden |
|---|---|---|---|---|
| 1 | 258 | 3.2 | 1.000 | 3.21 GiB |
| 8 | **1,203** | 15.0 | 0.583 | 1.87 GiB |
| 16 | 1,722 | 21.4 | 0.417 | 1.34 GiB |
| 24 | 2,119 | 25.8 | 0.335 | 1.07 GiB |

- union(k=8) = 15.0 GiB ≈ 4.7× single token (hermes measured F(8) = 4.52 — agreement
  within ~3%, different aggregation). **Speculation shrinks per-token expert bytes
  1.7–3.0× (exec/tok 0.85→0.34) and gives k-token advance route notice** — but the k=8
  union (15 GiB) still exceeds a 4 GiB WARM; only the k-token *increment* (1.87 GiB/token)
  is the prefetch rate, and it fits a 4 GiB WARM.
- **Decision E (does Re-Spark improve prefetch lead time?): yes structurally — draft
  tokens provide full forward passes k tokens early; but the byte-level union pressure
  means the WARM tier must be ≥ the union increment (≈2 GiB), which is satisfiable.**

### 2.9 Scenario forecast (`scenarios.json`; serial component model — ranges, not claims)

| scenario | SSD hit | SSD GiB/t | stall ms/t | sync+sched ms/t | serial total | tok/s range |
|---|---|---|---|---|---|---|
| conservative | 45.7% | 1.74 | 248 | 398 | 768 | 0.9–1.3 |
| plausible | 62% | 1.22 | 174 | 398 | 694 | 1.0–1.4 |
| aggressive | 85% + overlapped sched | 0.48 | 69 | 40 | 231 | 2.9–4.3 |

Measured resident route-stable decode (222.5 ms → 4.5 tok/s) sits at the top of the
aggressive range, consistent with ~100% hot hits there. **The scheduler/sync term
(398 ms) is 2–3× the compute+stall terms in every scenario — the first systems target.**

## 3. Hypotheses (testable, not established)

1. **H1 — scheduler/sync (310–475 ms/decode) is the binding constraint** above ~1.5 t/s.
   Cheap test: the A/B harness with per-phase timers (DSV4_DEBUG timers exist).
2. **H2 — 4 GiB WARM collapses SSD traffic to ~0 at runtime** (sim says 0.28% cold on
   these traces). Runtime test: measure SSD bytes at 4 GiB cache in arm B.
3. **H3 — HOT slot count, not eviction policy, is the second-order lever**: 6→24
   slots/layer moves 20→57% hot hits (sim), cutting RAM gather ~2.4→1.1 GiB/token.
4. **H4 — prefetch is a latency play, not a bytes play** on these traces (byte-level
   negative at every WARM size). Only runtime stall-time measurement can confirm.
5. **H5 — cross-layer expert prediction is dead** (re-confirmed: overlap 0.022–0.028 on
   2,836 tokens). Do not resurrect.
6. **H6 — route-stable prompts are the exception, not the rule**: the "resident
   route-stable 4.5 t/s" target depends on ~100% hot hits; typical prompts churn ~80% of
   slots per token (sim) and pay the RAM-gather path.

## 4. Recommendations (ordered)

1. **First GPU experiment stays A vs B** (page-cache residency vs explicit LRU RAM
   cache, 4 GiB), frozen E1 env, with **per-phase timing instrumentation** (scheduler,
   sync count, cache hit, promotion count, queue depth, SSD bytes, RAM bytes, CPU time)
   — the Avanza-style dashboard (see `docs/V4_AVANZA_PERFORMANCE_TRANSFER.md`).
2. If A/B confirms H1 (scheduler-bound), the next patch is the **batched transaction**
   path (§6 of the design doc): one read_begin/issue-all/wait per token instead of 43
   per-layer waits, plus offset-sorted read batches (−89% seek-cost proxy, §2.6).
3. Then **HOT slot expansion** (H3) — the arena/GPU geometry change (6→12/24 slots) —
   is the highest-value cache-side change per the working-set curves.
4. Prefetch (H4) and eviction-policy tuning are third-order until the above are measured.
5. Keep WARM at 4 GiB; do not allocate more RAM to the cache on this 32 GiB box.

## 5. Limits

- Simulations are byte/hit models: no timing, no overlap, no Btrfs fragmentation model
  beyond the labeled bandwidth constants, no multi-token batches, no KV/context sharing.
- The 512-token prompts are short relative to 1M context; longer sessions may grow the
  per-layer tail (unique experts/layer already at 237/256).
- Prefetch accounting uses a strict 2-token usefulness window.
- Scenario tok/s are component-model ranges, **not measurements and not promises**.

---

# 6. PHASE 2 — BUILT CPU-SIDE MECHANISMS (this session)

New deterministic tools in `results/v4_expert_cache_static/` (all CPU-only, GPU
guards enforced, no source changes to the fork):

| tool | purpose | status |
|---|---|---|
| `adv_sim.py` | non-displacing windowed prefetch (budget × window sweeps), Re-Spark oracle prefetch, hybrid ablation, per-slice validation, latency-hiding model | run, artifacts below |
| `planner.py` | deterministic batched request planner (orderings A–D, coalescing, plan digest) | run; determinism PASS |
| `sched_sim.py` | TokenPlan state machine (route→resolve→cache→request→promote→commit, ONE barrier) + lease/generation safety tests | **7/7 tests PASS** |
| `harness.py` | GPU-disabled A/B harness: env gate, identity hashes (bounded), full §25 output schema | verified; GPU fields `unavailable` |
| `io_microbench.py` | bounded real-span SSD comparison: buffered vs O_DIRECT vs sorted-coalesced | run (178 MB total) |

## 6.1 Non-displacing prefetch (NDP) — the invariant works

Design: WARM split into RESIDENT pool + PREFETCH pool (budget = pct of WARM);
prefetch inserts go only into the prefetch pool and **never evict a resident page**;
an accessed prefetch graduates to resident. `prefetch_budget_sweep.csv`:

| WARM | budget | window | cold (vs LRU) | SSD MiB/t (vs LRU) | pf precision | wasted MiB/t | net MiB/t |
|---|---|---|---|---|---|---|---|
| 256 MB | 10% | 8 | 73.9% (74.4) | 2,429 (2,448) | 12.3% | 144 | **−125** |
| 512 MB | 20% | 8 | 66.2% (68.4) | 2,176 (2,249) | 53.8% | 67 | +7 |
| 1 GB | 20% | 12 | 52.7% (56.1) | 1,732 (1,847) | 79.5% | 31 | **+84** |
| 1 GB | 50% | 12 | 52.0% (56.1) | 1,712 (1,847) | 94.6% | 4 | **+131** |
| 2 GB | 20% | 12 | 27.5% (29.9) | 905 (982) | 94.3% | 2.3 | **+75** |
| 2 GB | 50% | 12 | 27.5% (29.9) | 905 (982) | 94.3% | 2.3 | +75 |

Reading: the non-displacing invariant converts prefetch from byte-negative (naive,
N3/N4) to byte-positive at WARM ≥ 1 GB. Best configs: **1 GB / 50% / w12 (+131 MiB/t)**
or **2 GB / 20% / w12 (+75 MiB/t, lower waste)**. Precision rises with WARM size
(pool capacity reduces dropped prefetches). At 4 GiB the cold rate is already ~0.3%
and prefetch is a no-op (N5).

## 6.2 Hybrid policy — reverse-engineered (hybrid_ablation.csv)

| term weights | HOT rate (S=6, 4 GiB) |
|---|---|
| recency only (=LRU) | 19.95% |
| frequency only | 12.48% |
| prediction only | 7.01% |
| **0.4·rec + 0.4·freq + 0.2·pred** | **21.99% (best)** |
| 0.6·rec + 0.2·freq + 0.2·pred | 21.01% |

Recency is the dominant term; frequency and prediction are only useful as
**tiebreakers** on near-recency ties. The 2pp hybrid edge is real but small — policy
choice stays second-order vs tier sizing and scheduler work.

## 6.3 Useful-cache accounting

- Immediate reuse (same layer, next token): **42.6%** of accesses; within 2 tokens:
  **53.8%**; within 4: 63.3%; within 8: 73.9%.
- ⇒ a WARM eviction displaces a page that would be reused within 2 tokens ~54% of
  the time — the upper bound on "displaced useful pages", and the reason the NDP
  invariant (never displace for speculation) is the correct admission/eviction split.

## 6.4 Admission vs eviction split (final)

- **Admission** (should this expert enter WARM/HOT?): cold misses are always admitted
  (they were just used); prefetched entries are fetched-but-NOT-admitted — they live
  in the prefetch pool and graduate only on access (§6.1). This is the split that
  makes NDP byte-positive.
- **Eviction** (which resident leaves?): recency-dominant hybrid score (0.4/0.4/0.2)
  among resident entries only; prefetch-pool entries never trigger resident eviction.

## 6.5 Per-slice validation (slice_results.csv)

Policy ranking is **stable across all 8 slices**: hybrid > lru > freq on every
prompt (winner: hybrid 8/8; e.g. code: 21.0% vs 19.2% vs 12.4%; math: 32.6% vs
29.6% vs 13.5%). The trace corpus is not overfit to one workload; results are still
labeled "measured on this trace" pending runtime revalidation.

## 6.6 Re-Spark oracle prefetch (respark_prefetch.csv)

| WARM | k | cold (vs k=0) | SSD MiB/t | precision |
|---|---|---|---|---|
| 512 MB | 8 | 66.1% (68.4) | 2,175 (2,250) | 100% |
| 1 GB | 8 | 51.6% (56.1) | 1,698 (1,847) | 100% |
| 4 GB | 8 | 0.12% (0.21) | 4.1 (7.0) | 100% |

Oracle (100% precision) union prefetch saves ~3–8% SSD at small WARM and ~45% of the
tiny 4 GiB base. Conclusion: Re-Spark's byte value is bounded by WARM absorption;
its real value remains latency hiding + 2.1–2.9× exec reduction (k=8–24).

## 6.7 Planner (planner_results.csv, planner_summary.json)

- 774 cold spans/token (all-cold view); offset ordering (C) cuts the seek proxy
  **87.8%**; bounded coalescing (D) merges only 1.9% of requests (adjacent selected
  experts are rare — N10).
- Determinism: same input → same plan digest; rebuild check PASS.

## 6.8 Scheduler state machine (sched_sim.py)

TokenPlan route→resolve→cache→request→promote→**commit** with ONE barrier. Proven:
(1) all required experts ready before commit; (2) missing dependencies fail closed
with no partial commit; (3) partial requests retry cleanly; (4) **late I/O completion
for an evicted/stale-generation entry is rejected and never resurrects residency**;
(5) eviction refused while leased; (6) cancellation deterministic; (7) generations
monotonic. **7/7 tests PASS (exit 0).**

## 6.9 Latency-hiding model (latency_hiding_model.json)

| model | ms/token |
|---|---|
| serialized (compute + I/O + current sched) | 521.5 |
| partially overlapped (max(compute, I/O) + sched) | 446.1 |
| fully pipelined, current sched | 398.0 |
| fully pipelined, batched-transaction sched (design) | **122.3** |

Component model only; the batched-sched term (40 ms) is a design target, not measured.

## 6.10 Direct-I/O microbenchmark (io_microbench.json; 178 MB total, real spans)

| mode | MB/s (cold, after FADV_DONTNEED) |
|---|---|
| buffered scattered pread | 1,376 |
| O_DIRECT aligned | 892 (slower) |
| sorted-coalesced batch (best-case contiguous sample) | 1,732 (+26%) |

O_DIRECT re-confirmed not beneficial (N9); coalescing helps modestly at cold (N10).
Caveat: the sample was contiguous experts (best case); real sets are 2.4% adjacent.

## 6.11 DSEI lookup throughput

51,600-query benchmark: dict 45 µs/token, flat-array 64 µs/token, batched 39 µs/token.
**Metadata lookup is negligible** (µs) vs the ~400 ms scheduler term — no further
optimization warranted (N14); the existing DSEI map is the right structure.

# 7. DECISION MATRIX (final, §30 of the mega directive)

| Mechanism | Current status | Evidence | Next action |
|---|---|---|---|
| LRU | keep as baseline arm B | 19.9% HOT, correct baseline | use in A/B harness unchanged |
| Hybrid (0.4/0.4/0.2) | **best policy, +2pp** | ablation; winner 8/8 slices | ship as arm B+ after A/B |
| Frequency-aware | rejected for HOT eviction | N1 (12.5% vs 19.9%) | do not resurrect |
| Reuse-distance | rejected for HOT eviction | N2 (13.2%) | use only inside hybrid tiebreak |
| Naive prefetch | rejected (bytes) | N3/N4 (−26..−31%) | never deploy full-token WARM prefetch |
| Layer-window prefetch | viable within NDP | §6.1 (w8/w12 best) | fold into NDP design |
| Non-displacing prefetch | **adopted** (invariant: no resident displacement) | §6.1 net positive ≥1 GB | implement prefetch pool in cache |
| Request coalescing | minimal (1.9%) | planner D; N10 | keep offset sorting (87.8% seek ↓) |
| Batched transaction | designed + state machine proven | sched_sim 7/7 | implement after A/B confirms scheduler-bound |
| Scheduler fusion | designed (43→1) | audit + latency model (398→40 ms term) | highest-leverage code change |
| Re-Spark prefetch | bounded bytes; latency value | §6.6 | defer until pipeline fast |

## Answers to the six questions

1. **Most valuable change before the GPU run:** instrument A/B with the phase-timer
   dashboard (harness.py manifest is ready) — without it the scheduler attribution
   cannot be confirmed and nothing else can be prioritized.
2. **Minimum viable GPU experiment:** A vs B, same frozen env, ≥256 tokens, route
   equality + GEN parity + the §25 manifest fields (harness.py emits the exact schema).
3. **What falsifies the architecture:** if scheduler/sync time is < 30% of decode
   time in arm B (i.e., the 398 ms modeled term is not real), the batching/fusion
   priority is wrong; if 4 GiB WARM does not collapse SSD traffic at runtime, the
   tier model is wrong.
4. **What justifies a larger HOT tier:** runtime RAM-gather (arm B ram_bytes) > 1.5
   GiB/token — the sim predicts 2.42 GiB/token at S=6; 12–24 slots should cut it
   toward 1.1 GiB/token.
5. **What justifies predictive prefetch:** runtime stall-time reduction from NDP at
   1–2 GiB WARM (stall ms/token measurable in the dashboard) — only a real timing
   measurement can promote prefetch from latency-hiding hypothesis to feature.
6. **What justifies Re-Spark:** accepted-token rate with a draft model showing the
   expert-union exec reduction (k=8: 0.57 exec/token) AND the prefetch lead-time gain
   measured as reduced cold stalls; until then Re-Spark remains a post-pipeline item.
