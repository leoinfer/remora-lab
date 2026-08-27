# V4 REMORA Expert-Cache Design

**DeepSeek-V4-Flash Q8_K_XL (0731 UD) → SSD → RAM → VRAM expert working-set hierarchy**

**Status:** STATIC DESIGN — produced by static inspection only. No GPU, no training, no
long inference, no repository mutation, no active-job interference (the training job owns
the GPU). All claims are labeled `MEASURED` (prior research artifacts), `DERIVED`
(computation from measured geometry), or `HYPOTHESIS` (to be verified at runtime).

**Date:** session of the static exploration. **Home:** this file's directory.

**Evidence base:**
- Model artifacts: `[local path omitted] V4 Flash 0731 UD-Q8_K_XL/` (5 GGUF shards + dspark draft)
- Expert index: `traces/expert_index_0731_v2.bin` (DSEI v2, 33,024 entries)
- Runtime fork: `[local path omitted]` branch `research/hermes-v4-feasibility` @ `f3b8aba5d`
- Prior research: `docs/` + `results/` + `results/hermes-v4/` under this directory
- Traces: `traces/dsv4-first-trace.jsonl`, `traces/dsv4-16token-trace.jsonl`,
  `results/hermes-v4/traces/*.tr` (6 prompts × 256 tokens: tech, code, math, creative, convo, long-context)

---

## 1. Current V4 architecture (verified from artifacts)

### 1.1 Model identity

| property | value | evidence |
|---|---|---|
| architecture | `deepseek4` (GGUF v3) | shard-1 metadata |
| layers | 43 | `deepseek4.block_count` |
| n_embd / n_ff | 4096 / 2048 | metadata |
| experts / used | 256 / 6 | `expert_count`, `expert_used_count` |
| gating | SQRT_SOFTPLUS (enum 4) | `expert_gating_func` |
| expert weights | norm=True, scale=1.5 | metadata |
| shared experts | 1 per layer (`ffn_*_shexp`, BF16) | tensor inventory |
| hash layers | 3 (`deepseek4.hash_layer_count`) | metadata |
| attention | MQA (1 KV head, kv 512), q_lora 1024, attn indexer top_k 512, output groups 8, compress ratios | metadata |
| context | 1,048,576 (YaRN 16×, orig 64k) | metadata |
| vocab | 129,280 | tensor dims |
| expert tensor type | **MXFP4 (ggml type 39), 17/32 B/w** (4,456,448 B per 8,388,608 weights) | byte-per-element from shard-2 offsets |
| core tensor type | **BF16 (ggml type 30)** | byte-per-element |
| routing table | `ffn_gate_tid2eid` I32 [6, 129280] (learned token→expert map, hash layers only) | tensor inventory |
| sizes | experts **137.06 GiB** (33,024 × 4,456,448 B), core **13.71 GiB**, total **161 GB / 150.77 GiB** | index + inventory |

The label "Q8_K_XL" applies to the core/dense tensors; **the routed experts are MXFP4** in the
0731 UD file (verified by exact byte math, not assumed). The DSEI v2 index `quant_type=39`
(MXFP4) is consistent with the file.

### 1.2 Hardware (from prior research artifacts)

RX 9060 XT 16 GiB (device-local heap **15.92 GiB**), Ryzen 7 3700X, 32 GiB DDR4-3200,
Btrfs NVMe. **Consequence: the full 150.77 GiB model cannot be VRAM-resident, and even
RAM-resident (32 GiB) is impossible — the model only runs in a tiered/streaming mode on
this box.** The vanilla "load everything" path degrades to mmap + CPU (MEASURED 3.0 s/token
ngl=0).

### 1.3 Per-layer expert tensors

Each layer `blk.N` holds, for the MoE block:

- `ffn_gate_exps.weight` — MXFP4 [4096, 2048, 256], 1,140,850,688 B, contiguous per-expert slices of 4,456,448 B
- `ffn_up_exps.weight` — MXFP4 [4096, 2048, 256], same
- `ffn_down_exps.weight` — MXFP4 [2048, 4096, 256], same
- `ffn_gate_inp.weight` — BF16 [4096, 256] router projection
- `ffn_gate_shexp / ffn_up_shexp / ffn_down_shexp` — BF16 shared expert (always executed)
- `ffn_gate_tid2eid.weight` — I32 [6, 129280] (layers 0..2 only)

Expert slices are **contiguous and uniform** within each 3D tensor (verified: every DSEI
span is exactly 4,456,448 B; no padding between slices inside a tensor — GGUF alignment
applies at tensor boundaries only). This makes per-expert addressing exact and cheap.

---

## 2. Exact routing path (token → expert execution)

All names are `cb(name, il)` labels; the graph is built per decode in
`llama_model_deepseek4::graph::graph` (`src/models/deepseek4.cpp`, from ~line 1698) and the
MoE body in `llm_graph_context::build_moe_ffn` (`src/llama-graph.cpp`, 1773–2160).

```
token
 ↓  token_embd → hc_init (hyper-connection repeat ×4) → per layer il:
 ↓
[hash layers il < 3]  selected_experts = ggml_get_rows(ffn_gate_tid2eid, tokens)   ← NO hidden-state router
 ↓                    (deterministic per token id; known at token intake)
[layers 3..42]
 ffn_norm
 ffn_moe_logits      = mul_mat(ffn_gate_inp [4096,256], cur)          [256, n_tok]
 ffn_moe_probs       = sqrt(softplus(logits))                          (gating=4)
 ffn_moe_probs_biased= probs + exp_probs_b (when present)
 ffn_moe_argsort     = argsort_top_k(selection_probs, 6)               [6, n_tok]
 ffn_moe_topk        ← ★ INTERCEPTION POINT (eval callback on this tensor)
 ffn_moe_weights     = get_rows(probs, topk) → normalize → ×1.5
 ffn_moe_gate/up     = mul_mat_id(up_exps, cur, exec_ids)   exec_ids = selected_experts (vanilla)
                       or compact 6-slot arena ids [0..5] (compact path)
 ffn_moe_swiglu      = swiglu_split(gate, up), clamp ±10
 ffn_moe_down        = mul_mat_id(down_exps, act, exec_ids)
 ffn_moe_weighted    = down × weights
 ffn_moe_out         = sum over 6 experts
 ffn_shexp           = shared expert (always resident)
 ffn_out             = moe_out + shexp
 ↓  hc_post → next layer
 ↓
hc_head → result_norm → output (mul_mat with output.weight BF16 [4096, 129280])
```

**Key structural facts:**

1. **The router is a tiny, dense op**: 256 logits from a [4096,256] projection. It costs
   ~µs on GPU and produces `ffn_moe_topk` **before any expert weight is touched**. The
   expert IDs for a token exist well before the expert compute — the entire prefetch
   window is the rest of the layer's attention + the previous layers.
2. **Hash layers (0–2) have no router at all**: `ffn_gate_tid2eid` is a pure table lookup
   keyed by token id. Their expert sets are known the instant a token is sampled — a
   deterministic prefetch hook for 3/43 layers (7%).
3. **MUL_MAT_ID indexes expert slices by column id** in both the vanilla (ids = router
   output, 256 slices) and compact (ids = slot 0..5) paths. The backend reads exactly the
   selected expert slices. Nothing requires all 256 experts to be addressable at once —
   only that the `ids` values are in range of `ne[2]` of the weight tensor.
4. **The router's learned behavior is untouched by the compact path**: the same
   `ffn_gate_inp` + argsort computes the same top-6. The interception layer only changes
   *where the expert bytes come from* (residency), never *which experts are chosen*.
   Route-forcing (`DSV4_FORCE_ROUTES`) exists as a diagnostic to isolate route divergence
   from expert-fidelity divergence — the opposite direction is what REMORA needs
   (identical routes by construction).

---

## 3. Exact relevant source files/functions (fork `[home-relative path omitted]` @ f3b8aba5d)

### Interception & graph
| file | function / region | role |
|---|---|---|
| `src/models/deepseek4.cpp` | `dsv4_compact_arena_callback` (line 1132) | eval callback; reads topk ids after compute+sync; populates arena/staging; records `arena.last_ids[43][6]` per layer |
| `src/models/deepseek4.cpp` | `llama_model_deepseek4::graph::graph` (~1698) | graph build; arena wiring; `build_moe_ffn(...compact...)` with 6-slot tensors + exec_ids view; eval callback registration (line 2071) |
| `src/llama-graph.cpp` | `build_moe_ffn` (1773/1819) | router: logits → probs → argsort → `ffn_moe_topk` (1939) → weights (1950) → MMID gate/up (2012, 2044) → down (2133) → weighted sum |
| `ggml/src/ggml-backend.cpp` | eval-callback path (1820–1895); `ggml_backend_sched_set_eval_callback` (2140) | splits compute graph per `ffn_moe_topk`, computes range, **synchronizes**, calls callback(ask=false), resumes. This per-layer sync is the current interception cost |

### Cache tiers (existing fork code)
| file | component | role |
|---|---|---|
| `src/dsv4_expert_cache.h/.cpp` | `dsv4_expert_cache` | **RAM tier**: DSEI index load; LRU slots with `in_use` pinning; 8-thread QD8 batched `pread` pool; `get_expert_issue/read_wait` batch API; `prefetch_expert` (defined, **not yet called anywhere**); stats (hits/misses/ssd_bytes/ram_bytes); `dsv4_set_streaming_callback/on_decode_start/done` are **empty stubs** |
| `src/dsv4_compact_arena.h` | `dsv4_compact_arena` | 6-slot/layer CPU compact tensors (type follows model: MXFP4 or skeleton); `last_ids[43][6]` route record; per-decode stats (cache/populate/upload µs) |
| `src/dsv4_compact_gpu.h/.cpp` | `dsv4_compact_gpu_manager` | **VRAM tier**: per-layer 6-slot GPU tensors (MXFP4: 76.5 MiB/layer, 3.3 GiB/43); transient GTT staging (pool ~3 GiB, epoch reset per decode); `dsv4_vk_defer_copy` batched into next compute; associative `remap_experts` (identity→slot, avoids rank-churn re-uploads); ReBAR direct-write (MEASURED 0.34 GB/s — rejected) |
| `src/dsv4_skeleton_store.h` | `dsv4_skeleton_store` | quantize MXFP4 → Q2_K/Q3_K/Q4_K on first fetch, 512-entry LRU, stats (stores/evicts/hits) |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | `dsv4_vk_staging_alloc` (16760), `dsv4_vk_defer_copy` (16788), `dsv4_vk_staging_epoch_reset` (16863) | host-visible staging pool + deferred copy queue |

### Loader / residency (vanilla)
| file | function | role |
|---|---|---|
| `src/llama-model-loader.cpp` | `init_mappings` (1329) | mmap of all split files (with optional mlock); `use_mmap` from `LLAMA_LOAD_MODE_MMAP` |
| `src/llama-model-loader.cpp` | tensor load loop (~1532) | mmap path: tensors either aliased into an mmap buffer or `ggml_backend_tensor_set`-copied to the offload backend; non-mmap path: file reads, async uploads to GPU for non-host buffers. **No lazy expert loading exists** — every tensor is materialized at load. |
| `include/llama.h` | `tensor_buft_overrides` (used by `tools/dsv4_interactive.cpp`) | regex → buffer-type override. The demo pins `blk\.[0-9]+\.ffn_(gate|down|up)_exps\.weight` to CPU so the 137 GiB expert bank never enters VRAM |

### Tools
| file | purpose |
|---|---|
| `tools/dsv4_read_replay.cpp` | I/O A/B/C: replays route traces through the cache in sync / QD8 / WILLNEED / O_DIRECT / repack modes; `/proc/self/io` accounting; page-cache drop per token |
| `tools/dsv4_interactive.cpp` | end-to-end compact demo (VK + CPU argsort); per-(layer,expert) next-token transition table `g_trans` (predictor sketch, not wired); multi-prompt |
| `tools/s5_real_model_test.cpp` | E1 correctness harness (GEN compare, numdump, hashes) |
| `tools/dsv4_attn_scale.cpp`, `tools/e1a_golden_oracle.cpp` | attention slope / skeleton oracle diagnostics |

---

## 4. GGUF tensor/shard organization

- **Shard 1** (`[model payload omitted]`, 5.2 MB): metadata only (70 kv, 0 tensors).
- **Shards 2–5** (49.2 / 49.7 / 49.5 / 13.5 GB): 384 + 384 + 384 + 176 = **1328 tensors**;
  each shard is a self-contained GGUF v3 with `split.no`, `split.tensors.count`, `split.count`.
- Tensor data is 32-byte aligned at tensor boundaries; within a 3D expert tensor the
  256 expert slices are contiguous and identical-sized (4,456,448 B).
- The DSEI v2 index (`traces/expert_index_0731_v2.bin`, magic `DSEI`) stores per-expert
  spans: `layer(4) expert(4) tensor_type(1) file_idx(1) file_offset(8) byte_size(8)
  quant_type(1)` = 27 B × 33,024. It was built from the actual 0731 files and is
  byte-consistent with them (verified by comparing span math to shard-2 tensor offsets).
- **Per-expert addressability is already solved**: `(layer, expert, tensor) →
  (fd, offset, 4,456,448 B MXFP4)` with zero parse cost at runtime. The index is a static
  artifact, not a runtime scan.
- Shard boundaries: experts span shards 2–5 (`file_idx` 1..4 in the index; e.g. layer 0
  lives entirely in shard 2). The cache opens all 5 parts with `O_RDONLY` and picks `fd`
  per span.

---

## 5. Current residency / loading behavior

1. **Vanilla path (untouched upstream semantics)**: whole model mmap'd; per
   `n_gpu_layers`, tensors are copied into VRAM buffers at load; experts are **fully
   resident** wherever the layer lands (VRAM or CPU-mmap). No lazy loading, no eviction,
   no per-expert residency tracking. On this box this path is CPU-only for experts and
   measures ~3.0 s/token (MEASURED, ngl=0) — the 150.77 GiB cannot be GPU-resident.
2. **Fork's compact path (exists, partially validated)**:
   - experts pinned to CPU via `tensor_buft_overrides` (their bytes stay in mmap);
   - dense core offloaded (ngl=43, ~13.71 GiB fits the 15.92 GiB heap);
   - per decode, the eval callback intercepts each layer's `ffn_moe_topk`, reads the 6
     real expert IDs, and populates a 6-slot compact tensor (RAM arena and/or VK staging)
     from either (a) the mmap page cache (`DSV4_CACHE_OFF=1`) or (b) the `dsv4_expert_cache`
     RAM cache (QD8 batched reads);
   - MUL_MAT_ID then executes only the 6 resident slot experts.
   - **Current VRAM residency is per-decode transient**: slots are refilled per token;
     the *cache* (RAM) is the only persistent tier. VRAM does not yet hold a persistent
     hot expert set across tokens (the remap keeps slots warm *within* a decode).
3. **Measured state of the compact path** (prior artifacts): transport certified
   (E1 Phase-0, verifier 214/214; control rms 0.041); E1 skeleton matrix complete
   (Q3_K: 9/9 greedy-identical through 24 layers; Q2 only ≤4 layers); resident
   route-stable decode **222.5 ms/token** (copy 36.3 / cpu 19.3 / vk 47.5 / sync 87.8 /
   cb 0.5 / build 38.5 ms); perfect-hit steady state ~0.53 s/decode with the scheduler
   re-reserve included; MMID standalone 0.129 ms/layer (IQ3). The canonical-MXFP4 compact-VK
   path had a crash fixed in source but not yet rebuilt/validated at the time of the last
   artifacts.
4. **Known overheads (MEASURED, the attack surface for REMORA)**: scheduler loop +
   per-layer sync ≈ 310–475 ms/decode — larger than the entire compute budget; this must
   be overlapped/eliminated before hit-rate improvements pay out.

---

## 6. Proposed SSD → RAM → VRAM hierarchy

```
                 DeepSeek-V4-Flash
                        │
                     Router (UNTOUCHED: gate_inp → sqrt-softplus → argsort top-6)
                        │
                 expert IDs  (hash layers: table lookup, known earlier)
                        │
          ┌─────────────┴─────────────┐
          ↓                           ↓
      HOT VRAM                    WARM RAM
      6–N slots/layer             dsv4_expert_cache
      (compact tensors)           (LRU, QD8 pool)
          │                           │
          └─────────────┬─────────────┘
                        │ miss
                        ↓
                     COLD SSD
                  (DSEI-indexed preads)
                        │
                    prefetch
                        │
                        ↓
                     execute (MUL_MAT_ID on 6 slots)
```

### 6.1 Tier definitions (conceptual)

| tier | medium | content | identity |
|---|---|---|---|
| COLD | SSD (Btrfs NVMe) | full 137.06 GiB expert bank, canonical MXFP4 | `(layer, expert)` → 3 spans via DSEI v2 |
| WARM | RAM (32 GiB, budget e.g. 8–20 GiB) | recently used experts, **canonical bytes** (copy of MXFP4 span or page-cache-backed mmap) | cache key `(layer<<16)|expert` (already the fork's `make_key`) |
| HOT | VRAM (16 GiB heap; budget 2–8 GiB) | the per-token working set in compact 6-slot tensors (canonical MXFP4 or skeleton Q3/Q4) | slot identity (associative remap already implemented) |

### 6.2 The two competing WARM designs

- **(W1) Explicit cache** (existing `dsv4_expert_cache`): pread into heap buffers, LRU
  eviction with `in_use` pinning. Deterministic accounting (`ssd_bytes`, `ram_bytes`),
  works with `DSV4_FADV_DONTNEED`.
- **(W2) Page-cache-backed** (`DSV4_CACHE_OFF=1`): the mmap page cache *is* the warm tier.
  Zero-copy (the arena `set_expert` memcpys straight from `wg->data`); residency managed by
  the kernel. `posix_fadvise` WILLNEED/DONTNEED are the prefetch/evict primitives. Prior
  I/O A/B/C found neither WILLNEED nor O_DIRECT fixed cold latency (Btrfs extent
  fragmentation), so W2 is the *simplest* baseline but not a silver bullet.

REMORA's WARM tier should keep W1 as the primary path (explicit, measurable) with W2 as a
control arm — exactly the existing A/B/C harness.

### 6.3 Promotion / demotion / eviction (design, not yet implemented)

- **Promotion SSD→RAM**: batch (QD8) reads of all 6 experts of a layer at `read_begin`,
  or `prefetch_expert` for lookahead (exists, unwired).
- **Promotion RAM→VRAM**: populate staging (NT copies, `dsv4_populate_pool`) → deferred
  `dsv4_vk_defer_copy` batched into next compute — exists.
- **Demotion VRAM→RAM**: on slot eviction in `remap_experts`, the bytes are **already in
  RAM** (they came from there). VRAM→RAM demotion is therefore free/absent by design;
  the only lossy event is RAM eviction (LRU).
- **Eviction policy**: RAM = LRU + `in_use` pinning (exists); VRAM = per-layer 6-slot
  least-recently-used-by-age remap (exists). REMORA upgrade: per-expert statistics-driven
  scoring (§7) + cross-layer budget arbitration (§8.5).

### 6.4 What does NOT change (the REMORA constraint)

- The router, top-k selection, weights, gating, shared-expert path, attention — all
  byte-identical to the canonical graph. REMORA manages residency only.

---

## 7. Cache-policy options

Prior measured locality (17-token JSONL trace + 6×256-token union traces; `HYPOTHESIS`
until re-verified against the current implementation, but with strong prior support):

| signal | value | verdict |
|---|---|---|
| same-layer t→t+1 expert overlap | 2.2/6 experts (Jaccard 0.254, F1@6 0.368) | **dominant axis** |
| decay | −15% (t+2), −27% (t+3), −41% (t+7) | short-horizon signal |
| cross-layer intra-token (ΔL) | F1 ≈ 0.02 (noise) | **negative result — do not build on it** |
| effective experts/layer | 11.8–64.2 (mean 22.8) | popularity is concentrated |
| top-6% experts/layer | cover 31–64% of activations | long-tail exists but is small |
| expert-union over speculative blocks | F(k): k=8 → 4.52×, k=16 → 6.70× (exec/tok 0.57→0.42) | speculation compresses expert traffic 2.1–2.9× |

### Policy candidates (ranked for V4)

1. **LRU (per layer)**: current behavior; simple; captures the t→t+1 overlap for free
   (2.2/6 hit potential) but ignores popularity and load cost. **Baseline (Experiment B).**
2. **LFU/popularity (per layer)**: the effective-expert measurement (mean 22.8/layer)
   suggests a per-layer hot set of ~24–64 experts could serve 31–64%+ of activations at
   ~300–800 MiB/layer (MXFP4) — i.e., a 5–15 GiB RAM hot set across 43 layers. Strong
   synergy with the observed concentration; cheap to implement as per-layer counters.
3. **Temporal-locality / transition-aware**: per-(layer, expert) next-expert transition
   counts (the `g_trans` sketch in `dsv4_interactive.cpp`). F1 0.368 is modest but real;
   useful as a *prefetch ranker* (predict which experts to issue reads for next token)
   rather than an eviction policy.
4. **Predictive**: route-pattern predictor (hash-layer table + router logits history).
   The eventual predictor can be tiny (per-layer linear/table model; see §8). Must not
   train on GPU — offline trace learning only.
5. **Hybrid score** (recommended target for REMORA):

```
score(expert e, layer L) =
    α · short-term reuse prob   (t→t+1 transition model / last-use recency)
  + β · long-term frequency     (per-layer popularity)
  + γ · load-cost factor        (1/size; uniform here, so ≈ constant — all experts are
                                 12.75 MiB; but fragment-aware: Btrfs extent count per span)
  + δ · predicted-next-use      (predictor confidence, §8)
```

Because **every expert is exactly 12.75 MiB**, the size term is constant; the
differentiators are recency, frequency, and prediction confidence. `γ` matters only if a
future repacked/layout-variant file breaks uniformity.

**Compatibility verdict:** LRU is the correct *baseline*; the measured concentration
argues for LFU/popularity as the *second* policy; transition-aware prefetch is the
cheapest high-value *third* step. Cross-layer prediction is falsified by prior data and
should not be implemented.

---

## 8. Predictive prefetch design

### 8.1 What is known and when

| moment | knowledge available |
|---|---|
| token sampled (t) | hash-layer (0–2) expert sets of token t+1 **exactly** (table lookup); token t's full route once its forward pass completes |
| during layer il of token t | layers 0..il-1 routes of token t; router logits of token t's layer il (before expert exec) |
| end of token t | full 43×6 route of token t (this is exactly what `arena.last_ids[43][6]` already records; `have_last_ids` set at il=42) |

### 8.2 The prefetch loop (design)

```
current routing/state (token t complete, or hash-layer table for t+1)
 ↓
predict likely next experts per layer:
   P_t+1(L) = top-k of { transition(L, e_prev) ∪ popularity(L) }      [k ≈ 6–12]
   hash layers: exact via tid2eid lookup (no prediction needed)
 ↓
prefetch SSD→RAM (async, QD8 worker pool — exists as prefetch_expert, unwired)
 ↓
expert becomes WARM before needed
 ↓
next token's callback promotes WARM→HOT (staging + deferred copy)
 ↓
route executes (canonical)
```

### 8.3 Design answers (static)

- **Earliest prediction point**: hash layers — at token intake (zero lookahead cost).
  Dense layers — as soon as layer 42 of token t completes (the callback already sits
  exactly there; `dsv4_on_decode_done()` is the stub to hook).
- **Lookahead**: 1 token minimum (full previous route), up to k tokens with speculative
  decoding (§13). Prior F1@6 = 0.368 at Δt=1; useful prefetch precision should be judged
  by *bytes saved* not F1.
- **How many experts to prefetch**: 6–12 per layer (the 6 predicted + 6 alternates).
  At 12.75 MiB each, 12 × 43 × 12.75 MiB = 6.6 GiB per full lookahead — **too much for a
  whole-token lookahead in RAM budget**. Prefetch must be *layer-windowed*: only the next
  ~8–16 layers ahead (1.0–2.6 GiB), not the full next token. The scheduler executes layers
  in order; a 8–16 layer window at QD8 ≈ 24–48 sequential reads (4.25 MiB each) is the
  natural batch size.
- **Cancellation**: incorrect prefetches are harmless at the RAM tier (LRU evicts them;
  `in_use` pins protect consumed ones; `DSV4_FADV_DONTNEED` drops page-cache copies).
  VRAM pollution is impossible by construction (slots are per-token transient).
- **RAM reservation for prefetch**: a dedicated prefetch headroom (e.g. 1–2 GiB above the
  LRU budget) prevents prefetch from evicting the live working set; or unify: prefetched
  entries enter LRU with a low "touched" age so they are evicted first if unused.
- **Overlap with current expert execution**: yes, structurally — the QD8 worker pool
  reads into RAM while the GPU executes the current layer's MMIDs; the only serialization
  today is `read_wait()` at each layer's callback (batch-bound), and the scheduler sync
  per layer (~87.8 ms/decode measured — the true first target).

### 8.4 Predictor shape (tiny, no training run)

- Per-layer transition table: 256×256 counts per layer is 65k entries × 43 layers × 2 B ≈
  5.6 MB — fits trivially; learned offline from traces (existing `.tr`/`.jsonl` corpora),
  updated online with the already-recorded routes.
- Hash layers: pure lookup, zero learning.
- Optional: condition on router logits *shape* (top-2 margin) — a scalar feature; do not
  train anything larger.

### 8.5 Cache pollution guard

Prefetch only when (a) RAM headroom exists, (b) predictor confidence ≥ threshold, (c) the
layer is within the next window. Never prefetch into VRAM (VRAM = only actually-executed
experts, enforced by the callback).

---

## 9. Memory-bandwidth analysis (DERIVED from MEASURED constants)

### 9.1 Per-token expert traffic (canonical MXFP4)

| quantity | value |
|---|---|
| expert executions/token | 258 (6 × 43) |
| bytes per execution | 12.75 MiB (3 × 4,456,448 B) |
| full-miss bytes/token | **3.29 GiB** |
| unique expert bytes/token (route-stable, measured 45.7%-hit run) | churn 140 experts ≈ 1.79 GiB; SSD 5.23 GiB/token (incl. page-cache misses) |

### 9.2 Measured stage bandwidths (this machine)

| stage | bandwidth | class |
|---|---|---|
| SSD warm QD8 | 7.20 GiB/s | MEASURED |
| SSD cold scattered | 1.22 GiB/s (Btrfs 128 KiB encoded extents → ~33 reads per 4.25 MiB span, 2× read amplification) | MEASURED |
| SSD nominal (large seq) | ~2.9 GB/s | MEASURED |
| DRAM gather (cache→staging) | 6.6–7.0 GB/s | MEASURED |
| H2D (GTT staging) | 6.4 GB/s cap | MEASURED |
| ReBAR direct CPU→VRAM write | 0.34 GB/s | MEASURED (rejected) |
| VRAM read (MMID) | 0.129 ms/layer IQ3 (≈ 12.75 MiB × 6 / layer) | MEASURED |

### 9.3 Required hit rates (backwards solve from the prior equations doc)

To sustain R tok/s with per-token budget B/R:

| R | SSD budget | required SSD-hit rate |
|---|---|---|
| 5 | 553 MiB/tok | 83.2% |
| 10 | 277 MiB/tok | 91.6% |
| 20 | 138 MiB/tok | 95.8% |
| 30 | 92 MiB/tok | 97.2% |

(These are the *storage* constraints only; the measured scheduler+sync overhead of
~310–475 ms/decode dominates until fixed — see §9.4.)

### 9.4 Where the bottleneck moves (policy ladder)

| config | bottleneck analysis |
|---|---|
| **No cache** (vanilla mmap CPU) | compute: CPU MMID ~10 s/token-era (MEASURED); storage adds 1.5–2.5 s. Not viable. |
| **No cache** (compact, page-cache WARM) | cold SSD scatter: 1.22 GiB/s → 2.7 s/token of pure I/O at full miss (MEASURED 1906 ms/token) |
| **LRU RAM cache** (4–8 GiB) | 45.7% hit measured → SSD ≈ 5.23 GiB/token → storage-bound ~1.4 t/s even with perfect overlap; plus current serial overheads |
| **Large HOT tier** (VRAM-resident popular set) | moves the churn boundary to RAM (6.6–7.0 GB/s gather): at 45.7% RAM-hit the per-token gather ≈ 268 ms → ~3.7 t/s ceiling without other fixes |
| **RAM-backed WARM + prefetch** | turns scattered cold reads into batched QD8 (7.2 GiB/s warm ceiling): 92–277 MiB/tok needed at 10–30 t/s → requires 91.6–97.2% SSD hit |
| **Predictive prefetch** | the only path that *raises* the hit rate itself; F1@6 0.368 baseline suggests ~2.2/6 experts prefetchable per layer → upper bound of added hit ≈ +30%+ (HYPOTHESIS) |

**Static conclusion**: hit rate (not raw bandwidth) is the binding constraint above
~2 t/s; above ~5 t/s the scheduler/sync overhead is the binding constraint. Both must be
attacked; the fork's measured component numbers support this ordering.

### 9.5 Expert-size uniformity note

All experts are exactly 12.75 MiB (uniform MXFP4 slices). No expert is disproportionately
expensive by size; *fragmentation* (Btrfs extent count per span) can make individual
experts disproportionately slow to read — a candidate `γ` term if a future repack doesn't
fix it.

---

## 10. GGUF-specific questions and required changes

| question | answer |
|---|---|
| Q8_K_XL special constraints? | The experts are **MXFP4** (17/32 B/w), not Q8_K. MXFP4 row size = 32-aligned blocks; per-expert slices are exact multiples of block size (4,456,448 B / 17 B per 32 weights = integral). No sub-block spanning. |
| Quantization block layout | block_mxfp4 (32 weights + scale), no cross-expert blocks → slices are self-contained, cacheable, and independently dequantizable/requantizable (the skeleton store already does row-by-row conversion). |
| Tensor alignment | GGUF 32-byte alignment at tensor boundaries; expert slices inside a 3D tensor are contiguous. DSEI spans are exact (verified). |
| Shard boundaries | experts live in shards 2–5; spans never cross shard files (file_idx per span). |
| Can individual experts be independently mmap'd? | Not via the current loader (mmap is whole-file); but the fork reads spans directly with `pread` (O_RDONLY fds per shard) — functionally equivalent. A future `MAP_POPULATE`-style partial mapping per expert is unnecessary. |
| Can a tensor be promoted without unrelated tensors? | Yes — the cache already loads exactly one expert's 3 spans; the graph never touches the 3D tensor in the compact path (buft override keeps it out of VRAM). |
| Is expert-level caching practical with current GGUF abstraction? | Yes — the DSEI index is the required extra abstraction (27 B/expert, precomputed offline). GGUF itself needs no format change. |

### Required code changes (future, not implemented here)

1. **Scheduler overlap** (biggest lever): replace per-layer `sync` in the eval-callback
   path with (a) async topk readback via stream/event, (b) batched staging population, or
   (c) moving the callback to a compute-stream-side hook. Target: the ~310–475 ms
   sched+sync per decode.
2. **Wire `prefetch_expert`** into `dsv4_on_decode_done()` (stub) with a layer-windowed
   prefetch set (§8.3).
3. **VRAM-tier persistence**: decide whether HOT becomes a persistent cross-token
   per-layer slot cache (budget-arbitrated) or stays transient. The 6-slot transient model
   is correct for correctness; persistence is an optimization with the associative remap
   already present as the seed.
4. **Policy upgrade**: per-layer LRU → hybrid score (§7), with per-expert stats
   (`last_use`, `freq`, `reuse_interval`, `predicted_next`) — the stats struct of §11.
5. **Btrfs/repack**: consider a repacked contiguous expert file (the replay tool already
   references a `repack_init` idea; not in the current header — validate before relying on
   it) to convert 33-read scatter into sequential reads.
6. **MXFP4 MMID on Vulkan**: rebuild + validate the canonical-MXFP4 compact-VK path
   (E1 ran skeletons; the MXFP4 crash fix is in source, unbuilt at last artifact).

---

## 11. Expert working-set statistics structure (future, static design)

Per expert (11,008 identities) — a compact row, ~64–96 B, total ~1 MB:

```text
expert ID        u16        (0..255)
layer            u16        (0..42)
size_bytes       u32        (uniform 13,370,944 today; keep for generality)
current tier     u8         (COLD/WARM/HOT)
last_use         u64        (decode counter)
recent_freq      u16        (windowed count, e.g. last 256 tokens)
reuse_interval   u16        (rolling mean token gap between uses)
load_latency_us  u32        (EMA of span read time; fragmentation-aware)
source_tier      u8         (where last loaded from)
predicted_next   i16        (predictor score / next-use distance)
pin              u8         (in_use equivalent for policy exemption)
```

Storage: one flat file/array (memory-mapped), updated by the callback (already running
per layer) and the cache (already counting). No GPU work required to maintain it — all
events are CPU-side observations of the existing data path. A trace-based *offline*
simulator (existing `scripts/hermes_sim.py`, `scripts/cache_oracle.py`) can validate
policies before any runtime change.

---

## 12. Integration with REMORA context memory (design only)

```text
MODEL MEMORY                         CONTEXT MEMORY
SSD → RAM → VRAM                     COLD → WARM → HOT
     EXPERT CACHE                         CONTEXT CACHE
     (this design)                        (IDEA40 / REMORA context hierarchy)
```

Questions and proposed answers:

| question | proposal |
|---|---|
| Shared VRAM budget? | Yes — one allocator. Experts = weights (evictable, reloadable), context = KV/state (evictable, lossy if dropped). Both compete for the same 15.92 GiB heap minus the dense core (13.71 GiB) — i.e., the *residual* ~2 GiB plus GTT tricks. The **dense core is fixed priority**; experts and context share the residual. |
| Do model experts get hard priority? | Experts: yes for correctness-critical slots (an expert miss costs latency, not fidelity — reload is exact). Context HOT: hard floor for the *current* window (recent KV is not reloadable losslessly). Proposal: experts win the shared pool *beyond* the context floor. |
| Shrink context HOT on expert-miss spikes? | Yes, within bounds: expert misses are observable (cache stats) and predictable (routing); context compression is lossy — only demote context segments below the floor when expert demand is forecast to exceed budget (moving-setpoint controller from the REMORA master prompt, line 295/302). |
| Retrieval confidence → expert reservation? | Possible: high-confidence retrieval (hash layers, speculatively verified routes) allows *smaller* expert reservation (fewer alternates), freeing VRAM for context. Low confidence → larger expert prefetch window. |
| Compute-value → both caches? | Yes: the same value signal (accepted-token probability, route entropy) can rank both expert prefetch and context retention. Unified "value = expected compute saved per byte" scoring. |

**Future joint allocator** (no implementation now): one budget ledger with three classes —
dense core (immovable), expert working set (movable, exact-reloadable), context working
set (movable, lossy-demotable) — arbitrated per decode by a controller consuming:
route statistics (this design), retrieval confidence, acceptance statistics (speculative
path), and the hardware thermal/bandwidth setpoint. The REMORA master prompt's
"moving computational-maintenance setpoint" and "reserve mobilization" rules map directly
onto this ledger.

---

## 13. Re-Spark / speculative-decoding compatibility (design only)

Current assets: DSpark speculative machinery (draft model `[model payload omitted]`,
10.9 GB Q8_0; spec-verify path in the fork) and the measured **expert-union curves**:

| block k | expert-union F(k) | execs/block | exec/token |
|---|---|---|---|
| 1 | 1.00 | 258 | 1.00 |
| 8 | 4.52 | 1166 | 0.57 |
| 16 | 6.70 | 1728 | 0.42 |
| 24 | 8.23 | 2123 | 0.34 |

Proposed future flow:

```text
current token
 ↓
V4 routing (canonical)
 ↓
expert-cache prediction (§8)
 ↓
Re-Spark draft predicts next k tokens (draft model forward, cheap)
 ↓
draft routes → EXACT hash-layer expert sets (tid2eid) for t+1..t+k
        → predicted dense-layer sets via transition model
 ↓
prefetch the k-token expert union (F(k)×12.75 MiB: k=8 → 57.7 GiB — NOT
all at once; windowed: only the next 8–16 layers, ~1–2.6 GiB)
 ↓
verify speculative tokens; cache the verified routes
```

**Key hypothesis**: speculation gives the expert cache **advance notice of future routing
demand** (draft tokens have full forward passes, so their routes — or at least their
router logits — are available k tokens early). Two effects:

1. **More lookahead**: prefetch window becomes k tokens instead of 1 (F1@6 drops with
   distance, but the *union* grows predictably — the union curves are the empirical
   bounds; F(k) is measured, not assumed).
2. **Fewer executions**: expert-major execution reduces per-token expert compute 2.1–2.9×
   at k=8–24 (measured) — directly reducing the bytes that *must* be resident.

Caveat (HYPOTHESIS to test): draft-model routes are not the target's routes; only
*verified* positions give exact routes. The prefetch should rank by (verified ∪ predicted
with confidence), and the union curves were measured on the *target's* real traces, so
they bound the verified set, not the draft set.

---

## 14. Future experiment plan (post-training, GPU-enabled)

Ordered ladder; each step measures: expert hit rate, SSD misses/token, RAM misses/token,
bytes transferred/token, tok/s, per-token latency, VRAM/RAM/SSD bandwidth, plus
correctness (GEN compare / rms vs canonical) on the frozen E1 prompt set + 6-prompt trace
suite.

| # | experiment | configuration | gates |
|---|---|---|---|
| A | Baseline V4 expert residency | compact-VK canonical MXFP4, `DSV4_CACHE_OFF=1` (page-cache WARM), E1-frozen env (ARGSORT_CPU=1, staging 3456) | correctness parity with canonical reference (b0_k43 control, rms ~0.04) |
| B | Naive LRU RAM cache | same + `dsv4_expert_cache`, 4 GiB, mode=2 | A == B routes (trace compare); hit-rate/tok-s delta |
| **A vs B first** | **both required before C** | — | — |
| C | Larger HOT cache | VRAM slot expansion (e.g. 12–24 slots/layer via arena geometry change; budget 2–8 GiB) | correctness at >6 slots (ids view change) |
| D | RAM-backed WARM with bigger budget | 8–20 GiB RAM cache | RAM-miss rate vs budget slope |
| E | Prefetch (layer-windowed, §8) | wire `prefetch_expert` + `dsv4_on_decode_done` | SSD-miss reduction vs prefetch bandwidth |
| F | Predictive cache | transition model + hybrid score (§7/§8) | hit-rate lift vs E; pollution cost |
| G | Predictive + Re-Spark | draft model + union-prefetch (§13) | accepted-tok/s; union curve match |

**Do not implement the whole system at once.** The first runtime step after training is
**A vs B on the existing branch** — both paths already exist (A: CACHE_OFF, B: cache on);
the missing work is a controlled comparison harness (prompts, env, metrics, correctness),
which is pure tooling, not architecture.

---

## 15. Risks

1. **Scheduler/sync overhead dominates** (MEASURED ~310–475 ms/decode): until overlapped,
   no cache policy can produce >~2–4 t/s. Highest risk to the whole program; attack first.
2. **MXFP4 compact-VK path unvalidated**: E1 validated skeletons (IQ3/Q3/Q4); the
   canonical MXFP4 VK path had a crash fixed but unbuilt at last artifact. A rebuild +
   parity run is a prerequisite for A/B.
3. **Btrfs cold-read fragmentation** (MEASURED 1.22 GiB/s, 2× amplification): cold misses
   are ~3× slower than warm; prefetch converts cold→warm but the first-touch cost is real.
   Repack is a mitigation (validated separately).
4. **Prior locality results are trace-scale, not proven at runtime**: F1@6 0.368 from 17
   tokens; union curves from 6×256 tokens. Re-verify against live routing before
   committing to policy weights.
5. **Multi-token batch support**: the compact path is single-token decode; prefill runs
   token-by-token. Batch/prefill experts would multiply working-set demand.
6. **Slot-count assumption**: 6 slots/layer is baked into arena/GPU code (`n_slots`, ids
   width, `MAX_EXP`); raising it touches geometry + ids + remap — moderate, contained.
7. **Correctness regressions in the interception layer** (observed historically: stale
   arena copies, staging epoch races, ids staging reuse): the E1 verifier (214/214,
   GEN compare, hashes) must gate every experiment.
8. **Page-cache interplay**: W1 (explicit cache) + mmap double-buffers the same bytes
   (cache heap + page cache). `DSV4_FADV_DONTNEED` mitigates; measure RAM accounting.
9. **VRAM headroom**: dense core 13.71 GiB + KV (1M ctx, quantized) + staging (3 GiB
   pool) leaves a small residual; HOT-tier growth requires GTT or budget arbitration
   (§12).

---

## 16. Open questions

1. What exactly is the per-decode sched+sync breakdown at the current HEAD (MXFP4,
   rebuilt)? (The 310 ms "unattributed" item is the biggest unknown.)
2. Does the canonical MXFP4 compact-VK path reproduce b0_k43 (rms 0.04, GEN 13/13)?
3. Live F1@6 / effective-expert statistics on ≥1k tokens of the 0731 model — do the
   prior 17-token + 6×256-token results hold?
4. Is a repacked contiguous expert file feasible (Btrfs reflink/copy), and what does it
   do to cold-read bandwidth?
5. What RAM budget is actually free at runtime (OS + page cache + 13.71 GiB dense +
   KV)? The 32 GiB box is tight; the true WARM budget may be 8–16 GiB.
6. Can the eval-callback interception be moved to an async/stream path without changing
   the router graph? (Requires scheduler work; the TODO in ggml-backend.cpp 1883
   acknowledges the sync question.)
7. Does the hash-layer tid2eid routing cover more than 3 layers in later model versions?
   (Each added hash layer is a free prefetch hook.)
8. What is the draft model's route fidelity vs target (for G's union-prefetch)?

---

## 17. Final answers

### Q1: Can DeepSeek-V4-Flash Q8_K_XL plausibly become a REMORA SSD→RAM→VRAM expert-working-set system without changing learned routing?

**Yes — and it is the only viable execution mode on this hardware, not an optimization.**

Static evidence: (a) the expert bank (137.06 GiB) cannot fit VRAM (15.92 GiB) or RAM
(32 GiB); (b) experts are exactly-addressed slices (DSEI index, uniform 4.25 MiB MXFP4
spans) with zero parsing cost; (c) the fork already implements RAM-tier cache + 6-slot
VRAM working set + eval-callback interception that provably does not alter routing
(identical router graph, route-forcing diagnostics, E1-verified transport); (d) measured
t→t+1 expert overlap (F1@6 0.368) and popularity concentration (mean ~23 effective
experts/layer) give the locality the hierarchy needs; (e) measured bandwidths define
exactly what hit rates each tier must achieve. The learned routing is structurally
untouched: the interception layer consumes the router's *output* and manages only
residency.

### Q2: Minimum invasive modification to test the hypothesis?

**No new architecture — run A vs B on the existing branch:**

1. Rebuild HEAD with GGML_VULKAN (MXFP4 compact-VK fix already in source).
2. Freeze the E1 env (ARGSORT_CPU=1, staging 3456, CACHE_MB=4096, ngl=43, mode=2,
   arena=1, VK=1).
3. **A**: `DSV4_CACHE_OFF=1` (page-cache residency baseline).
4. **B**: cache on (explicit LRU RAM cache) — same prompts, same env.
5. Compare: route trace equality (DSV4_TRACE_FILE), GEN/rms parity vs canonical,
   hit rate, ms/token, SSD bytes (/proc/self/io), VRAM/RAM deltas.

Both arms exist today; only a comparison harness is missing.

### Q3: First experiment worth running after training finishes?

**Experiment A vs B** (baseline residency vs LRU RAM cache), because it (a) re-validates
the MXFP4 path, (b) establishes the true cold/warm/hit-rate baseline on real routing, (c)
calibrates the locality hypotheses against the current implementation, and (d) yields the
numbers that decide whether C (larger HOT tier) or E (prefetch) is the correct next step.
Only after A/B should any policy or prefetch work begin — and the scheduler/sync overlap
work should proceed in parallel, since it gates everything above ~2 t/s regardless of
cache design.

---

# 18. STATIC SIMULATION RESULTS (CPU-ONLY, 2,836-token corpus)

Full methodology, tables, and caveats: `V4_REMORA_EXPERT_CACHE_STATIC_RESULTS.md`.
Artifacts: `results/v4_expert_cache_static/` (sim_cache.py + CSV/JSON). Headlines:

1. **Locality at scale is weaker and longer-tailed than the 17-token study suggested**:
   unique experts/layer 237/256, entropy 6.65 bits, next-token overlap 1.68/6
   (F1@6 0.309), top-6 cumulative coverage only 23.6%, burstiness dispersion ~3,500.
   Cross-layer prediction re-falsified (overlap 0.022–0.028).
2. **Working-set curves**: 6 slots/layer → 20% hot hits; 24 → 57%; 64 → 88%
   (64/layer = 35 GiB — impossible in VRAM; the tail must live in WARM).
3. **RAM-tier**: 4 GiB WARM collapses SSD traffic to ~0.28% cold (9 MiB/token) on these
   traces; WARM > 4 GiB buys nothing byte-wise; the binding cost in the WARM regime is
   the RAM gather → staging → H2D path (2.42 GiB/token at S=6).
4. **Policies**: LRU wins the HOT tier; frequency/reuse eviction hurts on bursty
   traffic; hybrid is marginally best (22.0% hot, 2.35 GiB/t RAM gather). Policy choice
   is second-order vs tier sizing.
5. **Prefetch**: byte-level negative at every WARM size (−4% to −31% SSD); precision
   47–52% (windowed). Value, if any, is latency hiding — runtime measurement required.
6. **SSD pattern**: 774 preads/token; adjacent-expert coalescing negligible (2.4%);
   offset-sorted batches cut the seek-cost proxy 89%.
7. **Speculation unions**: k=8 → 1,203 experts / 15.0 GiB (F=0.583 dilution; ≈ hermes
   F(8)=4.52 cross-validated); per-token prefetch burden 1.87 GiB at k=8 — fits 4 GiB
   WARM.
8. **Scenario ranges** (serial component model, not promises): 0.9–1.3 t/s
   conservative, 1.0–1.4 plausible, 2.9–4.3 aggressive; the scheduler/sync term
   (398 ms) is 2–3× compute+stall in every scenario.

# 19. GIGATOKEN-INSPIRED CRITICAL-PATH OPTIMIZATION

External reference (systems principles only; no code copied): Gigatoken's demonstrated
GB/s-class gains from attacking "boring" overheads — SIMD preprocessing, minimized
branching, aggressive mapping caches, native file APIs, batching, long-tail cache care.

## 19.1 Overhead hotspots (per decode, from code + measured attribution)

| hotspot | measured / derived | class |
|---|---|---|
| sched-loop + per-range sync | 87.8 ms sync + ~310 ms sched (era) | per-token fixed cost |
| 43 read_begin/read_wait batches | 43 waits/token | per-token fixed cost |
| 258 mutex-guarded residency lookups | 6×43 lock acquisitions | per-expert cost |
| 43 staging epochs + defer-queue ops | 43 alloc/populate/queue cycles | per-layer cost |
| eval-callback string matching per node | `strstr(n, "ffn_moe_topk")` ×5+ per node over ~1,000+ nodes | per-node cost |
| graph build | 38.5 ms/token | per-token fixed cost |
| topk host readback | 43 × tensor_get | per-layer cost |

**Pattern: the runtime pays expensive fixed costs repeatedly** — the Gigatoken lesson
applies directly: these are the same class of "small repeated overheads" that dominate
wall time.

## 19.2 Per-expert fixed costs → one transaction

Current: 6 experts → 6 lock-guarded lookups → per-expert LRU list surgery → per-layer
batch wait. Target (design): top-6 IDs → one residency transaction (single lock, 6
probes, 1 LRU touch) → one read plan → coalesced I/O (offset-sorted batch) → one
promotion epoch → one synchronization point. The cache API already supports the batch
shape (`read_begin/issue/wait`); the per-layer wait and per-expert locking are the
removable costs.

## 19.3 Metadata caching (Gigatoken mapping-cache principle)

Already cached: DSEI spans (O(1) `(layer<<16|expert)` → 3 spans), tid2eid table
(3 layers), arena last_ids. Missing (cheap, CPU-only):
- per-layer HOT membership as a 256-bit bitset (4×64-bit words; vectorizable
  membership test for the 6 ids in ~2 SIMD ops),
- per-token dedup/sort of the 6 ids via a sorting network (no branches),
- a node-name → action precomputed table for the eval callback (eliminate the repeated
  `strstr` per node; names are fixed at graph build).

## 19.4 Vectorized / branch-light planning (CPU, no GPU)

- Residency: bitset AND + popcount for the 6 ids (AVX2; ~10 cycles/layer).
- Dedup: the router output is already distinct (argsort top-k); dedup is a no-op —
  remove the branch.
- Request planning: 18 spans/layer from the flat index; build the batch as a struct
  array, sort by (file_idx, offset) with a 18-element insertion sort (n=18, predictable),
  submit to the QD8 pool.
- LRU: replace list-scan-with-remove by a generation counter per slot (evict = lowest
  generation; no list surgery) — the sim's LRU behaves identically (recency = age).

## 19.5 Direct-I/O methodology

Prior A/B/C (measured): O_DIRECT no change; WILLNEED worse; warm QD8 7.2 GiB/s best;
cold scatter 1.22 GiB/s. **Conclusion: keep pread + page cache + QD8; do not adopt
O_DIRECT** on this workload/Btrfs. The productive I/O changes are: offset-sorted batch
submission (−89% seek-cost proxy) and a repacked contiguous expert file (converts
33-read scatter per expert into ~2 reads; HAR's sidecar format is the reference design —
§20).

## 19.6 Scheduler fusion

Merge the 43 eval-callback splits into one graph view per token with a single sync at
the end (ids read via async readback). The callback stays as the *residency* hook but
stops being the *synchronization* hook. This is the single highest-leverage change per
§19.1 (398 ms of the ~768 ms conservative scenario total).

## 19.7 Cache-policy implications (long tail)

The tail is real (top-48 = 70% coverage). One policy cannot be both "hold the frequent
head" and "never miss the rare tail"; the policy must be recency-dominant (LRU/hybrid)
with the tail paid in WARM bytes, not HOT slots. Keep the policy simple enough to be
cheaper than the overhead it replaces (the hybrid score's marginal 2% hot-hit gain is
only worth ~µs of added CPU per token — it is not worth list scans).

## 19.8 Proposed benchmark dashboard (per decode, both A/B arms)

expert lookups/s; cache decisions/s; metadata lookups/s; requests/s; bytes planned/s;
bytes read/s (SSD); bytes promoted/s (RAM→staging); H2D bytes/s; scheduler
transactions/token; synchronization points/token; CPU ms/token; I/O wait ms/token;
useful-hit rate (Avanza §8). First experiment remains **A vs B** — this audit found no
clearly superior minimal experiment; it found that A/B must be *instrumented* with the
dashboard to be conclusive.

# 20. HAR (HARDWARE-AWARE RUNTIME) INTEGRATION FINDINGS

HAR (`[home-relative path omitted]`, Rust) is the user's own tiered-memory runtime; the V4
expert-streaming design and HAR are complementary, not competing:

| V4 fork mechanism | HAR equivalent (verified this session) |
|---|---|
| DSEI v2 index (27 B/expert, O(1) spans) | `har.expert_sidecar.v1` (`HAR_EXPERT_SIDECAR_FORMAT.md`): 4096-aligned payloads, SHA-256 per entry, `(layer, expert, projection)` keyed index, no runtime tensor scanning; fixture `artifacts/model/tiny_compiled_package/expert_subset.harx` |
| `dsv4_expert_cache` (LRU, in_use pinning) | `ResidencyManager` (`crates/har-residency/src/manager.rs`): PageLease (eviction fails while leased), generations/epochs, tickets, replicas, `evict_vram/evict_ram` |
| QD8 pread pool | `WavefrontScheduler` (`crates/har-residency/src/scheduler.rs`): WAITING→NVME_READ→RAM_TO_VRAM→GPU_COMPUTE→DONE, mandatory-vs-speculative, deadlines, `cancel_speculative` |
| eval-callback interception | compiled plans (`.har`) carrying explicit residency decisions; no interpreter in the hot path |
| MXFP4 expert payload | `HAR_MXFP4_UD_OPERATION_PROPOSAL.md`: block_mxfp4 17 B/32 el, rows 2,176 B × 2,048, **rows independently addressable** (sub-expert transfer unit), payload checksum verified by O_DIRECT read |
| shadow prices / budget arbitration | `HAR_RUNTIME_SHADOW_PRICE_FORMULAS_V1.md` (SP01–SP20, fail-closed admission) — the future joint expert+context allocator (§12) can adopt this ledger |
| context hierarchy (design) | `HAR_HOT_COLD_MEMORY_PROJECTION.md`: EXACT_HOT / RECONSTRUCTED_WARM (LRU) / LATENT_COLD + TOKEN_ARCHIVE — the context tiering already exists in HAR |
| Re-Spark speculation | `HAR_SPECULATIVE_HORIZON_SENSITIVITY.json` (synthetic): H=8/A=6 → 98.9 MB streamed per accepted token, up to 7.9× vs token-by-token; `ExpertUnionEstimate` in `crates/har-residency/src/mtp.rs` |

Key transfers into the V4 plan:

1. **Row-level transfer granularity** (2,176 B rows): enables partial-expert promotion
   (first N rows), pipelined H2D, and finer prefetch — a capability the current
   whole-expert (4.25 MiB × 3) path lacks.
2. **Generation/lease discipline** (HAR): replace ad-hoc `in_use` with generation-tagged
   slots + lease-based eviction — the correctness-critical pattern that the fork's
   historical bugs (stale arena, staging races) were symptoms of missing.
3. **Speculation admission policy** (HAR: no prediction-only prefetch; speculative work
   only with measured queue slack) aligns with the sim's prefetch finding (§18.5):
   prefetch must be gated on slack, not issued blindly.
4. **VLT-0119 falsification** (HAR vault, measured trace): exact-set repeat 0.0%, mean
   Jaccard 0.27, p=40.6% vs break-even 50%/97.5% — an independent measured confirmation
   of this corpus's low next-token set fidelity; any predictor must be evaluated against
   these break-even curves before deployment.
5. **Repack path**: the sidecar format (aligned, checksummed, ordered payloads) is the
   reference for the "repacked expert file" that defeats Btrfs 128 KiB-extent scatter
   (cold 1.22 vs warm 7.2 GiB/s).

Division of labor (future): REMORA decides what matters (importance, policy, prefetch
set, Re-Spark horizon); HAR moves it (residency, page store, sidecar I/O, RDNA4
execution); llama.cpp fork remains the reference oracle for differential correctness.

# 21. DECISION FRAMEWORK (from simulation evidence)

| question | answer | evidence |
|---|---|---|
| A. Is explicit LRU worth testing? | **Yes** — it is the correct baseline arm (B), not because it wins on bytes (policies are within 2 pp at 4 GiB WARM) but because it is the measured, instrumentable reference for the scheduler-attribution question | §18.4, Avanza §9 |
| B. What HOT size first? | **6 slots/layer (status quo) for A/B; 12–24 for the follow-up** — 6→24 moves hot hits 20→57% and cuts RAM gather ~2.4→1.1 GiB/token; 64/layer (88%) is physically impossible in VRAM | §18.2 |
| C. Is prefetch worth prioritizing? | **No, not before the scheduler work and not as a bytes play** — byte-negative at all WARM sizes in sim; only a latency-hiding hypothesis remains | §18.5 |
| D. Most urgent scheduler optimization? | **One transaction per token: single read batch + single sync + async topk readback** (kills ~43 syncs/waits and most of the 398 ms term) | §19.1/19.6, Avanza §3 |
| E. Can Re-Spark improve prefetch lead time? | **Structurally yes** — k-token advance routes (hash layers exact; dense layers via draft forward), union increments 1.87 GiB/t at k=8 fit 4 GiB WARM; but predictor precision (≈50%) caps the benefit until the scheduler path is fast | §18.7, §13 |
| F. Minimum invasive runtime patch? | **Instrument A/B** (phase timers, sync counter, useful hits, queue depth) on the existing branch; the code changes are env-gated counters + CSV/JSON emission, CPU-only | Avanza §9 |
| G. First GPU experiment? | **A = page-cache residency vs B = explicit LRU 4 GiB**, frozen E1 env, identical prompts, with the dashboard; expect scheduler-bound both arms per sim | §18.9/§18.10 |

# 22. CACHE STATISTICS / EVENT TRACE FORMAT (compact, bounded)

## 22.1 Per-decode summary record (binary, 32 B — one per decode, append-only)

```text
u32 magic 0x56444543 ("CDEV")
u16 version = 1
u16 decode_seq
u32 n_hot_hit, n_warm_hit, n_cold_miss      (per decode, over 258 accesses)
u32 ssd_bytes, ram_gather_bytes             (promoted this decode)
u32 n_promotions, n_evictions_hot, n_evictions_warm
u16 n_prefetch_req, n_prefetch_correct      (2-token window)
u16 n_sync_points                           (scheduler transactions)
u32 sync_us, sched_us, build_us, callback_us (phase timers)
u32 max_queue_depth                         (QD8 pool high-water)
```

## 22.2 Per-access event stream (CSV, bounded sampling — ring of N=4096 events)

```text
token, layer, expert_id, hot_hit, warm_hit, cold_miss,
prefetch_requested, prefetch_correct, load_bytes, load_latency_est_us
```

- Emitted only when `DSV4_EVENT_SAMPLE=1`; ring-buffered in memory, flushed on a
  rolling 4,096-event window (≈16 decodes) — never a per-token file write.
- `load_latency_est_us` is filled from the QD8 pool's per-read completion timestamps
  (already available at `read_wait`); it is an *estimate*, not a device measurement.

## 22.3 Machine-readable summary

`dsv4_cache::stats` (existing) extended with: useful hits (Avanza §8), promotion
events, queue depth, sync count, and the phase timers — emitted as JSON at exit and
on a signal. This is the Avanza dashboard's data source (§19.8 of this doc) and is
the required instrumentation for the A/B experiment.

# 23. PREFETCH EVALUATION — FINAL NUMBERS (supplement to §18.5)

| predictor | per-layer set | coverage (of next 6) | precision | wasted MiB/token |
|---|---|---|---|---|
| P1 copy-last | 6 | 29.0% | 29.0% | 54.3 |
| P2 argmax-successor (demo) | 6 | 18.0% | **50.1%** | 14.9 |
| P3 transition-mass | 6 | 36.7% | 36.7% | 48.4 |
| P3 transition-mass | **12** | **49.6%** | 24.8% | 115.0 |
| P4 popularity | 6 | 31.4% | 31.4% | 52.5 |

Reading: P2 is the most precise per byte (50% precision, least waste) but the lowest
coverage; P3-top12 buys +13pp coverage at double the waste — the transition model
saturates at top-6 and the tail is unpredictable (consistent with the 6.65-bit
entropy and the VLT-0119 break-even falsification). **The prefetch budget should be
P2-sized (6/layer, ~0.2 GiB/token of reads at 47–52% correctness), never P3-top12.**

# 24. FINALIZED CPU-SIDE DESIGNS (post-simulation; see STATIC_RESULTS §6)

## 24.1 Batched expert transaction (final)

```
TokenPlan (per token, ONE commit barrier — sched_sim.py proves the semantics)
  route    : canonical router -> actual top-6 per layer (untouched)
  resolve  : bulk residency probe (6 ids, 1 lock) -> hot/warm/cold classification
  cache    : admission decision (cold -> WARM always; prefetch-pool entries wait)
  request  : offset-sorted span batch (DSEI O(1) spans; 774 -> 1 sorted batch)
  promote  : staging population (NT copies) + deferred H2D copies
  commit   : ONE barrier; all required experts ready else FAIL_CLOSED (no partial)
```

Sync-point reduction: 43 per-layer waits/syncs -> 1 commit per token. Proven
properties (7/7 tests): readiness before commit; missing dependency fails closed;
no partial state; deterministic cancellation; stale-generation completion rejected.

## 24.2 Admission vs eviction (final split)

- ADMISSION: cold misses always admitted (just-used evidence). Prefetched entries are
  FETCHED-BUT-NOT-ADMITTED (prefetch pool); they graduate on actual access. This is
  the property that makes non-displacing prefetch byte-positive (§6.1 results).
- EVICTION: recency-dominant hybrid score 0.4·rec + 0.4·freq + 0.2·pred among
  resident entries only. Prefetch-pool entries never trigger resident eviction.
- Generation/lease discipline (HAR ResidencyManager semantics) guards both: eviction
  refused while leased; stale completions never resurrect.

## 24.3 Non-displacing prefetch (final parameters)

- WARM ≥ 1 GiB required for positive economics; recommended 1 GiB/50%/w12 (net
  +131 MiB/t) or 2 GiB/20%/w12 (+75 MiB/t, 2.3 MiB/t waste, 94.3% precision).
- Window w = 8–12 layers (lead-time-safe: early layers cannot be hidden by
  same-token prefetch; w>12 adds no bytes at fixed budget).
- Never prefetch at 4 GiB WARM (no-op; cold already ~0.3%).

## 24.4 DSEI lookup (verified)

Current dict-of-tuples is sufficient: 45 µs/token vs 64 µs flat / 39 µs batched —
negligible vs the ms-scale decode. No change required.

## 24.5 Sidecar format review (directive §23; HAR har.expert_sidecar.v1)

Requirements check: exact expert offsets ✓ (entry fields), generation ✓
(generation-tagged slots in ResidencyManager), checksum/hash ✓ (SHA-256 per entry),
page size ✓ (4096 alignment), quantization format ✓ (quant_format incl. MXFP4),
alignment ✓, residency metadata ✓ (UnifiedPageRecord), optional prefetch metadata —
**minimal gap**: add an optional `prefetch_hint { predicted_next_use_token,
priority }` field per entry (additive; does not change the frozen V1). No other
schema change needed. The DSEI v2 index and the sidecar are complementary: DSEI is
the in-fork runtime index; the sidecar is the repacked, checksummed, publication
format (Btrfs-scatter fix path).

# 25. FIRST GPU EXPERIMENT — FINAL RUN BOOK (harness.py emits the manifest)

- Arm A: `DSV4_CACHE_OFF=1` (page-cache residency). Arm B: cache on, 4 GiB.
- Frozen env: ARGSORT_CPU=1, staging 3456, ngl=43, mode=2, arena=1, VK=1, same
  prompts (6 × 256 tokens), same router, same runtime generation (f3b8aba5d).
- Instrumentation: DSV4_DEBUG phase timers, sync counter, useful hits, promotion
  events, queue depth, /proc/self/io (SSD bytes), RAM deltas, GEN compare.
- Output schema (§25 directive) with `unavailable` (never zero-faked) for missing
  fields; run_id = f(git, model, config, workload) — harness.py verified.
- Decision gates: scheduler fraction of decode; SSD bytes at 4 GiB; RAM gather
  GiB/token; parity (rms vs canonical, GEN identical); then §7 matrix next actions.
