# DSpark + Memory Look-Ahead: Static Feasibility Analysis

> Research Date: 2026-07-30
> Context: Active DeepSeek-V4-Flash inference experiment running — static analysis only.

---

## PART 1 — Current DSpark Control Flow (DeepSpec Domain)

### 1.1 Where Draft Tokens Are Produced

**Location:** `archive/projects/src/DeepSpec/deepspec/eval/dspark/draft_ops.py`

Function `build_dspark_proposal()` is the core draft-generation path.

Sequence:

1. `forward_dspark_draft_block()` — single forward pass through the DSpark backbone (5-layer transformer). Takes `target_hidden_states` (concatenated activations from target layers [1,16,31,46,61]) and a noise embedding.

2. `model.compute_logits(proposal_hidden_states)` — project backbone output through lm_head → `[1, block_size, vocab]`.

3. `model.sample_draft_tokens(base_draft_logits, ...)` — autoregressive sampling via Markov head across `block_size` positions. The Markov head (VanillaMarkov with rank 256) applies per-step bias based on previously sampled tokens.

4. If confidence head exists: `_predict_confidence_logits()` — predicts per-position acceptance logits from hidden states + optional Markov embeddings.

### 1.2 What Determines Maximum Speculation Depth

**Fixed at `block_size` = 7 tokens.**

Defined in: `deepspec/eval/dspark/evaluator.py:37`

```python
@property
def max_proposal_tokens(self) -> int:
    return int(self.draft_model.block_size)
```

The `block_size` is a model-level constant set in the DSpark config (`config/dspark/*.json`), stored in the GGUF, and fixed at training time.

### 1.3 Is Speculation Depth Adaptive?

**Partially — confidence-gated truncation only.**

The DSpark evaluator supports a `--confidence_threshold` parameter. When set:

1. `_predict_confidence_logits()` computes per-position confidence via sigmoid of a learned linear head
2. `_confident_prefix_length()` finds the first position where confidence < threshold
3. The proposal is truncated to that prefix

**Key code** (`draft_ops.py:75-79`):
```python
proposal_draft_tokens = _confident_prefix_length(
    confidence_logits,
    block_size=block_size,
    threshold=float(confidence_threshold),
)
```

If `proposal_draft_tokens == 0`, returns empty proposal (0 draft tokens).

**Without confidence head** (threshold=0), all 7 tokens are always proposed.

There is **no dynamic adaptation** based on cache state, system load, or I/O conditions. The maximum is always 7, and truncation is purely confidence-based.

### 1.4 Existing Controls

| Control | Location | Behavior |
|---------|----------|----------|
| `block_size` | Model config (fixed) | Max draft tokens = 7 |
| `confidence_threshold` | CLI arg | Truncate proposal where sigmoid(confidence_logit) < threshold |
| `temperature` | CLI arg | Sampling temperature for draft (also affects target verification) |
| Standard rejection sampling | `base_evaluator.py:186-194` | `accept_prob = min(1, p_target / p_draft)` |
| Stop token termination | `base_evaluator.py:196-206` | Early termination if accepted token is EOS |

### 1.5 Where Target Verification Begins

**Location:** `base_evaluator.py:function verify_draft_tokens()`

Called from `generate_decoding_sample()` after `propose()` returns.

Sequence:
1. Target model receives `verify_input_ids` (current token + up to 7 draft tokens)
2. Target forward pass produces logits for all positions
3. For each draft position: compute `accept_prob = min(1, p_target(token) / p_draft(token))`
4. Compare against random uniform → accept/reject
5. If rejected at position k: sample from residual `max(0, p_target - p_draft)` for token k

### 1.6 Must Draft Complete Before Target Verification?

**YES — strictly sequential.**

```python
# base_evaluator.py:246-251
proposal = propose(...)           # draft ALL tokens first
verification = verify_draft_tokens(...)  # THEN verify
```

No pipelining. Draft generation for the next iteration starts only after verification and `update()` complete.

### 1.7 Acceptance/Rejection Calculation

**Location:** `base_evaluator.py:186-194`

```python
accept_prob = torch.clamp(
    selected_target_probs / selected_draft_probs,
    max=1.0,
)
accept_mask = (torch.rand_like(accept_prob) < accept_prob).to(torch.int64)
accept_prefix_mask = accept_mask.cumprod(dim=1)
```

Strict rejection sampling: accept with probability `min(1, p_target / p_draft)`.

On rejection at position k:
```python
next_token = sample_residual(
    target_probs[:, accepted_draft_tokens, :],
    draft_probs[:, accepted_draft_tokens, :],
)
```
Residual: `max(0, p_target - p_draft)`, renormalized and sampled.

### 1.8 Information Available Per Speculative Position BEFORE Target Verification

| Signal | Source | Available? | Notes |
|--------|--------|-----------|-------|
| Token ID | `sampled_tokens` after Markov head | YES | 7 speculative token IDs |
| Token probability | `draft_logits` → softmax | YES | Per-position full distribution |
| Confidence/margin | `confidence_logits` → sigmoid | Conditional | Only if model has confidence head |
| Hidden state (draft) | `proposal_hidden_states` | YES | Shape `[1, 7, 5120]` — backbone output per position |
| Hidden state (target) | `context.target_hidden_states` | YES | From LAST verified target forward pass |
| Draft logits | `base_draft_logits` | YES | Raw logits before Markov correction |
| Draft router output | N/A | **NO** | Draft model has no MoE router |
| Target-compatible rep | N/A | **NO** | No target-layer representations for future positions |
| Acceptance predictor | `confidence_head` | Conditional | Only with confidence head; predicts per-step acceptance |

**Critical gap:** The draft model produces NO expert routing information. It is a pure next-token predictor. It has no concept of the target's 256-experts-per-layer MoE architecture.

---

## PART 2 — Can DSpark Predict Future Expert Demand?

### 2A. Direct DSpark Routing Signal

**NO.** DSpark has no MoE layers. It is a dense transformer with 5 layers, 5120 hidden size. It does not produce router logits, expert IDs, or any expert-routing information.

The target model's router operates on target hidden states at each layer. The draft model's backbone outputs are in a different representation space.

### 2B. Draft Hidden-State Predictor

**Theoretically possible but non-trivial.**

Available signals:
- `proposal_hidden_states[i]` for position i (draft backbone output at that step)
- `target_hidden_states` from previous verification (actual target activations at known positions)
- Markov head embeddings (token-level information)

A predictor could be trained:
```
draft_hidden[i] → MLP → expert_probabilities[layer][expert] for future positions
```

**Cost estimate:** A tiny MLP (e.g., 5120 → 1024 → 256 per layer) × 43 layers = ~11M parameters. Forward pass for 7 positions = negligible vs. 284B target model.

**Key uncertainty:** How well do draft hidden states correlate with future target expert routing? This is an empirical question that requires real data collection. The draft model is trained to predict next tokens from target hidden states, which creates some representational alignment, but the degree is unknown.

### 2C. Token-Conditioned Transition Model

**Feasible — this is the most promising lightweight approach.**

Given:
- Last verified routing: `{(layer, expert_id): router_weight}` for all 43 layers at the current accepted token
- Speculative next tokens: `{t+1, t+2, ..., t+7}`

Build a model that, for each layer, predicts:
```
P(expert_id | previous_experts, speculative_token, layer)
```

**Approaches, in increasing complexity:**

1. **Per-layer co-occurrence table** — For each layer, build `[256, 256]` transition matrix counting which expert follows which in adjacent tokens. Look up next speculative token's likely experts as those that co-occurred with current token's experts in training data.

2. **Per-layer bigram by token cluster** — Group tokens into 1000-10000 clusters. For each cluster+layer, store empirical expert distribution.

3. **Tiny learned predictor** — Small MLP: `[token_embed + token_id + prev_expert_ids] → [256 probs per layer]`. Could be trained as a secondary task using routing traces already being collected.

**Storage cost:** Option 1 needs 43 × 256 × 256 × 4 bytes ≈ 11 MB. Option 2 needs more but still manageable. Option 3 needs ~1-5M parameters.

### 2D. Partial Target Router Execution

**This is feasible and potentially the most accurate.**

The target router computation per layer is:
```python
logits = gate_inp @ hidden_state  # [256, d] @ [d, 1] → [256]
probs = sqrt(softplus(logits))
probs += exp_probs_b
top_k_ids = argsort(probs)[:6]
```

**Cost:** For each layer:
- 1 MV multiply: 256 × d (d = 4096 for DeepSeek-V4-Flash) = ~1M MACs
- 1 softplus + sqrt + top-k ~ negligible

**Total for 43 layers:** ~43M MACs — trivially cheap (microseconds on CPU, ~0.01ms on GPU).

**What we need:** The draft backbone hidden state must be projected into the target's representation space at each intermediate layer. The target router consumes target hidden states at each MoE layer. The draft model does not produce intermediate representations at the target's 43 layers.

**Solution:** Run a lightweight "router probe" — a learned linear projection:
```
draft_hidden[i] → W_layer @ draft_hidden[i] → [4096] → target router → expert_ids
```

This requires 43 learned projections (one per target MoE layer), each 5120 × 4096 ≈ 21M params. Total: ~900M params — larger than desired but still tiny vs 284B target model.

**Better approach:** Train a single shared projection + layer-specific bias.

**Verdict:** Partial router execution is the most accurate option but requires either:
- A learned representation mapper (draft-space → target-space)
- Or accepting lower accuracy from direct token-conditioned prediction

### 2E. Hybrid Approach (Recommended for V1)

Combine multiple cheap signals:

```
expert_prediction =
    f(draft_token_id, draft_hidden_state, last_target_routing, layer)
```

**V1 design:**
1. Use **last verified routing** as the primary signal (empirically, 25% Jaccard overlap between adjacent tokens — from `REAL_ROUTING_LOCALITY.md`)
2. Use **speculative token ID** to modulate — look up token→expert co-occurrence statistics from trace data
3. For confidence weighting, use the DSpark **confidence head** output

**This requires NO changes to DSpark** — it only needs to consume DSpark's outputs (token IDs, confidence) plus the routing trace infrastructure that already exists.

---

## PART 3 — Adaptive Speculation Horizon

### 3.1 Rationale

Current behavior: fixed N=7 (or N from `--draft-max` in llama.cpp).

The core insight from this research: **more speculation is not always better** when memory is the constraint.

### 3.2 Marginal Value Model

For each additional speculative position k:

```
value(k) = expected_accepted_compute_benefit(k)
         + expected_memory_stall_removed(k)
         - draft_compute_cost(k)
         - expected_wasted_io(k)
         - expected_cache_pollution(k)
```

Net benefit of depth K: `V(K) = Σ_{k=1..K} value(k)`

The optimal speculation depth is `max K s.t. V(K) > 0`.

### 3.3 Model Signals (Already Available)

| Signal | Source | Form |
|--------|--------|------|
| Per-position confidence | `confidence_logits` → sigmoid → cumprod | `[0,1]` per position |
| Historical acceptance by position | `evaluator.py` position stats | Acceptance rate per depth |
| Proposal probability | `draft_probs` | Full distribution entropy |
| Rejection history | Runtime accumulator | Running rejection rate |

### 3.4 System Signals (Need Runtime Plumbing)

| Signal | Source | Form |
|--------|--------|------|
| GPU expert-cache hit rate | Future cache monitor | `[0,1]` |
| RAM-cache hit rate | `dsv4_expert_cache` stats | `[0,1]` |
| SSD queue depth | Future I/O monitor | Integer |
| Outstanding SSD bytes | Future I/O monitor | Bytes |
| RAM→GPU transfer backlog | Future transfer monitor | Bytes |
| GPU expert-slot pressure | Future GPU cache state | `[0,1]` (fraction occupied) |
| Eviction pressure | `dsv4_expert_cache` LRU stats | `[0,1]` |
| Compute backlog | `llama_decode` timing | ms/token |

### 3.5 Conceptual Controller

The simplest useful controller compares:

```
expected_benefit_of_deeper_speculation =
    P(acceptance) × (compute_benefit_per_token + cold_bytes_prefetched)

expected_cost =
    (1 - P(acceptance)) × draft_cost + wasted_io_bytes + pollution_risk
```

Extend horizon while `E[benefit] > E[cost]`.

---

## PART 4 — I/O-Budgeted Speculation (Byte-Budgeted)

### 4.1 The Key Idea

Instead of speculating exactly N tokens, speculate until the predicted cold-memory work reaches a budget.

**Example scenario:**

| Position | Confidence | Unique Cold Experts |
|----------|-----------|-------------------|
| t+1 | 0.95 | 150 experts (1.4 GiB) |
| t+2 | 0.90 | 80 more (0.7 GiB) |
| t+3 | 0.85 | 200 more (1.8 GiB) |
| t+4 | 0.70 | 120 more (1.1 GiB) |
| t+5 | 0.40 | 60 more (0.6 GiB) |

If prefetch bandwidth budget is ~2 GiB per token cycle:
- Speculating 2 tokens: 2.1 GiB cold I/O needed
- Speculating 3 tokens: 3.9 GiB — may exceed budget
- Speculating 4 tokens: 5.0 GiB — definitely exceeds

**The optimal horizon depends on cache state and I/O capacity.**

### 4.2 What's Needed

1. **Per-position expert prediction** (from Part 2) — which experts are likely needed at each future position
2. **Residency tracking** — is each predicted expert already in VRAM/RAM?
3. **Cold-byte estimation** — sum of (predicted expert bytes × P(needed) × P(not cached))
4. **I/O budget** — remaining SSD bandwidth quota for this token cycle

### 4.3 Integration Points

The DSpark proposal already contains `block_hidden` (backbone outputs for all 7 positions). A lightweight expert predictor could be applied per position before target verification.

---

## PART 5 — Expert Prefetch Priority Function

### 5.1 Candidate Scoring

For each predicted future expert `E = (layer, expert_id, speculative_position)`:

**Simple heuristic score:**
```
score = P_expert_needed × spec_confidence × urgency_factor / bytes
```

Where:
- `P_expert_needed`: estimated probability this expert is activated (from predictor)
- `spec_confidence`: cumprod confidence of reaching this speculative position
- `urgency_factor`: `1 / token_distance` (near-term experts more urgent)
- `bytes`: expert size in bytes

**More sophisticated score:**
```
expected_stall_saved = P_needed × spec_confidence × stall_cost_per_miss
expected_prefetch_cost = transfer_time × bandwidth_impact + cache_pollution_cost
score = expected_stall_saved - expected_prefetch_cost
```

### 5.2 Priority Queue Architecture

```
submit_prefetch(layer, expert_id, priority, deadline):
    queue.push({layer, expert_id, priority, deadline})

process_prefetch_queue():
    sort by priority
    while queue not empty AND budget remains:
        pop highest-priority
        if expert already resident: skip
        if expert predicted cold: queue SSD read + RAM→GPU transfer
        update budget tracking
```

### 5.3 Scoring Policies (Increasing Sophistication)

| Policy | Description | Implementation Cost |
|--------|------------|-------------------|
| **Greedy FIFO** | Prefetch all predicted experts in speculative order | Trivial |
| **Confidence-weighted** | Score = cumprod_confidence × P_expert_needed | Low |
| **Deadline-aware** | Score = 1 / (token_distance × layer_index) | Low |
| **Cost-benefit** | Score = expected_stall_saved / bytes | Medium |
| **Learned** | Learned score from historical prefetch benefit | High (V3) |

---

## PART 6 — Deadline-Aware Prefetch

### 6.1 Deadline Model

Each future expert has a natural deadline:

```
deadline(E) = current_time + token_distance × expected_decode_time
```

Example:
- t+1, layer 2 deadline is very soon (~2 × decode_time)
- t+1, layer 40 deadline is ~40 layer-times away
- t+4, layer 40 deadline is much further

### 6.2 Scheduling Representation

```python
{
    "token_distance": 1,          # t+1
    "layer": 2,                   # layer index
    "expert_id": 147,             # expert ID
    "spec_confidence": 0.95,      # cumprod confidence
    "p_needed": 0.85,             # predicted activation probability
    "residency": "cold",          # current location: vram/ram/ssd/cold
    "bytes": 9863168,             # total gate+up+down bytes (IQ3_XXS)
    "estimated_deadline": 0.05,   # seconds from now
    "transfer_time": 0.003,       # estimated RAM→GPU time
    "read_time": 0.010,           # estimated SSD→RAM time
}
```

### 6.3 Earliest-Deadline-First + Value Filtering

1. Filter: remove experts where `spec_confidence × p_needed < threshold`
2. Sort by `estimated_deadline` (ascending)
3. Process within I/O budget
4. Prefetch: SSD→RAM for cold, RAM→GPU for warm (only when needed)

### 6.4 Timeline Optimization

The key hypothesis:

```
GPU: compute t/L17 ─ compute t/L18 ─ compute t/L19 ─ compute t/L20 ...
NVMe:           read t+1/L2 expert ─ read t+1/L8 ... ─ read t+2/L2 ...
PCIe:                                     upload t+1/L2 expert ...
TARGET: expert already HOT when reached
```

**This is the central benefit of speculative memory look-ahead.** Without speculation, the system has zero look-ahead — every expert load starts only when the target model reaches that layer for the current token. With 7-token speculation, there's up to 7 × 43 = 301 layer-slots of look-ahead.

**Feasibility: YES, this is the most promising aspect of the proposal.**

The current `dsv4_bind_callback` mechanism already fires per-layer after router computation. Extending it to use speculative predictions rather than immediate router output is architecturally straightforward.

---

## PART 7 — Rolling Speculation Window

### 7.1 Current Behavior

**DSpark (Python):** Strict `draft_all → verify_all → update → draft_next` — no overlap.

**llama.cpp speculative:** Same — draft N tokens, verify N tokens, repeat.

### 7.2 Rolling Look-Ahead Concept

```
Target: verify t+1 ─ verify t+2 ─ verify t+3 ─ ...
Draft:  draft t+N+1 ─ draft t+N+2 ─ draft t+N+3 ─ ...

Timeline:
    Phase 0: Draft tokens [t+1, ..., t+N]
    Phase 1: Begin verify t+1; simultaneously draft t+N+1
    Phase 2: Verify t+2; draft t+N+2
    ...
```

### 7.3 Dependencies and Hazards

**KV cache dependency:**
- Draft generation needs the last accepted token's embedded representation
- Target verification extends target KV cache
- These are independent KV caches (separate models in DSpark, separate contexts in llama.cpp)

**State hazards:**
- DSpark draft needs `target_hidden_states` from the last verified position → must be updated after each token verification (not batch)
- This is already true in the current design (the draft reads target hidden states from context)

**Current architecture assessment:**

DSpark **could theoretically support rolling speculation** because:
1. Draft and target are separate models with separate KV caches
2. The draft backbone forward pass is non-causal within the block
3. Draft backbone could be extended with new tokens while target verifies previous ones

**But there are complications:**
- The DSpark attention mask (`create_dspark_attention_mask`) is designed for the fixed-block training pattern
- The Markov head samples autoregressively — new tokens depend on previous sampled tokens
- Confidence prediction is position-relative within the block

**Verdict:** Possible but requires a redesigned eval loop. The simplest V1 should keep the sequential draft-then-verify pattern.

---

## PART 8 — Rejected Speculation May Still Have Memory Value

### 8.1 The Hypothesis

Traditional view: rejected draft tokens are wasted work.

Our view: rejected tokens may still have predicted expert demand that overlaps with the actual continuation.

**Example:**
- Draft proposes token A at position t+3
- Target selects token B instead
- But at several layers, both A and B route through many of the same experts
- 60% of prefetched expert bytes were still useful

### 8.2 Separate Metrics

| Metric | Definition |
|--------|-----------|
| Token acceptance rate | Standard SD metric |
| Expert prefetch usefulness | `prefetch_useful_bytes / total_prefetch_bytes` |
| Prefetch waste | `prefetch_bytes_not_used / total_prefetch_bytes` |
| Cache pollution | `bytes_evicted_prematurely_due_to_prefetch` |
| Rejected-token reuse | `prefetch_bytes_from_rejected_positions_that_were_reused` |
| Stall removed | `time_saved_by_hot_experts / total_expert_wait_time` |

### 8.3 Measurement Approach

The existing routing trace infrastructure (`dsv4_trace_routing.cpp`) already captures per-token, per-layer expert IDs. To measure prefetch usefulness:

1. Record draft proposals (token IDs, confidence)
2. Record actual target expert selections
3. Simulate prefetch policy based on draft proposals
4. Measure overlap between prefetched and actually-needed experts

**This can be done offline with traces — no runtime changes needed.**

---

## PART 9 — Speculation Depth Should Depend on Cache State

### 9.1 Cases

| Case | Cache State | Implication for Speculation Depth |
|------|-----------|----------------------------------|
| A | All future experts HOT | Deep speculation adds only compute benefit |
| B | Many t+1 experts COLD, SSD idle | Deep speculation may be very valuable |
| C | SSD saturated, GPU cache full | Deep speculation increases pollution |
| D | Verification compute-bound | Memory-driven deep speculation helps little |
| E | Storage latency dominates | Deep speculation valuable if confidence acceptable |

### 9.2 Feedback Controller (Simple)

```
desired_depth = base_depth

if ssd_queue_depth > threshold:
    desired_depth -= 1   # already enough I/O work
if cache_miss_rate > threshold:
    desired_depth += 1   # need more look-ahead
if cache_eviction_rate > threshold:
    desired_depth -= 1   # polluting the cache
if compute_time >> io_time:
    desired_depth = min(desired_depth, base_depth)  # no memory benefit

clamp(desired_depth, min_depth, max_depth)
```

---

## PART 10 — First Implementable Controller Design

### 10.1 ADAPTIVE_DSPARK_V1

**Inputs:**
- `confidence[pos]` for pos = 1..7 (from DSpark confidence head)
- `speculative_token_ids[pos]` (from draft)
- `last_routing_data` (from last verification, via existing trace hooks)
- `cache_stats` (from `dsv4_expert_cache::get_stats()`)

**Algorithm:**
```
max_possible = 7
min_draft = 1 (always at least 1)
proposal_draft_tokens = max_possible

// Confidence gate (existing)
proposal_draft_tokens = min(
    proposal_draft_tokens,
    confident_prefix_length(confidence_logits, threshold)
)

// Marginal value gate (NEW)
for pos = 1 to proposal_draft_tokens:
    cold_bytes = estimate_cold_bytes(pos, speculative_token_ids[pos], last_routing_data)
    if cold_bytes < min_useful_lookahead:
        continue  // no memory benefit, but still may have compute benefit
    if prefetch_queue_bytes + cold_bytes > io_budget:
        proposal_draft_tokens = pos - 1
        break

return clamped(proposal_draft_tokens, min_draft, max_possible)
```

**Minimal code changes needed:**
1. `draft_ops.py:_confident_prefix_length()` — add marginal-value termination
2. `evaluator.py:_propose()` — pass cache/I/O state
3. New module: `expert_predictor.py` — predict future experts from speculative tokens

### 10.2 ADAPTIVE_DSPARK_V2 (Learned)

Replace heuristic marginal-value gate with a learned model:

```
input: [confidence[pos], token_id[pos], draft_hidden[pos], cache_state]
output: expected_value(pos) ∈ ℝ

accept all positions where expected_value > learned_cost_threshold
```

Train from historical traces using actual acceptance + prefetch-usefulness + stall data.

---

## PART 11 — Integration Points with Current DSV4 Runtime

### 11.1 Existing Infrastructure That Can Be Reused

| Component | Location | Purpose |
|-----------|----------|---------|
| `dsv4_expert_cache` | `moe-skip-f16/src/dsv4_expert_cache.cpp` | RAM cache + LRU + pread I/O |
| `dsv4_routing_state` | `moe-skip-f16/src/models/deepseek4.cpp` | Per-layer expert tracking |
| `dsv4_bind_callback` | Same file | Post-router expert load hook |
| `ffn_moe_topk` tensor | `llama-graph.cpp:1914` | Actual expert IDs per token |
| Routing trace tools | `v4-flash-hardware-aware/tools/*.cpp` | Expert routing capture |

### 11.2 Future Hooks (Interface Only — Do Not Implement Today)

**Expert prediction:**
```python
def predict_experts(
    draft_token_ids: List[int],          # speculative tokens [1..N]
    draft_hidden_states: torch.Tensor,   # [N, 5120]
    last_routing: Dict[int, List[int]],  # layer→[expert_ids] from last verify
    confidence: List[float],             # per-position confidence
) -> Dict[Tuple[int, int], float]:       # (layer, expert_id) → probability
```

**Prefetch submission:**
```python
def submit_prefetch(
    layer: int,
    expert_id: int,
    priority: float,        # from scoring function
    deadline: float,        # estimated deadline in seconds
    confidence: float,      # P(this expert is actually needed)
) -> None
```

**State queries:**
```python
def query_residency(layer: int, expert_id: int) -> str:
    # Returns "vram", "ram", "ssd", "cold"
    pass

def query_io_queue_state() -> IoState:
    # Returns queue depth, total bytes queued, estimated drain time
    pass

def query_gpu_cache_state() -> GpuCacheState:
    # Returns slot utilization, eviction rate, free slots
    pass
```

**Speculative expert candidates:**
```python
def submit_speculative_expert_candidates(
    candidates: List[ExpertCandidate],  # predicted future expert needs
) -> None:
    # Filters + prioritizes + queues prefetches
    pass
```

### 11.3 Integration Architecture

```
DSpark Proposal
    │
    ├──→ token_ids[1..N], hidden_states[1..N], confidence[1..N]
    │
    ▼
Expert Predictor (Part 2)
    │
    ├──→ per-layer expert probabilities for each future position
    │
    ▼
Priority Scorer (Part 5)
    │
    ├──→ scored expert candidates with deadlines
    │
    ▼
Prefetch Scheduler (Part 6)
    │
    ├──→ submits to dsv4_expert_cache
    │
    ▼
dsv4_expert_cache.get_expert() → RAM (warm) / SSD (cold)
    │
    ▼
dsv4_bind_callback → overwrites expert tensor slices → GPU compute
```

---

## PART 12 — Important Research Questions

### Q1: Is DSpark's speculation horizon fixed or adaptive?

**Fixed** at `block_size=7` by architecture. **Confidence-gated truncation** can reduce it per-proposal (if confidence head enabled). No dynamic adaptation to system state.

### Q2: What terminates a speculation sequence?

1. Maximum depth reached (7 tokens)
2. Confidence threshold crossed (confidence head predicts low acceptance prob)
3. Stop token encountered (post-verification)
4. Rejection sampling (standard SD termination)

### Q3: What confidence information is available per speculative position?

- Sigmoid of confidence head output → per-position acceptance probability (NOT cumulative)
- Cumprod of these → prefix acceptance probability (confidence that ALL positions up to k are accepted)
- Additionally: draft token log-probability, Markov-head derived probabilities

### Q4: Can future target expert IDs be known exactly before target execution?

**NO.** The target router's output depends on the actual target hidden state at each layer, which is not known until the target forward pass computes it.

### Q5: What is the cheapest plausible predictor?

**Token-conditioned transition model** (Part 2C) — a precomputed `[vocab_cluster × 256]` table per layer, indexed by speculative token ID and current expert set. Estimated cost: < 100 MB, O(1) lookup.

### Q6: Can expert prefetch happen while target verification is executing?

**YES, in C++ runtime.**

The current `dsv4_bind_callback` fires synchronously during graph evaluation. However, the mechanism could be extended:
- After router computes expert IDs for layer k, the callback runs
- The callback could also pre-emptively load experts for:
  - Remaining layers of the current token (using actual router output — already works)
  - Future tokens (using speculative predictions — new)

The SSD→RAM transfer (pread) can happen asynchronously while GPU computes other layers.

### Q7: Could drafting and verification eventually be pipelined?

**Theoretically YES** — separate models, separate KV caches, independent forward passes.

**Practical challenges:**
1. Target hidden states must be updated after each accepted token (dependency)
2. The DSpark attention pattern assumes the full block is processed at once
3. Markov head sampling is autoregressive — can't start drafting t+N+1 until t+N is sampled

**A simpler approach:** Pipelining is NOT needed for the V1 memory look-ahead benefit. The existing sequential pattern already provides up to 7-token look-ahead for prefetch.

### Q8: Is a byte-budgeted horizon technically feasible?

**YES.** The required components are:
1. Expert predictor (Part 2)
2. Residency query (Part 11)
3. I/O budget tracking (Part 4)

All are independently implementable. The horizon extension logic is a few lines of Python or C++.

### Q9: What state would an adaptive controller need?

| State | Source | Update Frequency |
|-------|--------|-----------------|
| Per-position confidence | DSpark confidence head | Per proposal |
| CPU/GPU compute backlog | Runtime timing | Per decode |
| VRAM expert cache hit rate | Future cache monitor | Per layer |
| RAM expert cache hit rate | `dsv4_expert_cache` stats | Per layer |
| SSD queue depth | Future I/O monitor | Continuous |
| Outstanding bytes | Future I/O monitor | Continuous |
| Cache eviction count | `dsv4_expert_cache` LRU | Per eviction |
| Accepted/rejected history | Runtime accumulator | Per token |

### Q10: Minimum code modification for V1 prototype?

**Python side (DeepSpec evaluator)** — small changes:
1. `draft_ops.py`: Modify `_confident_prefix_length()` to accept optional marginal-value termination (add ~20 lines)
2. `draft_ops.py`: New function `estimate_speculative_experts()` using token→expert lookup table (add ~50 lines)
3. `evaluator.py`: Pass cache state mock to proposal (add ~10 lines)

**C++ side (llama.cpp runtime)** — moderate changes:
1. `dsv4_expert_cache.cpp`: Add `prefetch()` method that reads into cache without immediate use (add ~30 lines)
2. `dsv4_expert_cache.cpp`: Add `query_residency()` (add ~15 lines)
3. `deepseek4.cpp` or new file: Expert predictor using speculative token IDs (add ~100 lines)
4. `dsv4_routing_state`: Extend `populate_slices()` to use predicted experts when available (add ~30 lines)

**Total V1: ~250 lines across 5 files.**

### Q11: Biggest technical risk

**The expert predictor may be inaccurate.**

If speculative tokens predict expert demand no better than random chance, then the prefetched experts will mostly be wasted, consuming cache slots and I/O bandwidth without benefit.

**Mitigation:** The fallback is to use the last accepted token's routing as the predictor (adjacent-token overlap of 25% is already valuable). Even random predictions would not harm correctness — only efficiency.

### Q12: What would falsify the entire idea?

**If NONE of these conditions hold:**
1. `P(expert_at_t+1 | expert_at_t) ≈ P(expert_at_t+1 | draft_token_t+1)` — speculative tokens add no information about expert transitions
2. Prefetch latency is fully hidden by compute (cache hits irrelevant)
3. SSD bandwidth exceeds aggregate expert demand (no stall to remove)
4. DSpark confidence head is uncorrelated with expert prediction accuracy

**Most likely falsification scenario:** The storage system achieves full bandwidth overlap with compute (no exposed stalls), making expert prefetch irrelevant.

**Easiest early falsification test:** Measure overlap between speculative-token-predicted experts and actual target experts using existing routing traces + replayed draft proposals. If overlap ≤ baseline (last-token routing), the idea has limited value.

---

## PART 13 — Prior Art / Existing Internal Work

### Found in Local Docs/Source

| Concept | Location | Status |
|---------|----------|--------|
| Expert cache LRU policy | `dsv4_expert_cache.cpp` | Done |
| Routing trace capture | `dsv4_trace_routing.cpp`, `trace_routing.cpp` | Done |
| Real routing locality analysis | `REAL_ROUTING_LOCALITY.md` | Done (17 tokens) |
| Cache simulation | `simulate_expert_cache.py` | Done (synthetic) |
| S5 compact GPU execution | `s5_compact_experts.cpp`, `test_streamed_expert_overwrite.cpp` | Done |
| Async SSD reader concept | `IMPLEMENTATION_PLAN.md` Phase 4 | Planned |
| `--gpu-core-only` flag | `common/arg.cpp` | Done |
| Expert address index | `tools/build_expert_index.py` | Done |
| Token-level expert working set | `REAL_ROUTING_LOCALITY.md` | Measured |
| Expert reuse distance | `REAL_ROUTING_LOCALITY.md` | Measured |

### NOT Found in Local Docs

| Concept | Status |
|---------|--------|
| Adaptive speculative depth | Not addressed |
| Speculation-aware expert prefetch | Not addressed |
| Confidence-driven expert prediction | Not addressed |
| Byte-budgeted speculation | Not addressed |
| Deadline-aware expert scheduling | Not addressed |
| Rolling/pipelined speculation | Not addressed |
| Rejected-token prefetch reuse | Not addressed |
| Cache-state-dependent horizon control | Not addressed |

All the concepts in this research document are **new** — no prior work found in the local codebase or docs.

---

## Appendix: File Reference Map

### DSpark Python Implementation

| File | Purpose | Key Functions |
|------|---------|---------------|
| `deepspec/eval/dspark/draft_ops.py` | Draft generation | `build_dspark_proposal()`, `forward_dspark_draft_block()`, `_confident_prefix_length()` |
| `deepspec/eval/dspark/evaluator.py` | Speculative eval loop | `_propose()`, `_update()`, `generate_one_sample()` |
| `deepspec/eval/dspark/confidence_head.py` | Confidence calibration | `ConfidenceHeadRecorder.observe()` |
| `deepspec/eval/base_evaluator.py` | Core SD loop | `generate_decoding_sample()`, `verify_draft_tokens()` |
| `deepspec/modeling/dspark/qwen3/modeling.py` | DSpark model | `forward()`, `sample_draft_tokens()`, `predict_confidence_step()` |
| `deepspec/modeling/dspark/common.py` | DSpark shared | `AcceptRatePredictor`, `extract_context_feature()` |
| `deepspec/modeling/dspark/markov_head.py` | Markov chain | `VanillaMarkov.sample_block_tokens()`, `GatedMarkovHead` |
| `deepspec/utils/sampling.py` | Sampling | `logits_to_probs()`, `sample_residual()` |

### C++ Runtime (Expert Cache)

| File | Purpose | Key Structures |
|------|---------|----------------|
| `moe-skip-f16/src/dsv4_expert_cache.cpp` | Expert cache | `dsv4_expert_cache::load_expert()`, LRU eviction, pread |
| `moe-skip-f16/include/dsv4_expert_cache.h` | Cache interface | `get_expert()`, `get_stats()`, cache types |
| `moe-skip-f16/src/models/deepseek4.cpp` | Model integration | `dsv4_routing_state`, `dsv4_bind_callback`, `populate_slices()` |
| `llama.cpp/src/llama-graph.cpp` | MoE graph builder | `build_moe_ffn()`, `build_lora_mm_id()` |

### Routing Trace

| File | Purpose |
|------|---------|
| `v4-flash-hardware-aware/tools/dsv4_trace_routing.cpp` | Expert routing tracer |
| `v4-flash-hardware-aware/tools/s4_streaming_hook.cpp` | Streaming expert load |
| `v4-flash-hardware-aware/tools/trace_routing.cpp` | Alternative tracer |

### Cache Simulator

| File | Purpose |
|------|---------|
| `v4-flash-hardware-aware/tools/simulate_expert_cache.py` | Cache simulation (LRU, Freq-LRU) |
| `v4-flash-hardware-aware/tools/analyze_expert_trace.py` | Reuse distance, popularity |

### Research Docs

| File | Content |
|------|---------|
| `v4-flash-hardware-aware/docs/REAL_ROUTING_LOCALITY.md` | Real trace analysis |
| `v4-flash-hardware-aware/docs/128K_MEMORY_LEDGER.md` | Memory budget |
| `v4-flash-hardware-aware/docs/CACHE_SIMULATION.md` | Cache simulation results |
| `v4-flash-hardware-aware/docs/IMPLEMENTATION_PLAN.md` | S1-S5 plan |
| `v4-flash-hardware-aware/docs/DEEPSEEK4_MOE_GRAPH.md` | MoE intervention points |
| `v4-flash-hardware-aware/docs/OPTIMIZATION_LADDER.md` | Full optimization stack |

---

## Appendix B: Quantitative Measurement Results

### B.1 Oracle Ceiling (from `dsv4-16token-trace.jsonl`)

| Horizon | Avg Unique Schedulable | MiB | Oracle Cumulative Value |
|---------|----------------------|-----|----------------------|
| 1 | 258 | 2,374 | 1.0× token-worth |
| 2 | 516 | 4,747 | 2.0× |
| 3 | 774 | 7,121 | 3.0× |
| 4 | 1,032 | 9,494 | 4.0× |
| 5 | 1,290 | 11,868 | 5.0× |
| 6 | 1,548 | 14,242 | 6.0× |
| 7 | 1,806 | 16,615 | 7.0× |

Key finding: **No saturation** — each additional horizon token reveals ~258 new schedulable experts.

Marginal new experts at each horizon vs preceding position: ~160/258 (~62%) regardless of horizon.

### B.2 Baseline Predictors — CORRECTED (Strict Temporal Holdout)

**WARNING: The originally reported F1=0.72-0.82 was LEAKED (training on test data).**
Under strict temporal holdout (first 70% train, last 30% test), the true values are:

| Predictor | Lookahead=1 | Lookahead=4 | Notes |
|-----------|-------------|-------------|-------|
| B1: t-1 same-layer | 0.292 | 0.192 | No training — unaffected |
| B2: union t-1..t-2 | 0.277 | 0.174 | No training |
| M2: Transition top-6/layer | **0.356** | **0.226** | Strict temporal holdout |
| M2: Transition top-12/layer | 0.325 | 0.241 | Wider, lower precision |
| M2: Transition top-24/layer | 0.266 | 0.206 | Too wide — waste dominates |

**Corrected interpretation:** The transition predictor beats B1 by ~22% relative (F1 0.356 vs 0.292) at lookahead=1. Modest but real. The original 0.719 result was ~50% leakage from training on test data.

Per-layer F1 ranges from 0.167 (worst) to 0.600 (best). Median layer F1: 0.367.
The predictor does not uniformly benefit all layers.

### B.3 Expert Churn and Burstiness

| Metric | Value |
|--------|-------|
| Token-to-token churn | ~62% new experts |
| Mean burst length | 1.53 tokens |
| Single-token bursts | 77.2% |
| Persist 2+ tokens | 22.8% |
| Adjacent Jaccard | 0.234 |
| Jaccard at gap=7 | 0.126 |

### B.4 Wrong-Token / Right-Memory

Even a completely wrong speculative token provides:
- ~23.4% Jaccard overlap with the actual token's experts
- ~95/258 experts match = ~874 MiB useful prefetch per token
- At depth 7: still ~12.6% Jaccard overlap (~325 MiB useful)

**Memory usefulness of speculation significantly exceeds token accuracy.**

### B.5 Cache Cap vs Oracle

| Cache Tier | Capacity (GB) | Useful Horizon Before Saturation |
|-----------|---------------|--------------------------------|
| VRAM (hot, ~300 exp) | 2.8 | ~1 token |
| RAM (warm, ~1,260 exp) | 11.6 | ~5 tokens |
| Combined | 14.4 | ~6 tokens |

The binding constraint is cache capacity, not future-information availability.

### B.6 Layer Diversity

| Layer Group | Unique Experts/17 tokens | Predictability |
|------------|------------------------|---------------|
| Early (0-3) | 72-77 | Low (diverse) |
| Mid (4-35) | 28-56 | Medium |
| Late (36-42) | 42-61 | Medium |
| Layer 26 (lowest) | 28 | High |
| Layer 39 (highest) | 61 | Low |

Different layers have very different predictability, suggesting per-layer prediction widths.

### B.7 Per-layer Expert Persistence

Top persistent expert at each layer used in 14-16/17 tokens (82-94%). ~9 per-layer experts are "always-hot" candidates for permanent VRAM residency.

---

## Appendix D: Research Claim Map

| Claim | Level | Status | Evidence |
|-------|-------|--------|----------|
| "Transition predictor beats t-1" | L1 | **CONFIRMED** | M2 F1=0.356 vs B1 F1=0.292 (strict holdout) |
| "Original 0.72-0.82 F1 was leaked" | L0 | **CONFIRMED** | ~50% leakage from training on test data |
| "Within-token cross-layer prediction" | L1 | **FALSIFIED** | F1≈0.02 — no signal |
| "Wrong-token/right-memory ~23% Jaccard" | L1 | **CONFIRMED** | Unaffected by leakage, 10× random baseline |
| "Oracle horizon shows no saturation" | L1 | **CONFIRMED** | Each horizon adds exactly 258 new experts |
| "Cache capacity is binding constraint" | L0 | **CONFIRMED** | ~6 tokens fills VRAM+RAM |
| "Ensembles beat single predictor" | L1 | **FALSIFIED** | Union reduces F1, weighted ensemble ties |
| "Online adaptation closes the gap" | L1 | **PARTIAL** | Prompt-local F1=0.253 vs global 0.356 |
| "DSpark token-conditioned prediction" | L0 | **UNTESTED** | Requires live DSpark traces |
| "T1 training fits on 16 GB VRAM" | L0 | **ESTIMATED** | ~5.5-6 GB for head/LoRA |
| "Adaptive horizon improves throughput" | L0 | **UNTESTED** | Requires live runtime implementation |
| "Multi-branch memory prediction" | L0 | **UNTESTED** | Requires DSpark logits API |

## Appendix C: Key Files for Future Work

| File | Change Needed | Estimated Lines |
|------|--------------|----------------|
| `draft_ops.py:_confident_prefix_length()` | Add marginal-value termination | +20 |
| `draft_ops.py` | New `estimate_speculative_experts()` using transition table | +50 |
| `evaluator.py:_propose()` | Pass cache/I/O state | +10 |
| `dsv4_expert_cache.cpp` | Add `prefetch()`, `query_residency()` | +45 |
| `deepseek4.cpp` or new file | Expert predictor using speculative tokens | +100 |
| `dsv4_routing_state` | Use predicted experts in `populate_slices()` | +30 |
| **Total V1** | | **~255** |

### Tools Created

| File | Purpose |
|------|---------|
| `tools/dspark_lookahead_sim.py` | Oracle ceiling, baselines, scheduler comparison |
| `docs/DSPARK_ORACLE_LOOKAHEAD.md` | Oracle ceiling report |
| `docs/DSPARK_ADAPTIVE_CONTROLLER.md` | Controller design |
| `docs/DSPARK_TRAINING_ROADMAP.md` | Training evolution plan |
| `docs/DSPARK_LOOKAHEAD_SIMULATOR_SPEC.md` | Simulator spec |
| `docs/DSPARK_MEMORY_LOOKAHEAD_TODO.md` | Implementation milestones |
| `docs/DSPARK_ROUTING_PREDICTABILITY_MAP.md` | Full predictability map (R21-R32) |
| `docs/DSPARK_INTRA_TOKEN_LOOKAHEAD.md` | Intra-token cross-layer — NEGATIVE result |
| `docs/DSPARK_EXPERT_CLUSTERING.md` | Expert co-occurrence / bundles |
| `docs/DSPARK_PREDICTOR_ATTRIBUTION.md` | Oracle decomposition |
| `docs/DSPARK_FUTURE_AWARE_CACHE.md` | Cache + admission policies |
| `docs/DSPARK_MULTIBRANCH_LOOKAHEAD.md` | Multi-branch speculation design |
| `docs/DSPARK_ONLINE_ADAPTATION.md` | Causal online adaptation results |
| `tools/dspark_predictability_map.py` | Comprehensive R21-R32 analysis script |
| `tools/dspark_strict_holdout.py` | Strict corrected holdout evaluation |


---

# PHASE 2 — DSpark as a Predictive Memory Controller (static continuation)

> Supersedes nothing; extends the research to the new primary question:
> "Can speculative model state control the heterogeneous memory hierarchy so
> that future exact-inference stalls are reduced, including when the
> speculative token itself is wrong?" Objective: correct useful tokens/sec.
>
> New/updated docs: DSPARK_MEMORY_CONTROLLER.md (synthesis),
> DSPARK_MEMORY_ACCEPTANCE.md, DSPARK_BELADY_ORACLE.md,
> DSPARK_ADAPTIVE_HORIZON.md, DSPARK_DEADLINE_PREFETCH.md,
> DSPARK_EXPERT_UNION.md, DSPARK_PAGECACHE_CONTROL.md,
> DSPARK_HARDWARE_ORACLE.md, DSPARK_30TPS_FEASIBILITY.md.
> Tools: tools/dspark_memcommon.py + 7 new analyzers; results in
> results/phase2/*.json; tests: tests/test_dspark_mem_common.py (10 tests).

## P2.1 New MEASURED hardware facts

- **Q8 expert sizes are uniform**: 4,456,448 B/tensor, 12.75 MiB per
  (layer, expert); 137.06 GiB total expert weights (expert_index_0731_v2).
  The "assume non-uniform" caution resolved: uniform (verified 33,024/33,024
  entries). Working set: 258 experts = 3.21 GiB/token (Q8), 2.32 (IQ3).
- Trace 0731_q8_128tok.tr is TRUNCATED (1 token, 16 layers) — unusable;
  all Phase-2 curves use dsv4-16token-trace.jsonl (17 tokens).

## P2.2 New findings (summary; details in per-topic docs)

1. Miss-conditioned prediction: transition-predictor F1 on RAM-miss subset
   = 0.107–0.172 (4–12 GiB) vs 0.354 all-experts. The locked F1 is NOT
   stronger where it matters. t-1 retention has zero miss-subset value
   (always resident). [MEASURED]
2. RAM misses ≈ first-appearance churn: last-7-token history covers 1.6%
   of the 12 GiB miss subset. Recent-history predictors are blind to the
   expensive tail. [MEASURED]
3. Belady bound: 0.574 hit rate at 8 GiB vs LRU 24 GiB; LFU best practical
   (0.531 @12 GiB). LRU wrong-eviction: 24–39% of evictions reuse later;
   lost hits = Belady−LRU gap exactly (validated). Learned next-use
   eviction (11-token train) ≤ LFU — negative result; use recency
   fallback or DSpark look-ahead instead. [MEASURED]
4. Union: F_union 1.79 @K=7; 64% of union experts single-use; best union
   predictor = keep-previous-window (0.587 byte recall vs 0.339 freq).
   Union verification load still 8.4× over the 30 t/s H2D budget. [MEASURED]
5. Deadlines/QD: in the bandwidth-saturated regime (Q8 + 300-VRAM +
   12 GiB RAM + QD4), every controller — including the clairvoyant oracle
   — stalls identically; oracle cuts cold demand 8.5× with zero stall
   benefit. QD4→16 = −23% stall. Prefetch value requires headroom;
   headroom requires byte reduction. [MEASURED, model-dependent]
6. Tiered actions: VRAM promotion needs p_need ≳ 0.27 (Q8); transition
   predictor achieves 0.11–0.20 miss-conditioned → only WARM_PAGECACHE
   passes. Page-cache hints must be batched at expert granularity
   (per-4K spam = 69% overhead). Coalescing beyond 64 MiB gaps is
   counterproductive (17.7× overread at 256 MiB). [DERIVED + MEASURED]
7. 30 t/s budget: 92 MiB SSD / 213 MiB H2D per token → 94–95% byte hit
   rate required at the SSD boundary (Q8); churn must fall 17× (SSD) and
   8× (H2D). Union does not fix sustained bandwidth. [DERIVED]

## P2.3 What survived and what died (Phase 2)

| Claim | Verdict |
|-------|---------|
| Prediction value measured in F1 | REPLACED by stall-avoided/byte |
| Miss-conditioned prediction improves on expensive subset | FALSIFIED (0.107–0.172) |
| t-1 retention as prefetch | FALSIFIED as prefetch (resident); valid as eviction baseline |
| Learned next-use eviction from short history | FALSIFIED (≤ LFU); recency fallback required |
| Byte/union leverage on bandwidth | CONFIRMED but insufficient alone |
| VRAM promotion threshold p ≥ 0.27 | DERIVED (design constraint) |
| Saturated-regime prefetch value | ZERO (measured, model-dependent) |
| DSpark token/hidden-state conditioning | UNKNOWN — live only (M1+) |

## P2.4 Immediate next steps (static complete; live pending)

1. Collect ≥128-token Q8 routing trace (fix .tr truncation) — unlocks all
   curves at scale (M1 prerequisite).
2. Measure the real SSD QD curve and fadvise costs on the model files.
3. When the main experiment finishes: M1 co-collection with
   DSPARK_LOOKAHEAD_TRACE_V1, then M2–M4 per the TODO; run M9–M18
   pre-registered experiments against the hardware oracle ceiling.
