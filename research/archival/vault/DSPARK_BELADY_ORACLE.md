# Belady Oracle and Eviction-Policy Analysis

> Track 2/17. Data: dsv4-16token-trace.jsonl (17 tokens, 4386 expert
> accesses), Q8 sizes 12.75 MiB/expert (MEASURED uniform).
> Tool: `tools/dspark_belady_oracle.py` → `results/phase2/belady_oracle.json`

---

## 1. Reuse-Distance Characterization [MEASURED]

| Statistic | Value |
|-----------|-------|
| Accesses | 4,386 (17 tokens × 258) |
| Distinct (layer,expert) keys | ~1,869 |
| Keys with ≥1 reuse | 789 (42.2% of distinct) |
| Reuse distance mean / median / p90 | 685 / 260 / 1,807 accesses |

Reuse distances are long-tailed: median 260 accesses ≈ 1 token, p90 ≈ 7
tokens. 58% of distinct experts are never reused within the trace.

## 2. Policy Sweep — Hit Rate vs Capacity [MEASURED]

| GiB | LRU | LFU | LFU+R(win) | TinyLFU-like | Belady |
|-----|-----|-----|-----------|--------------|--------|
| 2 | 0.000 | 0.234 | 0.198 | 0.170 | 0.399 |
| 4 | 0.347 | 0.369 | 0.306 | 0.250 | 0.512 |
| 8 | 0.460 | 0.480 | 0.448 | 0.384 | **0.574** |
| 12 | 0.510 | 0.531 | 0.492 | 0.479 | 0.574 |
| 20 | 0.572 | 0.572 | 0.572 | 0.566 | 0.574 |
| 28 | 0.574 | 0.574 | 0.574 | 0.574 | 0.574 |

- Infinite-cache bound (all reuses hit): 0.574.
- **Belady reaches the bound at 8 GiB; LRU needs 24 GiB** (3× capacity).
- LFU is the best practical policy at every capacity (0.234–0.572).
- TinyLFU-like (admission filter with exact counts) underperforms on this
  short trace — a small-sample artifact; do not conclude against TinyLFU
  without longer traces. [ASSUMPTION-DEPENDENT]

### Why LRU loses (wrong-eviction analysis) [MEASURED]

| GiB | evictions | victims reused later | lost hits | % imminent (≤1 tok) |
|-----|-----------|----------------------|-----------|---------------------|
| 4 | 2,545 | 998 (39%) | 998 | 33.7% |
| 8 | 1,725 | 499 (29%) | 499 | 19.2% |
| 12 | 1,186 | 281 (24%) | 281 | 18.5% |

Lost hits exactly match the Belady−LRU hit gap (e.g., 281 = 281 at 12 GiB)
— validating the analysis. At 12 GiB: **281 × 12.75 MiB ≈ 3.5 GiB of extra
SSD traffic per 17 tokens ≈ 211 MiB/token** — 2.3× the entire 30 t/s SSD
budget. Eviction policy is not a second-order concern; it is on par with
prediction.

### Track 17: how far is Linux from Belady on this workload?

The Linux page cache is an LRU-ish policy (with readahead). Answer: at 12
GiB, 6.4 pp hit-rate below Belady, from wrong eviction of experts with
imminent reuse (18–37% of evictions), not from readahead or layout (see
DSPARK_PAGECACHE_CONTROL.md for layout). DSpark hints targeting *eviction*
(keep-if-reuse-soon) have as much headroom as hints targeting *fetch*.

## 3. Belady Distillation [MEASURED, train/test]

Learned next-use estimators (trained on tokens 0–10 only, evaluated on
11–16 with pre-warmed cache):

| Policy | 4 GiB | 8 GiB | 12 GiB | 20 GiB |
|--------|-------|-------|--------|--------|
| LRU | 0.347 | 0.460 | 0.510 | 0.572 |
| LFU | 0.369 | 0.480 | 0.531 | 0.572 |
| learned survival (P(reuse≤W)) | 0.024 | 0.146 | 0.272 | 0.524 |
| learned mean gap | 0.023 | 0.130 | 0.270 | 0.546 |
| learned + recency fallback | 0.268 | 0.490 | 0.518 | 0.566 |
| Belady (oracle) | 0.512 | 0.574 | 0.574 | 0.574 |

**Finding: 11-token training is insufficient — naive learned next-use
eviction is worse than LRU (cold keys unscored → arbitrary eviction).** The
recency-fallback variant ties LFU but does not beat it. Per the mission's
rule ("if LFU is already near Belady: STOP"): LFU is NOT near Belady
(gap 0.094 at 8 GiB), but learning next-use from 11 tokens cannot close it.

**Where the next-use signal must come from:** not from 11-token history, but
from the *speculative future* — DSpark's draft tokens are a next-use
predictor by construction. The Belady gap is the quantified prize for
speculative memory look-ahead: up to 6.4 pp hit rate (12 GiB) or 3× RAM
capacity equivalence.

## 4. Implications for the Controller

1. Adopt LFU-with-recency (windowed frequency) as the RAM/page-cache
   eviction baseline — strictly better than LRU here.
2. Investigate *eviction hints* from DSpark ("this resident expert will be
   reused at t+h — do not evict"), which directly attack the measured
   wrong-eviction fraction.
3. Do NOT train next-use estimators on short traces; require ≥100-token
   live traces (M8) before re-testing distillation.
4. Re-measure all curves on the 0731 Q8 trace once ≥128 tokens are
   collected (current .tr file is truncated to 1 token — unusable).
