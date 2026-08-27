# V4 Negative-Knowledge Ledger

**Purpose:** permanently record rejected mechanisms with hypothesis / experiment /
result / scope / falsifier, so future sessions do not re-test them and do not mistake
them for viable paths. Every entry is backed by a deterministic CPU-only simulation on
the 2,836-token real-trace corpus (`results/v4_expert_cache_static/`) unless labeled
otherwise.

| # | hypothesis | experiment | result | scope | falsifier |
|---|---|---|---|---|---|
| N1 | Frequency-aware eviction improves HOT hit rate on V4 traffic | Policy C vs B (S=6, 4 GiB WARM), 2,836 tokens | HOT 12.5% vs LRU 19.9% — **hurts** (bursty traffic) | HOT tier eviction | any workload where C ≥ LRU by >2pp |
| N2 | Pure reuse-distance (transition-successor) eviction improves HOT | Policy D vs B | HOT 13.2% vs 19.9% — **hurts** | HOT tier eviction | D ≥ LRU by >2pp |
| N3 | Naive full-token prefetch into WARM saves SSD bytes | E vs B at WARM 256 MB–4 GiB | byte-NEGATIVE at every size (−26% to −31% SSD) | WARM < 4 GiB | ssd_reduction > 0 at any WARM |
| N4 | Prefetch into a 256 MB WARM is viable | prefetch sweep | a full-token request set = 3.29 GiB ≫ 20-expert WARM → thrash; waste exceeds savings | WARM < 1 GiB | — |
| N5 | At 4 GiB WARM, prefetch matters at the byte level | prefetch sweep + Re-Spark oracle | no-op: cold already 0.28%; oracle saves only ~3 MiB/t of a 9 MiB base | WARM ≥ 4 GiB | >10% SSD reduction at 4 GiB |
| N6 | Cross-layer expert prediction is useful | locality (2,836 tokens) | set overlap at ΔL=1..42: 0.022–0.028 (noise) — re-falsified | any prefetch policy | overlap > 0.10 at any ΔL |
| N7 | 6 slots/layer is an adequate HOT working set | working-set curves | 20% hot hits; 64 slots/layer needed for 88% | HOT geometry | runtime hot ≥ 40% at S=6 |
| N8 | Semantic cache hit == useful work | useful accounting | 42.6% immediate reuse; a hit that is evicted before reuse is not useful; raw-hit metrics overstate | metrics | — |
| N9 | O_DIRECT is beneficial for expert reads | io_microbench (bounded, real spans) | 892 MB/s vs 1,376 MB/s buffered — **slower**; matches prior A/B/C | this Btrfs/NVMe | O_DIRECT > buffered on the same spans |
| N10 | Sorted-coalesced cold reads give large gains | io_microbench best-case contiguous sample | +26% (1,376 → 1,732 MB/s), NOT the 7.6× of a warm page cache; real sets are 2.4% adjacent | cold reads | — |
| N11 | Scheduler/sync cost is dominated by per-expert sync | static audit + sched_sim | 43 sync boundaries/token are per-LAYER (not per-expert); the transaction is batching layers, not experts | scheduler | a run with 43 syncs ≈ 1 sync |
| N12 | The hybrid policy's gain comes from frequency or prediction alone | hybrid ablation | recency-only = LRU (19.95%); freq-only 12.5%, pred-only 7.0%; gain only as tiebreakers (0.4/0.4/0.2 → 22.0%) | policy weights | any single-term ≥ hybrid |
| N13 | Re-Spark union prefetch dramatically cuts SSD at 4 GiB WARM | Re-Spark oracle sweep | 100% precision but bounded: −45% of a 9 MiB base at 4 GiB; −8% at 1 GiB | bytes | >30% absolute cold reduction at 4 GiB |
| N14 | DSEI metadata lookup is a bottleneck | lookup microbench (51.6k queries) | 39–65 µs/token across dict/flat/batched — negligible vs ms-scale decode | metadata | lookup > 1 ms/token |
| N15 | Prior 17-token locality (F1@6 0.368, 23 eff. experts/layer) transfers at scale | 2,836-token metrics | F1@6 0.309, 237 unique/layer, top-6 cum 23.6% — weaker and longer-tailed | locality | — |

**Cross-cutting scope note:** all results are measured on this 2,836-token trace corpus
(6 prompts × 256–512 tokens + 2 short traces); they are hypotheses until revalidated at
runtime (§28 of the mega directive). Synthetic traces were not used except for
data-structure stress tests (planner/sched_sim), and remain labeled synthetic.

| # | hypothesis | experiment | result | scope | falsifier |
|---|---|---|---|---|---|
| N16 | Non-displacing windowed prefetch is byte-positive | NDP threshold sweep (budget 20%, window 12) | zero crossing ≈ **640 MiB WARM**; +40 MiB/t @768 MiB, +84 @1 GiB, +75 @2 GiB, +3 @4 GiB (saturated) | this trace corpus; budget/window shift the crossing | net benefit < 0 at 1 GiB on any new trace |
| N17 | WARM sizing dominates HOT slots on the Pareto frontier | HOT {1..64} × WARM {256 MiB..8 GiB} grid, 70 configs | HOT=1/WARM=4 GiB (4.5 GiB total) → 0.21% cold; HOT=4/WARM=2 GiB → 32.4%; HOT slots are a latency/VRAM trade, not a bandwidth lever at WARM ≥ 2 GiB | LRU policy, 2,816-token corpus | any config where HOT≥8 + WARM=2 GiB beats HOT=1 + WARM=4 GiB on cold% at equal memory |
| N18 | Hybrid weights are a sharp optimum | robustness sweep 0.2/0.6/0.2 … 0.6/0.2/0.2 | **broad plateau 22.4–23.0%** over recency 0.3–0.5; eviction counts ±1.7% | S=6, 4 GiB WARM | >1.5pp spread across the plateau on new data |
| N19 | Python request planner is hot-path viable | planner validation (2,816 tokens) | planning 1.55 ms/token > modeled I/O savings 0.49 ms/token (3.2×) | Python prototype | native implementation ≤ 50 µs/token (expected; then planning is negligible) |
| N20 | Re-Spark union prefetch substantially cuts SSD at 4 GiB WARM | Re-Spark oracle sweep k=0..8 | 100% precision but −45% of a 9 MiB base at 4 GiB; −8% at 1 GiB | bytes only | >30% absolute cold reduction at 4 GiB |
