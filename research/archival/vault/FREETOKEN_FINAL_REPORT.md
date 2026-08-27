# FT2 DECISIVE VERDICT

FREETOKEN LRU: **HAS HEADROOM vs ORACLE — BUT THE HEADROOM IS CAUSALLY LOCKED**
BEST CAUSAL PREDICTOR: LFU (+4–5pp) / persist_evict (+0.0–0.3pp) — all capture <10% of the Belady gap
ORACLE GAP: 2.9x LRU hit-rate at 9–19% capacity; 1.9x at 37.5%; ->1.15x near pool size
128GB FOR DSV4: **INSUFFICIENT** for FreeToken-native ds_fp4 artifact (16.1 GiB deficit); **MARGINALLY_SUFFICIENT** for <=110GiB re-quantized artifacts only
192GB: holds native artifact (+48GiB) but unofficial on AM4/Zen2; ~zero throughput delta while VRAM cap unchanged
CURRENT 32GB: three-tier NOT interactive-viable (NVMe miss budget violated 7–20x); 0.8–4 t/s band

## Decisive numbers (byte-exact unless noted)
- Expert slot ds_fp4 = 13,369,344 B; pool 43x256 = 11008 slots = 137.06 GiB; non-expert resident 9.49–9.95 GB; routed all-miss 3.45 GB/token; shared+attn+embed active stream 9.24 GB/token.
- Our VRAM expert cache after reservations: ~333–355 slots = 3.0–3.2% of pool.
- Pin budgets (DIMM−7GiB): 25/57/89/121/185 GiB @32/64/96/128/192 → deficits −112/−80/−48/−16/+48 vs pool.
- MEASURED_BY_US: RAM read 29.4 GB/s contig (16.8 strided); NVMe seq O_DIRECT 2.90 GB/s; rand 12.75MiB qd1 p50 1.89 ms p90 2.89 ms. PCIe pinned-gather modeled 22–28 GB/s (PCIe4 x16 analog; unmeasured — no GPU window).
- q* on our box: threshold C>2·Π FAILS (29.4<50) → offload backend; CPU coexecution near-useless here.
- Replay (real traces): LRU/Belady/FIFO/LFU/persist_evict/hotsplit at equal caps; bursts are FULL-footprint (p50 misses/token = k at ≤20% cap).
- Calibration: model reproduces RTX5090 22–25 t/s within bounds (Fig4b anchor 39% miss@~14% cap) AND our historical 2.7–2.8 t/s (NVMe-bound) with one resource model.

## Throughput bands (RX 9060 XT, DSV4-Flash class)
A 32GB+3-tier: 0.8–4 t/s | B 64GB: 1.5–6 | C 96GB: 3–10 | D 128GB+≤110GiB artifact: 8–13 (central ~10) | E 192GB native: 8–11 | predictive uplift: ≤ +0–2 t/s (unproven).

## Answers (§36)
1 137.06 GiB pool (147.17 GB). 2 Shared/dense ≈9.5–10 GB resident + 9.24 GB/token streamed. 3 Pool-sized pinned RAM + margins. 4 Their hosts carry 180–512 GiB. 5 Ours: 32 GB → storage-bound; VRAM cache starved to 3.2%. 6 LRU hit @19% cap ≈16% (pretraining-trace pessimistic; real-serving anchor much higher: 61%@11%). 7 Belady @19% ≈47%. 8 Best causal ≈LRU+0–5pp. 9 <10% captured. 10 Sweet spot unreachable (>40% of pool needed; we have 3.2%). 11 RAM wall 29.4 GB/s (only binds CPU-lane; lane unused). 12 PCIe ~25 GB/s = primary miss-path wall. 13 CPU experts: not useful on Zen2+DDR4 (threshold fails). 14 NVMe budget @20 t/s: ≤10.9 cold misses/token (conservative). 15 Three-tier: dead at 32GB for interactivity; viable design rule = prediction-driven with hard budgets. 16 22–25 t/s reconstructed ✓. 17 32GB: 0.8–4. 18 128GB: 8–13 w/ smaller artifact; INSUFFICIENT otherwise. 19 192GB: 8–11. 20 Predictive residency: demoted until better signals exist. 21 128GB: worth it ONLY paired with ≤110GiB artifact plan. 22 192GB: not materially faster; buys artifact freedom; platform-risky. 23 First port: byte-exact placement runtime (banks+pin-after-fill+slot-LRU), not predictors. 24 Biggest FT weakness: no third tier + no trace tooling + idle-only elasticity. 25 Biggest ours: three-tier HAR with hard miss budgets + R4X re-quant to shrink pools below RAM ceilings.
