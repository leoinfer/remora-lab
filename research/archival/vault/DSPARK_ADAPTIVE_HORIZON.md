# Adaptive Speculation Horizon — Static Results

> Track 4. Tool: `tools/dspark_adaptive_horizon.py` →
> `results/phase2/adaptive_horizon.json`
> Causal transition predictor (trained only on tokens < t), RAM-miss
> conditioning at 12 GiB, Q8 sizes.

---

## 1. Prediction Value by Horizon [MEASURED, 17-token trace]

Cumulative (union of horizons 1..H), miss-conditioned:

| H | useful MiB | wasted MiB | value/byte |
|---|-----------|-----------|------------|
| 1 | 446 | 15,848 | 0.027 |
| 2 | 638 | 19,112 | 0.032 |
| 3 | 853 | 21,112 | 0.039 |
| 4 | **1,033** | **22,096** | **0.045** |
| 5 | 1,046 | 22,670 | 0.044 |
| 6–7 | 1,046 | 22,670 | 0.044 (no additional windows) |

- Value-per-byte peaks at **H=4**; H=5 adds 13 MiB useful for 574 MiB waste.
- Deeper horizons add little on this trace; the h=5..7 windows are also the
  noisiest (fewer valid windows near trace end).
- Miss-conditioned P(needed) per horizon: **0.20 (h=1), 0.12 (h=2), 0.15
  (h=3), 0.11 (h=4)** — the predictor's non-resident predictions are right
  only 11–20% of the time.

## 2. Fixed vs Adaptive [MEASURED + DERIVED]

| Policy | useful MiB | waste MiB | vpb |
|--------|-----------|-----------|-----|
| fixed H=1 | 446 | 15,848 | 0.027 |
| fixed H=2 | 638 | 19,112 | 0.032 |
| fixed H=4 | 1,033 | 22,096 | 0.045 |
| fixed H=7 | 1,046 | 22,670 | 0.044 |
| adaptive (any threshold) | 446 | 15,848 | 0.027 |

The V1 adaptive rule ("extend while next horizon's vpb ≥ threshold")
degenerates to H=1 because the *per-horizon* vpb (0.008–0.15) never exceeds
the tested thresholds — while the *cumulative* vpb keeps rising to H=4.
Calibration lesson: the rule must use **marginal cumulative** value with a
threshold in units of "useful ms per prefetched GiB", and must be calibrated
on live data (M11). Pre-registered, not yet conclusive.

## 3. Design Guidance (V1 controller)

1. **Horizon band H ∈ [2,4]** is the static optimum zone on this workload;
   H=1 under-covers, H≥5 adds waste without value.
2. Horizon should be *score-gated* rather than *fixed*: per-horizon
   prediction score (transition count) is the runtime proxy; only extend
   to h while the h-th horizon's score mass remains above a floor
   (calibrate floor at M11).
3. The byte-budget view (Track 4 of the mission) is the correct
   formulation: extend while predicted useful cold bytes ≤ remaining
   I/O budget (SSD 92 MiB/token at 30 t/s) AND predicted marginal
   value/byte > threshold.
4. With A_mem_bytes ≈ 3–4.5% and 92 MiB/token of SSD budget, the V1
   controller can afford ~2–4 MiB of useful prefetch per token — i.e.,
   **~1 expert per 2 tokens** at today's predictor quality. Expectation
   management: the memory controller's first live wins will be small
   until predictor quality (or DSpark conditioning) improves.

## 4. Falsification

If live DSpark conditioning does not lift miss-conditioned P(needed)
above ~0.35 (which would make VRAM promotion viable at the 0.27
threshold), the adaptive-horizon machinery should be shelved in favor of
eviction-policy work (Belady gap) and byte reduction.
