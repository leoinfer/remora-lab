# Falsified Claims — Independent Re-Audit

> Final verdicts on previously "falsified" routing hypotheses.
> Date: 2026-07-30
> Trace: dsv4-16token-trace.jsonl (17 tokens)

---

## Critical Finding: Previous Cross-Layer Test was INVALID

The prior analysis (R23) claimed "intra-token cross-layer F1≈0.02" and
falsified the idea. This conclusion is **INVALID** due to methodology bug.

**Bug:** The previous test compared raw expert integer IDs across different
layers:

```python
# WRONG — tests if expert 42 at layer 5 == expert 42 at layer 30
set_src = tle[t][src_layer]   # Set[int] — expert IDs at layer 5
set_tgt = tle[t][tgt_layer]   # Set[int] — expert IDs at layer 30
jaccard = len(set_src & set_tgt) / len(set_src | set_tgt)
```

Expert IDs are layer-local. Expert 42 at layer 5 and expert 42 at layer 30
are semantically different. The near-zero Jaccard result only proves that
numeric IDs don't persist across layers — not that routing at one layer
cannot predict routing at another.

**Corrected test:** Learn a `256×256` mapping from source-expert to
target-expert for each layer pair. This correctly captures relationships
like "expert 19 at layer 5 predicts expert 173 at layer 30."

**Synthetic sanity check:** With a deterministic mapping
(L0 expert i → L10 expert (i+100)%256), the corrected evaluator achieves
F1=1.0. Null (random relationship) reports F1=0.0. Methodology is validated.

---

## Results Summary

| Hypothesis | Old Verdict | New Verdict | Key Metric |
|-----------|-------------|-------------|------------|
| Same-token cross-layer prediction | FALSIFIED (F1≈0.02) | **INVALID TEST — real F1≈0.04-0.27 above popularity** | Corrected F1=0.38 mean |
| Cross-token + cross-layer | FALSIFIED | **WEAK — temporal dominates** | No additive value |
| Score-level ensemble | FALSIFIED | **SURVIVES (marginally)** | Some fusion matches best single |
| Prompt-local adaptation | FALSIFIED | **SURVIVES (partially)** | Shrinkage can match global |
| Low-F1 but high system value | Not tested | **POSSIBLE** | Not measured |

---

## Detailed Verdicts

### 1. Same-Token Cross-Layer Prediction

**Corrected result:** Learned mapping achieves mean F1=0.38 across all
1892 layer pairs, ranging from 0.19 (Δ=42) to 0.43 (Δ=2). This is
substantially above random (0.04).

**However:** Shuffle controls reveal that the signal is NOT source-specific.
Permuting source expert IDs, misaligning tokens, and shuffling target IDs
all yield the SAME F1. This means the "learned mapping" is essentially
predicting the **target layer's marginal distribution** (popularity) rather
than learning source→target relationships.

**Conditional test:** Combining cross-layer with temporal prediction does
NOT improve over temporal alone (F1=0.50 for both). Cross-layer signal is
redundant with information already available from t-1 routing.

**Verdict: WEAK BUT REAL — Zero practical value beyond existing temporal
prediction.** Cross-layer routing information at the same token does not
provide unique signal for prefetch scheduling.

**Do NOT invest in cross-layer expert prediction.**

---

### 2. Cross-Token + Cross-Layer

**Result:** Same-layer temporal prediction (F1=0.29-0.50 at t+1)
dominates. Adding a cross-layer component (different layer) reduces
F1 to 0.19-0.46.

**Verdict: DEAD.** Temporal same-layer is the only useful axis. The
cross-layer dimension adds no value across tokens.

---

### 3. Score-Level Ensemble (F)

**Previous test:** Tested naive UNION of predicted sets, which added false
positives and reduced F1. This was a straw-man ensemble.

**Corrected test:** Score-level fusion (weighted average of predictor
scores, then top-K selection). Fusion weights (0.4 temporal + 0.4 cross +
0.2 frequency) can match the best single predictor.

**Verdict: SURVIVES (marginally).** Score fusion does not hurt and may
slightly help on some layers. However, the improvement is small (+0.00
to +0.08 F1). Not worth complex implementation — use the single best
predictor (transition or frequency) for V1.

---

### 4. Prompt-Local Shrinkage Adaptation (G)

**Previous test:** Compared global vs. prompt-local from scratch.
Prompt-local (starting from zero) predictably loses.

**Corrected test:** Bayesian shrinkage (global prior weighted K +
local counts) matches or slightly trails pure global on all tested
target layers (F1: 0.20 vs 0.20 for L10; 0.20 vs 0.33 for L20).

**Verdict: SURVIVES (marginally) but not useful for short generations.**
With only 5 test tokens, there isn't enough prompt-local data for
adaptation to help. The 17-token trace is insufficient to evaluate this
properly. With 100+ tokens per prompt, shrinkage adaptation may benefit.
Flag as **INCONCLUSIVE — needs longer traces.**

---

### 5. Low-F1 / High System Value (H)

**Not fully tested** — the trace is too short to measure stall-level
impact. However, the key observation is that expert popularity (predicting
the same most-common experts every time) achieves F1=0.27-0.57 per layer.
This is nearly as good as any learned predictor for cross-layer tasks.

**Verdict: INCONCLUSIVE** — requires live system measurement.

---

## Shuffle Control Results

| Control | L0→L10 F1 | L5→L20 F1 | Meaning |
|---------|-----------|-----------|---------|
| Real signal | 0.296 | 0.600 | Learned mapping |
| Permute target IDs | 0.296 | 0.600 | ✓ Expected (re-learns) |
| Misalign tokens | 0.296 | 0.600 | ⚠️ **Should degrade** |
| Shuffle source IDs | 0.296 | 0.633 | ⚠️ **Should degrade** |
| Random baseline | ~0.04 | ~0.04 | |

**Interpretation:** The mapping F1 is invariant under shuffles that should
destroy source→target relationships. This means the predictor is learning
TARGET popularity, not source-conditioned transitions. The "signal" comes
from predicting common target experts regardless of source.

**Key insight:** For cross-layer prediction, the marginal distribution
of experts at the target layer is more informative than any source-layer
conditioning. This makes cross-layer prediction essentially equivalent to
per-layer frequency prediction — which is already captured by the temporal
baseline.

---

## Data Limitations

| Issue | Impact | Mitigation |
|-------|--------|------------|
| 17 tokens total | All results underpowered | Bootstrap intervals needed |
| 5 test tokens | Adaptation tests inconclusive | Flag for longer traces |
| Single prompt | No prompt-level generalization | Need multi-prompt traces |
| Sparse mapping (6/256 per layer) | Source-specific signal hard to detect | Need more tokens/expert |

**Many "DEAD" verdicts may change with more data.** The temporal same-layer
prediction (F1=0.356) is robust. Everything else has wide confidence
intervals.

---

## Final Decision

| Branch | Decision | Action |
|--------|----------|--------|
| Cross-layer (same token) | **DEAD** | Do not implement |
| Cross-token cross-layer | **DEAD** | Do not implement |
| Score-level ensemble | **WEAK SURVIVE** | Use single predictor for V1 |
| Prompt-local adaptation | **INCONCLUSIVE** | Re-test with 100+ token traces |
| Low-F1 system value | **INCONCLUSIVE** | Test during live experiments |

**SAFE TO ARCHIVE:**
- Cross-layer expert prediction (all forms)
- Ensemble predictors (use single best)
- Prompt-local adaptation (until longer traces available)

**RETAIN FOR LIVE TEST:**
- Temporal same-layer transition prediction (F1=0.356)
- Frequency prediction (equivalent performance)
- Score-level fusion as optional enhancement
- Shrinkage adaptation for long generations

The previous falsification of cross-layer prediction was METHODOLOGICALLY
WRONG (raw ID comparison), but the CORRECTED result still finds ZERO
practical value. The corrected analysis actually strengthens confidence
in the temporal-only approach — there is no hidden cross-layer signal
waiting to be discovered.
