# WARM GEOMETRY — CAUSAL ANALYSIS (Phase 2, Lane B)

Author: LANE B hostile-validation. Code: `phase2/laneB/` (bcommon.py, exp_a..exp_f).
Artifacts: `WARM_GEOMETRY_PC_SWEEP.json`, `WARM_GEOMETRY_CONTROLS.json`,
`WARM_GEOMETRY_CROSS_CORPUS.json`, `WARM_GEOMETRY_STRUCTURAL_PROBES_LB.json`.
Scoring mirrors `p6_scale.py`; all cells computed on full 81,920 candidates (shape-asserted).
Seed 20260823 everywhere; fit inputs guarded programmatically to `*_train_*` files.

## 0. Reproduction spot-check (gate for everything below)

Frozen transform (1m-train fit, center → rmPC16 → normalize) at k=16 reproduced the claim exactly:
**R@10 = 0.94 dev / 0.86 holdout** (`spot_check_k16_vs_claim.reproduces_claim = true`),
raw baseline **0.33 / 0.22** also exact. All deltas below are trusted against this gate.
The final L2-normalize step is a no-op under cosine scoring — asserted numerically
(`normalized_only == raw_dot` to fp32; `centered_normalized == centered`). The recipe's
active ingredients are centering and projection only.

## 1. What the raw pathology IS

- Uncentered 10m-dev spectrum: **pc1_frac = 0.9962, effective rank ≈ 1.01**. Every embed is
  ≈ one common direction plus a tiny residual; pooled queries sit nearer the mean than the
  section cloud (prior audit: raw norms p50 42.3 vs 27.7). Cosine therefore scores common-
  component overlap: target-cos p50 ≈ 0.88 but hardest-negative ≈ 0.90 (**median margin −0.019/−0.020**),
  and cross-type neighbors beat the target (prior WARM_PHASE4 numbers reproduced by our raw arm).
- Prior-artifact correction (preserved disagreement): `WARM_CENTERED_SVD.json` /
  `WARM_EMBEDDING_CENTERING_RESULT.json` label pc1_frac=0.5442 as holding for BOTH centered and
  uncentered spectra. That is wrong as labeled: 0.5442 is the **centered** 1m-train pc1 fraction;
  uncentered pc1_frac is 0.996 (consistent with PHASE4's uncentered svd_top10_energy_frac[0]=0.9964).
  Their arm results stand; the spectral labeling does not.

## 2. PC sweep shape (A) — narrow useful band, then a cliff

R@10 by k (dev | holdout), fit on 1m train:

| raw | k0 | k1 | k2 | k4 | k8 | 16 | 24 | 32 | 64 |
|-----|----|----|----|----|----|----|----|----|----|
| .33/.22 | .39/.42 | .85/.74 | .82/.74 | .88/.74 | .91/.74 | **.94/.86** | **.97/.92** | .91/.74 | .88/.70 |

Not monotone: sharp jump at k=1 (most of the dev gain comes from removing PC1 alone),
plateau, **peak at k=24 (.97/.92)**, then decline — at k=64 you have removed signal, not noise
(holdout falls back to .70). Top-16 centered PCs carry **91.9%** of centered variance
(10m train); post-centering eff_rank ≈ 5 (10m) / 3.16 (1m), npc95 = 21/18. The useful
intervention window is roughly the dominant-format subspace; oversweeping deletes instance signal.

## 3. Random-direction control (C) — direction-specificity SURVIVES decisively

Removing any 16 random orthogonal directions with identical dimensionality loss (same centering):
Haar frames R@10 mean **0.39 dev / 0.42 holdout** (5 seeds each); eigenbasis-uniform **0.39 / 0.428**;
bottom-16 **0.39 / 0.42** — i.e., indistinguishable from centering-only. Fraction of the
(top16 − centered) gain recovered: **≈ 0.00 both splits**. Verdict rule stated in artifact:
"top-PC dominance" interpretation survives; it is NOT generic "any projection of that subspace
denoises scale".

## 4. Whitening comparison (B)

Full-spectrum PCA whitening (all 3072 dims, eigenvalues clamped below 1e-6·λmax; ZCA verified
identical to PCA whitening under cosine to 1e-6, as rotation-invariance requires):
R@10 **0.85 dev / 0.86 holdout**, margins +0.026/+0.017. So variance *equalization* across the
whole spectrum recovers most but not all of rm16's advantage (0.94 dev); on holdout it ties.
Consistent mechanism: the damage is done by a few dominant axes; once they are handled,
further per-axis reweighting adds little.

## 5. Fit/test separation and corpus transfer (D)

- Refit on **10m train** (in-population): R@10 0.88 dev / 0.74 holdout — *worse* than the
  1m-fit frozen recipe (0.94/0.86). No fitting-on-eval-corpus advantage; the result is not
  an artifact of eval-corpus overfitting.
- Reverse transfer, 10m-fit tested at 8192-candidate 1m dev/holdout: raw 0.42/0.34 → **0.90/0.70**
  (1m-fit reference at same scale: 0.96/0.88). Works both directions across corpus versions.
- Leave-one-type-out fits (remove ALL train sections of type T): identical per-family recall to
  full fit — expected, since T is 2–12 of 81,920 sections (negligible perturbation; documented limit).
  Stronger test, **routine-only fit** (exclude every fact-bearing train section): per-family recall
  unchanged (e.g., manifest 1.00/1.00, reservoir 0.556 dev / 0.133 holdout). The transform never
  needs to see target-like content: it fits **generic filler/format statistics**.

## 6. Leakage audit (F) — clean at text level

Exact normalized 8-word shingle matching, FULL v10_1m_train archive vs FULL 10m dev/holdout
(stronger than the required sampling): **0 matched sections in either split** (0 rare-type,
0 targets). Headline excluding "matched" candidates is therefore trivially unchanged (0.94/0.86).
Supplement: generator templates ARE shared across corpus versions (same six fact-family bigrams;
11 exact question-template strings overlap 1m-train ↔ 10m-dev). That is structural sharing of the
benchmark family, not candidate-text leakage — it bounds what "cross-corpus" can mean here.

## 7. Structural probes (E)

Spectra (§1–2). Type probe (multinomial logistic, trained on 10m TRAIN sections, tested on all
81,920 dev sections): overall acc ≈ majority baseline (~0.9996) in all spaces as expected under
imbalance; macro-F1 **raw 0.871 → centered 0.851 → transformed-rm16 0.788**, per-class recall of
every rare type stays 1.0. PC removal does NOT remove summary-type identity; it dents it slightly.
Surface correlations: **no pad-dilution analog** — length↔cos-to-mu r = 0.008, length↔norm r = 0.004
(lengths 60–185 words, std 35). Instead there is a **positional artifact**: section_index ↔ norm
Pearson **−0.50**, ↔ cos-to-mu **−0.52**.
Queries: intra-template cosine = **1.000000** (bitwise-identical pools), inter-template p50 0.9973;
silhouette by template 1.0 vs by fact-family 0.84 in both spaces. Query vectors encode surface
template almost entirely.

### What the top-16 PCs actually ARE

Extreme-scored sections: PC1 ← indices {0,1,2,4}; PC2 ← {11111,11211,...}; PC3 ← {20000,20001,...};
PC4 ← {11111,1111,...}. PC1 correlates with vector norm at **r = −0.987** and with archive position
at ρ = +0.63. The dominant centered components are (i) the global-mean/norm axis and (ii)
**archive-index-prefix magnitude statistics** ("Section N:"/"N:" token-length effects) plus routine
filler word-bag statistics — not fact content. This explains the index↔norm correlation and why a
routine-only fit reproduces everything.

## 8. Mechanism narrative (measured, not assumed)

1. `last_boundary_pooled` embeddings of this synthetic corpus live near a rank-1 common direction
   (uncentered eff_rank 1.01) composed of backbone anisotropy + shared "Section N:"-prefix and
   filler-template statistics. Cosine in that cloud measures format agreement, so the true
   fact-phrase signal (question "…manifest entry a1…" ↔ summary "manifest entry a1") is buried
   under ~0.996 of shared cosine.
2. Centering removes the shared offset; removing the top-k centered axes removes the norm/index/
   format axes specifically (random axes do nothing — §3). Margins flip from negative to positive
   (median −0.019 → +0.029/+0.025); sibling_top1 goes 0.042-historical → 0.73/0.60, far above the
   ~1/n_family ≈ 0.03 that pure type-clustering would give, so surviving discrimination is
   **instance-level within family**: the residual space aligns the unique fact phrase shared by
   question and its target summary while different-family and different-fact sections fall away.
3. Type identity survives removal (macro-F1 0.79), so the removed subspace is not "the semantics";
   it is the format scaffolding. The surviving signal is lexical-distributional phrase match inside
   a synthetic vocabulary — genuinely instance-specific, but only as semantic as "fact-phrase
   overlap", with values absent from both sides by construction.
4. Failure mode is family-concentrated: all 4 unique holdout failures are reservoir-level questions
   ranked 11–32 — within-family confusability, the predicted residue after format denoising.

## 9. Statistical caveat found during audit (material)

`v11_10m_dev.jsonl` has **31 unique questions in 100 lines**; holdout **38 unique in 50 lines**
(literal duplicate lines ⇒ bitwise-identical embeddings, identical targets). On deduplicated
queries the headline barely moves — **R@10 = 0.9355 dev (29/31) / 0.8947 holdout (34/38)**,
raw 0.355/0.237 — but binomial uncertainty at n=31/38 is ±~0.08–0.13. Any future comparison that
wins by <0.1 against this baseline is not decision-grade without more unique questions.

## 10. Strongest remaining alternative explanation

"The repair equalizes a few pathological variance axes; ANY method that suppresses dominant
axes (whitening, mean-subtraction variants) would do the same." Supported in part: full whitening
ties rm16 on holdout (0.86). Refuted in its strong form by §3: random 16-dim projections recover
none of the gain. A second alternative — success is family/type clustering, not instance match —
is refuted by sibling_top1 0.73/0.60 vs ≈0.03 chance and per-family median rank 1. Residual risk:
everything here is one synthetic generator family with shared templates and duplicated queries;
no evidence yet about open-domain text.

## 11. FALSIFICATION VERDICT: **SURVIVES** (with precision caveat §9)

Three most decisive numbers:
1. **Random-16 controls recover 0% of the gain** (haar/eigen-uniform/bottom R@10 = 0.39/0.42 =
   centered-only baseline) vs top-16 = **0.94/0.86** — the effect is direction-specific, not
   dimensional bookkeeping.
2. **Zero text leakage** (0/81,920 shingle-matched sections either split) and **routine-only fits
   reproduce the full result** (per-family recall unchanged) — generalization across corpora is real
   and needs no target content in the fit.
3. **Margin flip**: median (target − hardest-negative) cosine −0.019/−0.020 raw → +0.029/+0.025
   transformed, with sibling_top1 0.73/0.60 ≫ 0.03 type-chance — instance-level signal exists after
   the intervention and did not before.

Caveat attached: effective sample is 31/38 unique queries; treat third-decimal recall differences
as noise.
