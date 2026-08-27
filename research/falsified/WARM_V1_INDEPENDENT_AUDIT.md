# WARM-V1 INDEPENDENT AUDIT — AGENT 1 (HOSTILE REVIEWER)

**Date:** 2026-08-23 · **Claim under attack:** geometry transform (train means + remove top-16 PCs + L2 norm) lifts 10M semantic R@10 from 0.33/0.22 → **0.94/0.86** at 81,920 candidates, cross-corpus fit, leakage-audited.

**Verdict: CONDITIONAL** — the core result is real and survives every invalidation attempt I could mount; it holds only under stated scope restrictions (same-generator synthetic domain; hybrid/reader claims excluded as evidence for semantics).

Full machine-readable detail: `WARM_V1_INDEPENDENT_AUDIT.json`. Clean-room code + raw outputs: `phase2/audit/{a1..a6}*.py`, `phase2/audit/out/*.json`.

---

## 1. Metric reimplementation (check #1) — PASS, exact

I rebuilt cosine retrieval and the transform from scratch (no p5/p6 imports), fit only on `results/1m/train_*`:

| | claimed | my clean-room |
|---|---|---|
| raw dev R@10 / MRR / rank-med | 0.33 / 0.1242 / 17 | **0.33 / 0.1242 / 17** |
| raw holdout R@10 / MRR / rank-med | 0.22 / 1023 / 24 | **0.22 / 1023 / 24** |
| transformed dev R@1/@5/@10, MRR, sim_p50 | 0.73/0.88/0.94, 0.804, 0.574 | **identical to ≤0.001** |
| transformed holdout R@1/@5/@10, MRR, sim_p50 | 0.60/0.82/0.86, 0.689, 0.5425 | **identical to ≤0.001** |

Audit-process note: my *first* pass scored **0.39/0.42** — I sliced `Vh[:, :16]` instead of `Vh[:16].T` (rows vs columns of right singular vectors). Legacy `torch.svd`'s column convention makes `p6_scale.py` correct. This metric is razor-sensitive to PC-convention bugs; the final clean-room matches bit-for-bit on the projector (max abs diff 0.0 between `torch.svd` and `torch.linalg.svd` paths). Also confirmed: `rank_median=1` in `WARM_GEOMETRY_RANKS_FIX.json` is right; the stale `4715/47605` fields in `WARM_GEOMETRY_SCALE_FULL.json` are wrong (self-flagged).

## 2–3. Population & gold integrity (#2, #3) — PASS

- 81,920 sections per split, **all distinct texts** (sha1), summaries unique; targets in range, unique; scoring ran over the true full width (my sims were 100×81920).
- Fact names unique per archive → each question matches **exactly one** summary lexically; expected values unique within archive and appear in **no other section's text** (exhaustive scan). Golds unambiguous.
- Minor: dev = 100 questions but only **31 unique facts** (max 4× repetition); holdout 38/50. Fact-level n is what matters for confidence intervals.

## 4. Template sharing (#4) — FAIL (the known weakness, now counted)

Exactly **one** question template across all six corpus×split sets (`v10_make_1m_corpus.py:89-92` ≡ `v11_make_10m_corpus.py:91-94`, same RNG seed 4339):
> *"Reading the archive, find where the \<FACT\> is recorded and report its value."*

7/100 dev questions and 5/50 holdout questions are **verbatim string-identical** to transform-fit-corpus questions (same fact names like "manifest entry a1"). Answer values: zero overlap (good).

## 5. Text contamination (#5) — PASS

Word-5-gram shingles, 3,000 docs/side, 9M pairs per split-pair: collision rates **2.4–2.8e-05**, max shared shingles 1–2, mean Jaccard ≈ 0.004, **zero exact duplicate sections** across any archive pair (including fit corpus v10_1m_train). Consistent with chance under the 5-word filler vocabulary.

## 6. Fit-boundary leak (#6) — CONCERN, headline survives

The subtle leak is real and large:
- **99.97% / 99.96%** of dev/holdout candidate summaries have index-stripped forms among the fit corpus's **37 summary forms**; **9.93% (~8,136)** are verbatim string matches. "Cross-corpus" = cross-*instance* of the same generator.
- **Recomputed headline after removing all verbatim-overlap candidates** (73,784 remain): **dev 0.97, holdout 0.958** — the result does not depend on overlapping candidates; it slightly improves. Restricted to indices ≥8192 (outside the fit archive's literal index range): R@10 = 1.0 (n=33/17).
- Removing *all* form-overlapping candidates is degenerate (24/33 survive; targets themselves removed) — evidence of form-space saturation, not extra robustness.
- No answer-value leakage anywhere; k=16 was selected using 1m-dev (p5 arms), not 10m eval — acceptable but worth stating.

## 7–8. Sibling buckets & index priors (#7, #8) — PASS

Buckets: routine 81,88x (contains no targets) + six fact buckets of size 3–11; sibling_top1 well-defined, not degenerate. Trivial priors: most-popular-index **0.00**, ascending-index **0.08**, log-band histogram **0.00**, chance 1.22e-04 — nothing approaches 0.94. Dev's rank↔index spearman 0.356 traces entirely to two hard facts (idx 1168/3508) repeated 3×; holdout rho=0.013. Target indices are front-loaded (67% < 8192) by generator design, equally available to all methods.

## 9–11. Lexical honesty & hybrid vacuity (#9, #11)

My independent lexical implementation: **R@1=R@5=R@10=MRR=1.0 both splits**, unique match per question, expected value in gold first sentence 100%, zero value strings in summaries. The lexical claim is honest.

Consequently **union(lex8, sem16)+verifier = 1.00 is fully entailed by the lexical channel alone** — the hybrid demonstrates **zero** marginal value of semantic retrieval on this suite. Producer script for `WARM_HYBRID_RESULT.json` was never committed (provenance gap).

## 12. Oracle reader (#12) — CONCERN

`oracle_reader.py` feeds the **gold section span** (oracle-conditioned) to a **reference** llama-server (build-r4x-vk, port 39815), n=12 first dev questions, ~10.5 s/answer. EM 1.00 / 0.917 is an oracle upper bound on a reference backend — not end-to-end chain performance.

---

## Verdict rationale

**Nothing on my invalidation list materialized**: no labels in fitting, no duplicated golds, no population subset trick, no index shortcut, no textual near-dup contamination, and the headline *survives removal of exactly the candidates that overlap the fit corpus* (0.97/0.958 ≥ claimed 0.94/0.86).

**CONDITIONAL because**: (1) everything lives inside a one-template synthetic generator whose entire candidate text space overlaps the fit corpus's form space — claims must be scoped accordingly ("cross-corpus" ≠ out-of-distribution); (2) the surviving semantic signal operates on fact-name tokens embedded in summaries — de-anisotropied token matching, not deep relational memory; (3) dev has ~31 independent units; (4) hybrid and reader results may not be cited as semantic-retrieval or chain evidence; (5) three producer scripts are missing from the repo (headline numbers re-derived here; latency subclaims remain unverifiable from source).

— Agent 1, hostile reviewer. Audit code: `phase2/audit/`. Raw outputs: `phase2/audit/out/`.
