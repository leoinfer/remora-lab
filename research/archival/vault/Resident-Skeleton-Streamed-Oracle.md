---
title: Resident Skeleton + Streamed Oracle
type: architecture
status: design-gated
---

# Resident Skeleton + Streamed Oracle (RSSO)

RSSO is the working name for an exact speculative architecture for the dense
Qwen3.6-27B Q8_K_XL lane. A resident skeleton is an altered, training-free
draft graph made only from original checkpoint tensors. A streamed oracle is
the complete original Q8 target. The oracle remains authoritative for every
committed token.

The initial sequence is: freeze/autopsy → dependency graph → cost model →
residency planner → exact LayerPack design → teacher-forced skeleton search →
offline break-even simulation → gate review. The RSSO gate remains
**DEFER — INSUFFICIENT RSSO/STATE/ROOFLINE EVIDENCE**. The minimal B0 fix is
qualified only in an isolated production worktree and the 10-run deterministic
baseline is frozen; phase/physical-byte evidence is still incomplete. No
hot-path implementation is authorized.

External authority: `[local path omitted]`,
`RSSO_MODEL_DEPENDENCY_GRAPH.md`, `RSSO_IMPLEMENTATION_GATE_REVIEW.md`, and
`QWEN36_27B_REPEATABILITY_CERTIFICATE.md` and
`QWEN36_27B_REPEATABILITY_ADDENDUM_2026-08-04.md`.

RSSO is not a trained 3B model, not approximate sparse inference, and not a
claim about the Qwen Q2 MoE lane or DeepSeek. The exact mode requires temporary
draft state, full target verification, accepted-prefix commit, and rejected
suffix discard.
