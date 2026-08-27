# REMORA reclaim

**Status:** `PROPOSED`

REMORA reclaim makes eviction a dependency-aware operation. It separates
portioning, reclaim, restoration, reserve capacity, and provenance so a
recovered artifact is not mistaken for valid current state.

Sources: [`CONTEXTFOLD_RECLAIM_MODEL.md`](../../archival/contextfold/CONTEXTFOLD_RECLAIM_MODEL.md),
[`V4_REMORA_EXPERT_CACHE_DESIGN.md`](../../archival/vault/V4_REMORA_EXPERT_CACHE_DESIGN.md),
and [`REMORA`](../../systems/REMORA.md). Related conjectures include CCC,
UARC, ENSM, and the safe-region requirement.
