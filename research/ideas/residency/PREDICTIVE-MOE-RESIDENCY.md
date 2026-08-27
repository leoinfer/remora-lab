# Predictive MoE residency

**Status:** `PROPOSED`

Predictive MoE residency uses route and workload forecasts to stage expert
state before use, while accounting for forecast error, transfer contention,
epoch invalidation, and eviction recovery. Predictive placement is useful only
if it beats a conservative baseline under trace replay.

Sources: [`V4_REMORA_EXPERT_CACHE_DESIGN.md`](../../archival/vault/V4_REMORA_EXPERT_CACHE_DESIGN.md),
[`Resident-Skeleton-Streamed-Oracle.md`](../../archival/vault/Resident-Skeleton-Streamed-Oracle.md),
and [`F9`](../../theory/checkers/F9-checker.md).
