# ContextFold idea family

**Status:** `PARTIALLY_IMPLEMENTED`

ContextFold explores causal context compression and selective materialization.
Its ideas include base-plus-residual KV state, key-first/value-late access,
prefix atlases, ContextPack levels, Delta-Certified Skipping, Causal-Closure
Cache, and dependency-versioned exact reuse.

Read the system overview in [`research/systems/ContextFold.md`](../../systems/ContextFold.md)
and the exact source set in [`research/archival/contextfold`](../../archival/contextfold/).
The implementation and quality boundaries are recorded in OP-01/OP-02 and
C-01/C-03/C-05; none of those records is a blanket 10M-context or quality
claim.
