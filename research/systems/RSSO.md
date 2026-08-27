# Resident Skeleton and Streamed Oracle (RSSO)

**Status:** `PROPOSED`

RSSO separates a compact resident model skeleton from a streamed authority
path. The skeleton supplies stable low-cost structure; the oracle supplies
larger or higher-fidelity information only when a certificate, routing
decision, or uncertainty policy requests it. The boundary is explicit so a
cache hit cannot silently become an authority claim.

The design is related to ContextFold's causal materialization, DSpark/MTP's
future-token proposals, and expert residency. It is not equivalent to loading
an entire model into memory and is not a substitute for a verified model
execution path.

## Required contracts

- define the authority source for every accepted output;
- version the resident state and all dependencies used to produce it;
- distinguish a proposal, a verified token, and a reusable exact artifact;
- charge transfer, verification, and eviction work to the same trace; and
- invalidate artifacts at an explicit state boundary.

The source model and checker queue are [`RSSO_MODEL.md`](../archival/contextfold/CONTEXTFOLD_RSSO_MODEL.md),
[`Resident-Skeleton-Streamed-Oracle.md`](../archival/vault/Resident-Skeleton-Streamed-Oracle.md),
and [`F4`](../theory/checkers/F4-checker.md). The related conjectures are
[`C-01`](../conjectures/C-01-CCC.md), [`C-04`](../conjectures/C-04-ENSM.md),
and [`C-08`](../conjectures/C-08-state-boundary.md).

## Falsifiers

RSSO is weakened or rejected if a compact skeleton cannot meet its error and
latency budget, if oracle fetches dominate the saved work, or if exact reuse
cannot be certified under changing dependencies. No public performance claim
is attached to this entry.
