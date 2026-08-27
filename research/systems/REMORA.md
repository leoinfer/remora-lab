# REMORA

**Status:** `PARTIALLY_IMPLEMENTED` as a research program

REMORA is the resource-aware inference-control umbrella around elastic
speculation, residency, reclaim, and evidence-carrying state. It treats
accepted tokens, bytes moved, verification work, cache state, and available
slack as one control problem. The name covers several mechanisms; it is not a
claim that one finished REMORA runtime exists.

## Main mechanisms

| Mechanism | Research role | Public record |
| --- | --- | --- |
| TBEH | Tail-Bounded Elastic Horizon: choose speculative depth from predicted acceptance and tail risk | [`TBEH`](../ideas/speculation/TBEH.md) |
| PHASE | Represent speculative outcomes as a branch DAG and choose verification work | [`PHASE`](../ideas/speculation/PHASE.md) |
| RSSO | Resident skeleton plus streamed oracle, with explicit authority boundaries | [`RSSO`](RSSO.md) |
| Portion / Reclaim | Move, retain, and evict state according to value and dependency closure | [`REMORA-RECLAIM`](../ideas/residency/REMORA-RECLAIM.md) |
| Computational refrigerator | Preserve reusable artifacts with provenance, validation, and dependency versions | [`C-01`](../conjectures/C-01-CCC.md) |
| Waste Ledger | Account for rejected work, stranded bytes, and avoidable transfers | [`WASTE-LEDGER`](../ideas/scheduling/WASTE-LEDGER.md) |
| Tiered reserve | Keep capacity for recovery, verification, and burst demand | [`REMORA-RECLAIM`](../ideas/residency/REMORA-RECLAIM.md) |
| Fast/slow clocks | Adapt routing and maintenance on different time scales | [`C-06`](../conjectures/C-06-SPEH.md) |

## REMORA Metabolism

The native metabolism subsystem is a first-class Rust control surface, not an
omitted research metaphor. Its frozen public scope is documented in
[`PROVENANCE_AND_SCOPE.md`](../../docs/remora_metabolism/PROVENANCE_AND_SCOPE.md).
The implementation is [`har-metabolism`](../../har/crates/har-metabolism/),
wired into runtime, residency, and decode-control observation boundaries.

| Part | Rust module | Meaning |
| --- | --- | --- |
| Portion | [`portion.rs`](../../har/crates/har-metabolism/src/portion.rs) | Decide whether optional work is affordable. |
| Reserve / mobilization | [`reserve.rs`](../../har/crates/har-metabolism/src/reserve.rs) | Protect recovery capacity and bound reserve debt. |
| Refrigerator | [`artifact.rs`](../../har/crates/har-metabolism/src/artifact.rs) | Carry causal identity and classify validity. |
| Reclaim | [`reclaim.rs`](../../har/crates/har-metabolism/src/reclaim.rs) | Classify spent work without inventing value. |
| Salvage | [`salvage.rs`](../../har/crates/har-metabolism/src/salvage.rs) | Admit retained artifacts only when expected value clears cost. |
| Waste Ledger | [`ledger.rs`](../../har/crates/har-metabolism/src/ledger.rs) | Record unique work and evidence-backed credit. |
| Moving maintenance setpoint | [`setpoint.rs`](../../har/crates/har-metabolism/src/setpoint.rs) | Estimate the minimum healthy budget. |
| Uncertainty-adjusted safe surplus | [`surplus.rs`](../../har/crates/har-metabolism/src/surplus.rs) | Bound optional work under uncertainty and pressure. |
| Fast/slow clocks | [`clock.rs`](../../har/crates/har-metabolism/src/clock.rs) | Separate immediate observations from slow adaptation. |
| Energy and replay | [`energy.rs`](../../har/crates/har-metabolism/src/energy.rs), [`trace.rs`](../../har/crates/har-metabolism/src/trace.rs) | Preserve explicit UNKNOWN energy and deterministic replay. |

The associated research records remain visible in the REMORA manifest: Portion,
Reclaim, computational refrigerator, value-weighted salvage, Waste Ledger,
tiered reserve, reserve mobilization, moving maintenance setpoint,
uncertainty-adjusted safe surplus, and fast/slow adaptation clocks. The
bodybuilding/macros, fatigue/recovery/RIR, and investment/salvage analogies
remain in their cards as historical working analogies; they are not runtime
semantics.

## Evidence boundary

The source corpus contains formal sketches, simulator-oriented queues,
counterexamples, and partial Rust control surfaces. It does not establish a
full-model speedup, universal acceptance improvement, or safe autonomous
adaptation. The cheapest next checks are trace replay, finite-state policy
comparison, dependency-closure validation, and adversarial acceptance tests;
see the [`formalization queue`](../archival/authoritative/REMORA_FORMALIZATION_QUEUE.md)
and [`experiment queue`](../archival/authoritative/COMPLETE_EXPERIMENT_QUEUE.md).

## Runtime boundary

Related implementation surfaces are Rust crates under
[`har-decode-control`](../../har/crates/har-decode-control/),
[`har-residency`](../../har/crates/har-residency/),
[`har-memory`](../../har/crates/har-memory/),
[`har-metabolism`](../../har/crates/har-metabolism/), and
[`har-certificates`](../../har/crates/har-certificates/). The research notes
are not loaded by the executable. Python prototypes, native helper processes,
and foreign inference backends are outside the production boundary.

## Source and status

The consolidated source is the [`REMORA master prompt`](../archival/authoritative/REMORA_MASTER_PROMPT_2026-08-03-1.md),
with the manifest, conjectures, open problems, and negative knowledge linked
from [`research/README.md`](../README.md). Status labels are normalized for
navigation; the archival evidence labels remain authoritative.
