# REMORA Metabolism — Provenance and Scope

**Status:** FROZEN PUBLIC SCOPE

This document fixes the public scope of the native `har-metabolism` subsystem.
It separates historically explicit mechanisms from later compatible
formalizations, experimental extensions, and rejected or falsified claims.
The public copy omits private source paths, model payloads, and raw receipts;
the mechanism boundaries and non-claims remain explicit.

## 1. Canonical public sources

| Source | Role |
| --- | --- |
| [`REMORA_NEW_IDEA_MASTER_MANIFEST.md`](../../research/archival/authoritative/REMORA_NEW_IDEA_MASTER_MANIFEST.md) | REMORA Portion through Fast/Slow Adaptation Clocks |
| [`REMORA_DISCOVERY_NOTES.md`](../../research/archival/authoritative/REMORA_DISCOVERY_NOTES.md) | verified-token transition and dependency-closure exactness |
| [`REMORA_OPEN_PROBLEM_PORTFOLIO.md`](../../research/archival/authoritative/REMORA_OPEN_PROBLEM_PORTFOLIO.md) | OP-01..OP-12, including the OP-09 accounting identity |
| [`REMORA_FORMALIZATION_QUEUE.md`](../../research/archival/authoritative/REMORA_FORMALIZATION_QUEUE.md) | replayable checkers and falsifiers |
| [`REMORA.md`](../../research/systems/REMORA.md) | public system-level map and implementation boundary |
| [`PFM.md`](../../research/systems/PFM.md) | artifact lifecycle, reserve, and accepted-token accounting context |
| [`REMORA-RECLAIM.md`](../../research/ideas/residency/REMORA-RECLAIM.md) | reclaim, restoration, reserve, and provenance record |
| [`WASTE-LEDGER.md`](../../research/ideas/scheduling/WASTE-LEDGER.md) | rejected-work and avoidable-transfer accounting |
| [`H27.md`](../../research/ideas/atlas/H27.md) | salvage-aware work valuation and its preserved investment analogy |

These sources are the public authority for the research meaning. The native
crate is the implementation surface; a Rust implementation or test does not
promote an unverified model-quality or throughput claim.

## 2. Scope statement

REMORA Metabolism is a native, bounded, deterministic runtime-control family.
It classifies observed work, assigns one value class to each event, maintains
explicit tiered reserve, validates artifact provenance, reclaims and salvages
only causally valid artifacts, and emits accounting through a Waste Ledger.

The controller charges exact committed tokens and verified authority work. It
does not charge raw generated positions as accepted output, grant imagined
overlap credit, or grant speculative reuse credit before reuse is observed.
Unknown causal identity, validity, generation, kernel/backend identity, state
consistency, transfer ownership, or exactness class fails closed. REMORA never
bypasses HAR exactness gates and never authorizes a foreign execution path.

The shared formal object is the verified-token state transition:

```text
(state, candidate, schedule) -> (accepted, state', retained, debt)
```

The public Waste Ledger identity is:

```text
B - C = avoided_baseline_work + measured_overlap
        - extra_work - contention - unreturned_reserve_debt
```

## 3. Mechanism classification

### A. Historically explicit mechanisms — native contracts

| # | Mechanism | Native contract |
| --- | --- | --- |
| A01 | REMORA Portion | Budget optional work; account for underage and overage; emit ADMIT / DEFER / REJECT / FAIL_CLOSED. |
| A02 | REMORA Reclaim | Classify spent work as EXACT_REUSABLE / CONDITIONAL_REUSABLE / INFORMATIONAL_ONLY / UNRECOVERABLE. |
| A03 | REMORA Refrigerator | Carry a causal-prefix provenance envelope; a changed causal leaf causes MISS / INVALIDATED. |
| A04 | REMORA Salvage | Admit retained work only when expected avoided cost exceeds holding, opportunity, and validation cost. |
| A05 | REMORA Waste Ledger | Record unique events and grant reuse/overlap credit only with evidence. |
| A06 | REMORA Tiered Reserve | Track capacity, committed, available, protected, debt, expiry, and pressure across resource tiers. |
| A07 | Reserve Mobilization | Spend protected reserve only after an explicit post-action viability check. |
| A08 | Moving Maintenance Setpoint | Estimate the minimum healthy budget from observable runtime state. |
| A09 | Uncertainty-Adjusted Safe Surplus | Reduce optional capacity under uncertainty, cold misses, rollback, pressure, low salvage, or interference. |
| A10 | Fast/Slow Adaptation Clocks | Update block/cache/transfer state quickly and hotsets/phenotype/equilibrium slowly; invalidate slow state on identity changes. |
| A11 | Accepted-token roofline | Measure value using exact accepted work and complete cost, not generated positions alone. |
| A12 | Resource-complementarity bound | Admit background work only while every selected resource remains above its safety reserve. |
| A13 | Energy economics | Keep joules/token and energy-delay metrics UNKNOWN without an appropriate physical energy source. |

### B. Compatible formalizations

The public implementation also carries the following typed forms: handoff
packets; artifact lifecycle states; deadline/slack calculation; expected-value
and salvage formulas; reserve budgets; post-action viability; and the TBEH
accepted-output efficiency expression `eta(K) = E[A(K)] / E[T(K)]`. These are
formalizations of the A mechanisms, not claims that every policy is optimal.

### C. Experimental extensions

Thermal/clock-drift models beyond V1, MAX/BALANCED/EFFICIENT profiles, a
standalone resident energy-delay metric, and future-state-compressed
materialization remain explicitly experimental. They are not part of the V1
critical path.

### D. Rejected, falsified, or unsupported claims

- A distinct PFM-A incremental materialization mechanism is not established by
  the current oracle evidence.
- Cache-hit-as-speed and byte-reduction-as-speed are not sufficient metrics.
- Approximate future hidden/KV/recurrent state is not authoritative without
  exact dependency closure and verification.
- Double-counted credit, overlap credit without a witness, and reuse credit
  before observed reuse are invalid.
- Missing rankings and private raw evidence are not reconstructed or silently
  promoted.

## 4. Native Rust implementation

The first-class implementation is [`har-metabolism`](../../har/crates/har-metabolism/):

| Part | Rust surface | Public companion |
| --- | --- | --- |
| Controller and common estimates | [`controller.rs`](../../har/crates/har-metabolism/src/controller.rs), [`common.rs`](../../har/crates/har-metabolism/src/common.rs) | deterministic composition and fail-closed decisions |
| Portion | [`portion.rs`](../../har/crates/har-metabolism/src/portion.rs) | optional-work admission |
| Reserve and mobilization | [`reserve.rs`](../../har/crates/har-metabolism/src/reserve.rs) | tiered capacity and bounded debt |
| Artifact refrigerator | [`artifact.rs`](../../har/crates/har-metabolism/src/artifact.rs) | provenance and reuse class |
| Reclaim | [`reclaim.rs`](../../har/crates/har-metabolism/src/reclaim.rs) | reuse classification |
| Salvage | [`salvage.rs`](../../har/crates/har-metabolism/src/salvage.rs) | expected-value admission |
| Waste Ledger | [`ledger.rs`](../../har/crates/har-metabolism/src/ledger.rs) | unique-event accounting |
| Moving setpoint | [`setpoint.rs`](../../har/crates/har-metabolism/src/setpoint.rs) | maintenance floor |
| Safe surplus | [`surplus.rs`](../../har/crates/har-metabolism/src/surplus.rs) | uncertainty-adjusted optional budget |
| Fast/slow clocks | [`clock.rs`](../../har/crates/har-metabolism/src/clock.rs) | multi-timescale observations |
| Energy state | [`energy.rs`](../../har/crates/har-metabolism/src/energy.rs), [`energy_measurement.rs`](../../har/crates/har-metabolism/src/energy_measurement.rs) | explicit UNKNOWN boundary |
| Snapshots and replay | [`snapshot.rs`](../../har/crates/har-metabolism/src/snapshot.rs), [`trace.rs`](../../har/crates/har-metabolism/src/trace.rs) | deterministic evidence surface |

The runtime integration points are [`RuntimeMetabolism`](../../har/crates/har-runtime/src/metabolism.rs),
[`ResidencyMetabolism`](../../har/crates/har-residency/src/remora.rs), and
[`DecodeMetabolismGate`](../../har/crates/har-decode-control/src/metabolism.rs).
They observe or record work after the owning subsystem establishes exactness;
they do not change token acceptance, bypass residency transitions, or provide
a fallback backend.

## 5. Verification and non-claims

The crate's unit tests and [`invariants.rs`](../../har/crates/har-metabolism/tests/invariants.rs)
exercise unknown-input fail-closed behavior, evidence-gated reuse and overlap
credit, deterministic decisions, bounded reserve mobilization, ledger
conservation, trace replay, and explicit UNKNOWN energy state. Run them with:

```text
cargo test -p har-metabolism --locked
```

This lane proves bounded Rust control/accounting behavior. It does not prove
full-model generation, model quality, universal acceptance improvement,
throughput, or energy efficiency. GPU energy remains UNKNOWN unless a suitable
GPU-only source is supplied by the caller. Research-only source notes and
Python prototypes are not runtime dependencies.

## 6. Relationship to the original working analogies

The research cards retain the original reasoning trail. For example, H20 keeps
fatigue/recovery/RIR, H22 keeps bodybuilding/macros, and H27 keeps the
investment/capital/salvage analogy. Those analogies are design lenses; the
technical claims above are defined by observable state, accounting identities,
correctness gates, and replayable tests.
