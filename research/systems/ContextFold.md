# ContextFold

**Status:** `PARTIALLY_IMPLEMENTED` as a codec and research line

ContextFold is a proposal for causal context materialization: preserve a
small exact base, add progressively finer residuals, and materialize only the
prefix or state needed by the current query. Its goal is effective context
capacity under bounded storage and transfer, not a claim that a dense model
has been run at ten million tokens.

## Core model

The ContextPack line uses a base-plus-residual representation with explicit
causal dependencies. Key-first/value-late materialization, prefix atlases,
hot/warm/cold placement, and reclaim are separate mechanisms that must each
carry their own correctness certificate. Delta-Certified Skipping may omit
work only when a bound covers the downstream effect.

The primary source set is [`CONTEXTFOLD_EXECUTIVE_REPORT.md`](../archival/contextfold/CONTEXTFOLD_EXECUTIVE_REPORT.md),
[`CONTEXTFOLD_FORMAL_SPECIFICATION.md`](../archival/contextfold/CONTEXTFOLD_FORMAL_SPECIFICATION.md),
[`CONTEXTPACK_FORMAT.md`](../archival/contextfold/CONTEXTPACK_FORMAT.md), and
[`CONTEXTPACK_SCHEMA.json`](../archival/contextfold/CONTEXTPACK_SCHEMA.json).
The open-boundary records are [`OP-01`](../open-problems/OP-01.md),
[`OP-02`](../open-problems/OP-02.md), [`C-01`](../conjectures/C-01-CCC.md),
[`C-03`](../conjectures/C-03-CAL.md), and [`C-05`](../conjectures/C-05-UARC.md).

## Evidence boundary

The archive records codec work, formal models, missing-data notes, and
counterexamples. It does not establish retrieval quality, long-context model
quality, or an end-to-end 10M-token result. Exactness, error propagation,
dependency closure, and transfer cost remain testable research questions.

## Runtime boundary

The reviewed runtime surface is the Rust
[`har-contextfold`](../../har/crates/har-contextfold/) crate and related HAR
storage/memory crates. Research documents are not runtime inputs. Model files
and context payloads are caller-supplied and are not included here.
