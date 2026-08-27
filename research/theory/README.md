# Theory and checkers

The checker queue turns research claims into finite, replayable obligations:
roofline accounting, stopping decisions, certificate replay, state exactness,
delta bounds, dependency closure, branch enumeration, critical paths,
residency replay, value-of-computation ledgers, phenotype safety regions,
certificate composition, source invariants, and trace completeness.

The exact F0–F15 records are under [`checkers/`](checkers/), with ContextFold
and REMORA formal source under [`../archival/`](../archival/). These are
CPU/source-level research specifications; they do not imply GPU or full-model
validation.
