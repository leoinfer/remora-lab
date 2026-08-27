# Research library

This is a first-class archive of the local-AI research program, not optional
commentary around HAR. It preserves implemented and partial systems,
experiments, hypotheses, conjectures, open problems, formal models,
optimization principles, rejected ideas, and future directions.

Start with:

- [`../RESEARCH_IDEA_INDEX.md`](../RESEARCH_IDEA_INDEX.md) — searchable human
  index of every recovered record;
- [`../research_idea_index.json`](../research_idea_index.json) — canonical
  machine-readable index and status vocabulary;
- [`SOURCE_REGISTER.md`](SOURCE_REGISTER.md) — public-safe provenance and
  inclusion/exclusion decisions;
- [`archival/authoritative/`](archival/authoritative/) — preserved source
  documents, including the complete HERMES atlas, REMORA manifest, open
  problems, conjectures, counterexamples, formalization queue, and experiment
  queue;
- [`systems/`](systems/) — readable system-level entry points for REMORA,
  ContextFold, HAR, R4X/R4KV/R4F, DSpark/MTP, ExpertPack, and MARC-Symbiote;
- [`ideas/`](ideas/) — H/N atlas cards and thematic mechanism notes;
- [`open-problems/`](open-problems/) and [`conjectures/`](conjectures/) —
  exact section-level records;
- [`experiments/`](experiments/) — the complete E001–E096 queue as individual
  cards;
- [`falsified/`](falsified/) — preserved negative knowledge and
  counterexamples; and
- [`theory/`](theory/) and [`roadmap/`](roadmap/) — checkers, equations,
  formal boundaries, and staged research plans.

The governing measurement rule for implausible results is in
[`docs/methodology.md`](../docs/methodology.md): preserve the anomaly, build
stronger known-answer tests, and try to falsify the claim before publishing
either a breakthrough or a dismissal.

## Reading rules

The archival documents retain original names and equations. A source's
historical claim is not silently upgraded: `PROVED`, `MACHINE-CHECKED`,
`DERIVED UNDER ASSUMPTIONS`, `CONJECTURED`, `FALSIFIED`, `BLOCKED`,
`SIMULATOR-ONLY`, and related evidence labels keep their original scope.
The normalized status in a card or index is a navigation aid, not a stronger
claim.

Each card records, where the source makes it available, origin/date,
motivation, mechanism, expected benefit, evidence, counter-evidence,
dependencies, implementation location, failure modes, cheapest falsifier,
related ideas, provenance/originality status, and next experiment. Missing
fields are intentionally marked unknown or not established rather than filled
from inference.

Research-only source material is not a runtime dependency. The production HAR
path remains the Rust-only tree under [`../har/`](../har/); no research note,
Python prototype, C/C++ source, llama.cpp/GGML tree, CMake component, or
foreign execution backend is loaded by the cargo-built runtime.
