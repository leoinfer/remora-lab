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
- [`../docs/remora_metabolism/PROVENANCE_AND_SCOPE.md`](../docs/remora_metabolism/PROVENANCE_AND_SCOPE.md)
  — frozen public scope and verification map for the native REMORA metabolism
  subsystem;
- [`alice/`](alice/) — the 2026-09-19 → 2026-09-22 Alice campaign: a second
  model family, its MTP rollback defect, backend records, prefill, and host
  arena;
- [`host-kv/`](host-kv/) — block-stationary host-KV reuse: physical host read
  at the link roof, logical service by reuse, and the registered-memory
  prerequisite;
- [`qwen27b/`](qwen27b/) — the Qwen3.8-27B ROCm side campaign: ladder, KV
  format defects, and wide-M verification economics;
- [`ssd-action-memory/`](ssd-action-memory/) — symbolic action memory on the
  cold tier, including the retraction of its 5.10× headline;
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

Research-only source material is not a runtime dependency. The native Rust-only
tree under [`../har/`](../har/) is a research side lane rather than the runtime
of record for the program: no research note, Python prototype, C/C++ source,
llama.cpp/GGML tree, CMake component, or foreign execution backend is loaded by
the cargo-built HAR runtime, and correspondingly HAR is not the runtime that
executes most current model experiments. Those execute on external
llama.cpp-derived research branches.
