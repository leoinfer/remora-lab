# HAR Roadmap

Repository milestones are **M0–M12** (GitHub milestones, no dates). The
engineering dependency DAG is **H1–H16** (`HAR_DEPENDENCY_DAG.json`).
Milestone completion is evidence-based only; a milestone closes when its
acceptance evidence exists in the repository, never on a calendar.

## Milestones

| ID | Name | Meaning | Evidence required to close |
|---|---|---|---|
| M0 | Repository Foundation | Private repo, governance, issues/PRs, CI, hygiene | This repository state; CI green on the bounded matrix; governance docs accurate |
| M1 | First HAR Plan Compiled | The `.har` language front end compiles a real Qwen Q4 program to a physical plan | `har-compile` output for `QWEN_Q4_REFERENCE.har`; schema-valid plan; conformance record |
| M2 | First Real Operation Replayed | One captured llama.cpp operation replays under HAR-owned control flow | Differential fixture verdict `PASS_EXACT` for one captured op |
| M3 | First Exact Q4 Token | One deterministic token produced under Rust-owned control flow, exact vs reference | Token-by-token differential fixture, `PASS_EXACT` |
| M4 | First Native RDNA 4 Kernel | First HAR-owned Vulkan compute operation validated on RX 9060 XT | Hardware-locked GPU test result + captured manifest |
| M5 | Q4 Reference Parity | Q4 decode matches reference within the agreed contract | Bounded differential suite, exact/tolerance verdicts |
| M6 | Q4 Measured Performance Win | First official Agent 1 measurement showing a real win | Agent 1 manifest with frozen evidence; no extrapolation |
| M7 | Exact NVMe-to-VRAM Expert Slice | One expert projection transferred and executed from a persistent VRAM slot, exact | Residency fixture + GPU-locked test |
| M8 | Real ContextFold KV Roundtrip | Real Q8 KV page captured, codec-applied, reconstructed, exact roundtrip | KV roundtrip fixture `PASS_EXACT`; quality table |
| M9 | Q8 Bounded Decode | Q8 decode within the agreed correctness bound and memory bound | Bounded decode fixture + memory budget record |
| M10 | DeepSeek-V4 Bounded Decode | DeepSeek-V4 decode within agreed bounds (MoE streaming path) | Bounded decode fixture + residency records |
| M11 | Private Alpha | Internal alpha: build on a clean machine, known limitations documented | Alpha checklist (`docs/maintainers/RELEASE_CHECKLIST.md`) |
| M12 | Public Readiness | All public-release gates pass | `docs/maintainers/PUBLIC_RELEASE_GATE.md` checklist |

## Engineering DAG (H-series)

The H1–H16 DAG in `HAR_DEPENDENCY_DAG.json` is the technical dependency
map (compiling core → phenotype → replay → decode → Vulkan → residency →
KV → parity → performance). M-series milestones are the repository-level
evidence gates that map onto the DAG; see the DAG file for per-node state
and owners.

## Current state

- M0 in progress (this foundation).
- H1–H16: mostly `READY_BUT_BLOCKED` / `NOT_STARTED` — no committed
  baseline existed until now. See `HAR_GLOBAL_STATUS.md` for blockers
  B-001 … B-010.

## Non-goals until M5/M6 evidence

- No public performance numbers.
- No non-RDNA4 backend commitments.
- No feature-driven scope growth outside the idea registry.

Ideas live in `spec/research/idea-registry.json` (canonical IDs); the
roadmap only tracks committed milestones, not ideas.
