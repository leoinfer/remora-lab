# Public research source register

This register records what was recovered, how it was treated, and where the
public-safe representation lives. Exact local paths, host identity, raw model
payloads, private receipts, and internal agent coordination remain outside
this candidate.

| Source ID | Research material | Public action | Public destination | Originality / license boundary |
| --- | --- | --- | --- | --- |
| `hermes-v4-idea-atlas` | Complete HERMES-V4 mechanism atlas | include source text with path and hardware identifiers generalized | `research/archival/authoritative/HERMES_V4_COMPLETE_IDEA_ATLAS.md` | the project-originated atlas with later technical refinement; novelty not independently established |
| `complete-atlas` | H01–H39 and N01–N26 inventory | include and split into normalized cards | `research/archival/authoritative/COMPLETE_RESEARCH_ATLAS.md`, `research/ideas/atlas/` | local research record; status labels preserved |
| `remora-manifest` | 30 numbered ideas, TBEH, and PFM | include source and M01–M30 cards | `research/archival/authoritative/REMORA_NEW_IDEA_MASTER_MANIFEST.md`, `research/ideas/manifest/` | local research record; no novelty claim |
| `remora-open-problems` | OP-01–OP-12 portfolio | include source and individual records | `research/archival/authoritative/REMORA_OPEN_PROBLEM_PORTFOLIO.md`, `research/open-problems/` | formal research agenda; evidence labels retained |
| `remora-conjectures` | C-01–C-10 exact conjecture set | include source and individual records | `research/archival/authoritative/REMORA_NEW_CONJECTURES.md`, `research/conjectures/` | conjectures, not feature or novelty claims |
| `remora-formalization` | F0–F15 checker queue | include source and checker cards | `research/archival/authoritative/REMORA_FORMALIZATION_QUEUE.md`, `research/theory/checkers/` | CPU/static queue; live promotion remains gated |
| `remora-negative-knowledge` | CE-01–CE-28 counterexamples | include source and individual records | `research/archival/authoritative/REMORA_COUNTEREXAMPLE_LEDGER.md`, `research/falsified/counterexamples/` | negative evidence is scope-limited |
| `remora-experiment-queue` | E001–E096 protocols and states | include source and individual cards | `research/archival/authoritative/COMPLETE_EXPERIMENT_QUEUE.md`, `research/experiments/` | queue records are not results unless explicitly labeled |
| `remora-crosswalk` | discovery notes, synthesis graph, master prompt | include after path/identity scrub | `research/archival/authoritative/REMORA_DISCOVERY_NOTES.md`, `REMORA_CROSS_IDEA_SYNTHESIS_GRAPH.md`, `REMORA_MASTER_PROMPT_2026-08-03-1.md` | historical program records; source ranking gaps remain visible |
| `vault-idea-registry` | 88-row normalized idea registry | include sanitized JSONL and use as index input | `research/archival/vault/IDEA_REGISTRY.jsonl` | registry labels mapped to public status vocabulary |
| `contextfold-line` | ContextFold, ContextPack, RSSO, key-first, UARC, reclaim, certificate, and missing-data notes | include reviewed specifications and counterexamples | `research/archival/contextfold/`, `research/systems/ContextFold.md` | static/formal/source-audit lane; no real-KV or speed claim implied |
| `remora-architecture-notes` | PFM, TBEH, LayerPack, ExpertPack, transport, state and precision notes | include selected source documents | `research/archival/vault/`, `research/systems/` | local design records; implementation status remains explicit |
| `r4x-r4kv-r4f-notes` | quantization, KV, and Flash-Next format research | summarize/include format briefs; omit payloads and raw receipts | `formats/`, `research/r4x/`, `research/r4f/`, `research/archival/vault/` | model-specific and experimental; no universal quality claim |
| `residency-and-freetoken` | expert movement, cache, residency, and negative analyses | include public-safe reports; no upstream code copied | `research/archival/vault/`, `research/falsified/` | inspiration/reference separated from derived code in `PROVENANCE.md` |
| `warm-10m` | effective-context geometry, retrieval, leakage, and negative audits | include selected reports and ledgers; omit datasets/checkpoints | `research/falsified/`, `research/systems/ContextFold.md` | empirical scope and contamination caveats retained |
| `external-hdd-local-ai-archive` | mounted archive of local-AI artifacts | inspect metadata/member names; exclude payloads and raw receipts | none; disposition in `RESEARCH_CORPUS_STATUS.md` | private paths, model files, internal metadata, and raw evidence not cleared |
| `external-hdd-source-small` | compressed historical source pack | inspect archive member names only; exclude | none | opaque archive not cleared file-by-file for licenses or legacy build boundaries |
| `external-hdd-r4x-plan` | model-derived R4X planning manifest | inspect schema/idea relevance; exclude | none | model-specific plan, not a standalone redistributable source artifact |
| `external-hdd-container-dump` | unrelated system backup | exclude | none | outside research publication scope |
| `root-mega-source-register` | whole-machine source-register metadata | represent each entry in the closure matrix; do not copy private paths or hashes | `PUBLICATION_COVERAGE_MATRIX.md` | source metadata is evidence of discovery, not a license to redistribute contents |
| `root-mega-artifact-register` | mega-program artifact manifest and generated indexes | represent each registered artifact by ordinal and disposition | `PUBLICATION_COVERAGE_MATRIX.md` | model, binary, receipt, build, and private artifacts remain excluded by class |
| `machine-archaeology-trees` | non-Git research trees and top-level files | aggregate identity-preserving inventory only; no private paths | `research/archaeology/CLOSURE_PASS.md` | source-tree identity is withheld; file-level clearance is required for promotion |
| `har-worktree-inventory` | canonical HAR worktrees, archived worktrees, and branch metadata | aggregate 69 worktree records; publish only reviewed Rust already in `har/` | `PUBLICATION_COVERAGE_MATRIX.md` | dirty/private coordination state is not publication content |
| `flash-next-campaign-summary` | current R4F/Flash-Next bounded campaign | publish sanitized methodology, seams, failures, and gate state | `research/flash-next/CURRENT_CAMPAIGN.md` | raw receipts, model payloads, and private execution context withheld |
| `flash-next-campaign-rust` | current campaign Rust adapter and probe worktree | pending sanitization and provenance review; do not copy blindly | `PUBLICATION_COVERAGE_MATRIX.md` | dirty worktree and incomplete first-token gate |
| `flash-next-research-tooling` | current and historical research-only Python helpers | review only; no runtime or source copy | `research/tools/RESEARCH_TOOLING_REVIEW.md` | local paths, receipts, model payloads, and foreign-runtime boundaries require review |
| `flash-next-model-and-receipts` | R4F container, model-derived artifacts, raw receipts, and build outputs | exclude by payload/receipt class | `PUBLICATION_COVERAGE_MATRIX.md` | weights/data and raw execution evidence are not public-safe |

## Search boundary

The archaeology searched local research directories, Git/worktree inventories,
archived worktree references, markdown/text/JSON research state, experiment and
handoff records, and the mounted HDD's documentation/manifests and archive
member lists. Raw binary payloads, model weights, checkpoints, caches, logs
with private context, and opaque archives were not treated as publishable idea
sources.

The final normalized whole-machine crosswalk, including current campaign and
HDD records, is [`../PUBLICATION_COVERAGE_MATRIX.md`](../PUBLICATION_COVERAGE_MATRIX.md).
