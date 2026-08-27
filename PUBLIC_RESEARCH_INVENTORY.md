# Public research inventory

This inventory is the sanitized result of a full local archaeology pass over
Git repositories, linked worktrees, non-Git research trees, archives, and
recent source material. Exact local paths, host identity, branch names, agent
records, model locations, and raw receipts are retained only in a private
audit; they are not publication content. The original code-centered candidate
was expanded into this private GitHub staging candidate after the publication
stop; this repository is not public.

The inventory is broader than the current public tree. “Summarize” means that
the research question or result is represented by a fresh note. “Blocked”
means that no source or data is copied until provenance, license, or privacy is
resolved. “Excluded” means it is outside the local-inference publication
scope.

## Included in this candidate

| Source ID | Kind | Research area | Action | Destination |
| --- | --- | --- | --- | --- |
| `har-runtime-private` | Git/worktree | HAR Rust runtime migration | include reviewed Rust source | `har/` |
| `r4kv-local` | Git/worktree | compressed KV pages and profiles | include reviewed Rust source | `har/crates/r4kv` |
| `r4x-runtime` | non-Git research tree | R4X weight/block geometry | summarize and retain bounded parser | `formats/r4x`, `research/r4x` |
| `r4x-wide` | non-Git research tree | wider R4X execution experiments | summarize | `research/r4x` |
| `r4x-extreme` | non-Git research tree | aggressive quantization experiments | summarize with caveats | `research/r4x` |
| `flash-next-r4f` | non-Git research tree | Flash-Next bring-up | summarize; mark incomplete | `formats/r4f`, `research/flash-next` |
| `contextfold-codec-v0` | Git repository | compressed/effective context | summarize; omit mixed artifacts | `research/effective-context` |
| `warm-10m` | Git repository | effective-context target | summarize hypothesis | `research/effective-context` |
| `r4kv-research` | worktree/research tree | KV geometry and transfer accounting | include reviewed Rust portion | `formats/r4kv`, `har/crates/r4kv` |
| `expert-residency` | non-Git research tree | expert cache and residency | summarize; Rust pieces included in HAR | `research/moe-residency` |
| `remora-dualflow` | non-Git research tree | resource/reclaim accounting | summarize | `research/moe-residency` |
| `mtp-speculation` | non-Git/Git research | multi-token prediction and acceptance | summarize; scheduler code in HAR | `research/mtp-speculation` |
| `csd-qwen38` | non-Git research tree | model-shaped dense experiments | summarize; no model/data copy | `research/flash-next` |
| `csd26-block-mtp` | non-Git research tree | block speculation | summarize | `research/mtp-speculation` |
| `q38-mtp-challenges` | Git/research trees | acceptance and quality tests | summarize historical limits | `research/mtp-speculation` |
| `q6-regression` | non-Git research tree | quantization quality | summarize falsified/uncertain results | `docs/research/falsified-results.md` |
| `local-bench` | small Rust repository | model-free benchmark calibration | include reviewed MIT-licensed Rust tool; omit machine data and raw receipts | `benchmarks/local-bench` |

## Historical and blocked sources

| Source ID | Kind | Action | Blocking condition |
| --- | --- | --- | --- |
| `laguna-s-compression` | Git repository | high-level note only | no confirmed project license in the archaeology pass |
| `har-x` | Git/research tree | high-level note only | model, tokenizer, data, and training material need file-level rights review |
| `v4-flash-hardware-aware` | Git/research tree/archive | high-level note only | private metadata and historical receipts are not publication artifacts |
| `freetoken-autopsy` | non-Git research tree | cite upstream and summarize | raw reports contain internal coordination and private evidence |
| `agent-runtime-autopsy` | non-Git research tree | omit raw reports | internal labels, paths, and legacy hybrid recommendations |
| `rocmfpx-research` | Git/third-party tree | reference only | upstream/modified native code boundary not copied |
| `deep-spec-archive` | archive/Git tree | exclude | separate project and provenance not needed for this release |
| `historical-patch-sets` | archive | inspect only | duplicates or private context until file-level comparison |
| `executor-windtunnel` | archive | exclude from candidate | binary/archive payload and unresolved provenance |
| `external-hdd-local-ai-archive` | mounted-disk archive | inspect and exclude | archived model payloads, private paths, raw receipts, internal metadata, and generated reports |
| `external-hdd-source-small` | compressed source archive | exclude | opaque archive; legacy language/build boundaries and file-level licenses were not cleared |
| `external-hdd-r4x-plan` | model-derived manifest | exclude | model-specific plan and evidence, not a standalone public source artifact |
| `external-hdd-container-dump` | system backup archive | exclude | unrelated container backup outside the publication scope |

## Full-machine exclusions

Model directories, download caches, virtual environments, build outputs,
browser/application data, private vault material, unrelated trading/chess/
shopping projects, and third-party source clones were searched for inventory
purposes but are not publication inputs. No model weight, checkpoint,
tokenizer dump, dataset, screenshot, raw trace, or private receipt is copied.

## Expanded research corpus

The expanded candidate's canonical machine-readable index is
[`research_idea_index.json`](research_idea_index.json), with a human-readable
table in [`RESEARCH_IDEA_INDEX.md`](RESEARCH_IDEA_INDEX.md). It contains 280
distinct records:

- 39 HERMES mechanisms and 26 broader named families;
- 30 numbered manifest ideas;
- 12 open problems, 10 conjectures, 16 formal checker specifications, and
  28 counterexamples;
- 96 experiment queue entries; and
- 88 preserved registry records, including cross-family systems and
  scheduling, residency, speculation, and proof-carrying ideas.

The archival source documents are preserved under [`research/archival/`](research/archival/).
Section-level cards and thematic entry points are under
[`research/ideas/`](research/ideas/), [`research/systems/`](research/systems/),
[`research/open-problems/`](research/open-problems/),
[`research/conjectures/`](research/conjectures/),
[`research/experiments/`](research/experiments/),
[`research/falsified/`](research/falsified/), and [`research/theory/`](research/theory/).
The source register records which material was summarized, copied as reviewed
Rust, or excluded pending provenance, license, privacy, or model-payload
review.

## Inventory decision

The fresh candidate publishes the runtime and readable research map now. It
does not pretend that every local experiment is publishable. The omitted
source IDs remain useful work items and are linked to explicit provenance or
license decisions in [PROVENANCE.md](PROVENANCE.md). The separately mounted
HDD was checked for additional research material; its archive, model payloads,
receipts, and opaque source bundles remain outside this candidate.

The closure-pass coverage matrix is the authoritative whole-machine
crosswalk. It records every discovered unit within the declared archaeology
boundary, including material that is pending sanitization or explicitly
excluded: [`PUBLICATION_COVERAGE_MATRIX.md`](PUBLICATION_COVERAGE_MATRIX.md).
