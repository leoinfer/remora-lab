# Provenance and attribution

This file separates four relationships that are often conflated:

- `inspiration` — a public idea influenced a research question;
- `technical reference` — public behavior or a file/API specification was
  studied;
- `adapted implementation` — code or a concrete algorithm was modified;
- `derived code` — a distributed file contains material from the source.

The public candidate contains original local Rust implementations and fresh
documentation. No upstream inference implementation is copied into `har/`.
The model teams below are credited for model architecture and model artifacts;
that credit is not a claim that the HAR implementation is theirs.

The expanded candidate also contains a first-class research archive. The
HERMES, REMORA, ContextFold, R4X/R4KV/R4F, residency, and speculative-decoding
documents are preserved as local research records with their original names,
equations, evidence labels, counterexamples, and unresolved status. A source
document may include an original idea, an independent rediscovery, a technical
adaptation, or a synthesis influenced by prior work; the archive does not
assume novelty. See [`research/SOURCE_REGISTER.md`](research/SOURCE_REGISTER.md)
for the file-level inclusion boundary and
[`docs/ACKNOWLEDGEMENTS.md`](docs/ACKNOWLEDGEMENTS.md) for the attribution
request.

## Included or directly informing the candidate

| Source | Relationship | License / obligation | Included material | Confidence |
| --- | --- | --- | --- | --- |
| Local HAR runtime source (`har-runtime-private`) | derived code from the local research program | Apache-2.0 in this candidate; retain notice | Rust HAR crates, fresh public cleanup | confirmed by source review |
| Local R4KV source (`r4kv-local`) | derived code from the local research program | Apache-2.0 in this candidate; retain notice | `har/crates/r4kv` | confirmed by source review |
| Local benchmark source (`local-bench`) | adapted Rust research tool | MIT; notice in `benchmarks/local-bench/LICENSE` | `benchmarks/local-bench` without Git history, machine data, or raw receipts | confirmed by manifest and source review |
| Qwen team, [Qwen3](https://github.com/QwenLM/Qwen3) | model architecture and model-format reference | upstream model and code terms apply to the model user obtains; no model files copied | model-shaped loader/test language only | confirmed |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | technical comparison and file-format/oracle reference | MIT upstream license; no code copied and no dependency linked | historical comparison language and numeric format context | confirmed |
| [Vulkan specification](https://registry.khronos.org/vulkan/) | API and synchronization reference | Khronos specification terms; no specification text copied | API usage decisions | confirmed |
| [ash](https://github.com/ash-rs/ash) | Rust Vulkan binding dependency | MIT/Apache-2.0; Cargo resolves the exact version | Cargo dependency | confirmed |
| [FreeToken](https://github.com/FlashML-org/FreeToken) | inspiration and technical reference for expert/token movement | Apache-2.0 in the reviewed upstream tree; no code copied | research notes only | confirmed locally |
| [colibri](https://github.com/JustVugg/colibri) | inspiration and technical reference for large-MoE residency | Apache-2.0 in the reviewed upstream tree; no code copied | research notes only | confirmed locally |
| [dflash](https://github.com/z-lab/dflash) | technical reference for speculative decode | MIT; no code copied | research notes only | confirmed |
| [Medusa](https://github.com/FasterDecoding/Medusa) | technical reference for multi-head speculation | Apache-2.0 in the reviewed upstream tree; no code copied | research notes only | confirmed locally |
| [LayerSkip](https://github.com/facebookresearch/LayerSkip) | technical reference for early exit/self-speculation | CC BY-NC; no code or data copied; non-commercial terms block redistribution of derived material | research notes only | confirmed |
| [safetensors](https://github.com/huggingface/safetensors) | tensor-container reference | Apache-2.0 upstream; no code copied | format comparison only | confirmed |
| [Transformers](https://github.com/huggingface/transformers) | model implementation/reference vocabulary | Apache-2.0 upstream; no code copied | architecture comparison only | confirmed |
| [ROCm documentation](https://rocm.docs.amd.com/en/latest/about/license.html) | hardware/software licensing and capability reference | component-specific terms; no ROCm source copied | hardware notes only | confirmed |
| [Mesa RADV documentation](https://docs.mesa3d.org/drivers/radv.html) | Vulkan driver capability reference | Mesa licensing applies to Mesa source; no Mesa source copied | driver-boundary notes only | confirmed |

## Local research inputs that are not copied

| Source ID | Category | Decision | Reason |
| --- | --- | --- | --- |
| `contextfold-codec-v0` | effective-context/codec research | summarize only | mixed historical artifacts and a nested third-party runtime require file-level review |
| `warm-10m` | effective-context experiments | summarize only | raw checkpoints, datasets, and receipts are not publication inputs |
| `laguna-s-compression` | compression research | blocked pending license decision | source tree had no confirmed project license in the archaeology pass |
| `har-x` | model/training research | blocked pending license decision | contains model/data/training material without a confirmed redistribution license |
| `v4-flash-hardware-aware` | historical Flash-Next research | summarize only | publication archive contains private metadata and unverified historical receipts |
| `rocmfpx-research` | third-party implementation study | reference only | includes upstream and modified native code; no derived files copied |
| `upstream-inference-clones` | third-party source trees | excluded | upstream implementations are not part of the HAR runtime |
| `external-hdd-local-ai-archive` | mounted-disk archive | excluded | archive contains model payloads, private paths, raw receipts, internal metadata, and generated reports |
| `external-hdd-source-small` | compressed source archive | excluded | opaque archive not cleared for legacy language/build boundaries or file-level licenses |
| `external-hdd-r4x-plan` | model-derived manifest | excluded | model-specific plan/evidence rather than a standalone redistributable source artifact |
| `external-hdd-container-dump` | system backup archive | excluded | unrelated container backup outside the research publication scope |

Unknown provenance is a blocker for inclusion. At the time of this candidate,
`unknown_provenance_files_remaining` is zero for files actually present in the
tree; the blocked source IDs above are not represented by copied files.

## Attribution rules for future changes

Every adapted or derived file must name its source, exact license, license
notice location, and the change boundary in this document and
`provenance.json`. Model architecture credit must remain separate from
runtime implementation credit. If a similarity review cannot distinguish
independent implementation from derivation, omit the file until it can.
