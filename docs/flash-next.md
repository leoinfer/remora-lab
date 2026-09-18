# Flash-Next

Qwen3.8 Flash-Next is the current primary research target: a 125B-parameter
hybrid-attention MoE with roughly 6B activated parameters, 512 experts, 48 trunk
layers, and one native MTP layer, running on 16 GB of VRAM, 32 GB of RAM, and
NVMe.

**Current state (2026-09-18):** the model generates coherent text end to end on
the reference machine, passes its deployment canaries and sustained generation
tests, and holds a 262,144-token context in the Q8 KV configuration. The dated
record is [`research/flash-next/CURRENT_RESEARCH_LOG.md`](../research/flash-next/CURRENT_RESEARCH_LOG.md),
the campaign summary is [`research/flash-next/CURRENT_CAMPAIGN.md`](../research/flash-next/CURRENT_CAMPAIGN.md),
and the storage-path mechanism result is
[`research/flash-next/PREFETCH_RESULT.md`](../research/flash-next/PREFETCH_RESULT.md).

**Runtime boundary.** Most current Flash-Next experiments execute on
llama.cpp-derived research branches and the surrounding measurement tooling.
That is an external reference implementation used as a research platform: it is
not part of HAR and HAR does not depend on it.

**R4F** is the name used for an experimental native Flash-Next-oriented
container and execution direction inside HAR. The current candidate documents
the boundaries and admission vocabulary only. It does not claim a complete
format, native full-model generation, or a speedup. Its bounded disposition is
[`repro/flash-next/full-model/`](../repro/flash-next/full-model/).

Before a release can call the native R4F path ready it needs a frozen byte
layout, complete tensor coverage, recurrent-state numerical tests, rejected-work
recovery tests, public fixtures, and an end-to-end receipt. Until then, use the
R4F document as a research brief, not an interoperability promise.

**Representation.** Current Flash-Next representation work is recorded in
[`research/representation/`](../research/representation/): BF16 is the quality
authority, the deployed Q2 bank is a temporary control, and the final bank is
being regenerated from BF16 with heterogeneous per-role precision.
