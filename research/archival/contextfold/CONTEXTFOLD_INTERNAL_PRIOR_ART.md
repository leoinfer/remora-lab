# ContextFold internal prior art and non-reinvention record

## Existing local work

The active vault and `[local path omitted]` already contain static
RSSO, memory placement, MTP amortization, dense critical-path, and exact block
verification notes. Those documents are prior hypotheses and gates, not live
ContextFold evidence. In particular:

- `../RSSO_COST_MODEL.md`, `../RSSO_BREAK_EVEN_SIMULATION.md`, and
  `../RSSO_LAYERPACK_DESIGN.md` already model resident skeleton/block movement and
  explicitly gate implementation.
- `../QWEN36_27B_MEMORY_PLACEMENT_MAP.md` records the Qwen hybrid geometry and
  prior NGL20/24/26 placement observations.
- `Architecture/Exact-Block-Verification.md` records the target-authoritative
  hidden/recurrent/KV state requirement for candidate verification.
- The source tree has `llama_kv_cache`, `llama_memory_recurrent`,
  `llama_memory_hybrid`, `llama_kv_cache_dsa`, `llama_kv_cache_dsv4`, and
  `llama_kv_cache_iswa`; ContextFold does not rename any of these.

## Prior art classification

| Existing mechanism | What it already supplies | What ContextFold would add | Status |
|---|---|---|---|
| KV cache type/placement/offload | K/V tensor precision and CPU/backend placement | HOT/WARM/COLD lifecycle and movement accounting | static proposal |
| Flash Attention | resident tiled attention and local online accumulation | streamable immutable block manifest and cross-tier merge contract | static proof/prototype |
| unified KV and sequence cells | sequence sharing and batch layout | causal shared-prefix atlas with authenticated root | static proposal |
| prompt/state save | serialized runtime state and tokens | versioned ContextPack codec/validity closure | schema/prototype |
| recurrent rollback snapshots | bounded candidate suffix rollback | cold replay/checkpoint policy covering recurrent closure | scalar model |
| ISWA / DSA / DSV4 | architecture-specific cache layouts | generalized heterogeneous state abstraction | formal specification |
| RSSO | block-stationary reuse model | no novelty claim unless cold movement exceeds existing batching | negative/default gate |

## External-family caution

Paged attention, retrieval, KV compression, and learned memory are known broad
families, but this lane does not import an implementation or claim novelty. The
relevant comparison is functional: a generic page/block table would provide
indirection, whereas ContextFold requires causal identity, exact state closure,
lossless codecs, and replay/materialization policy. Approximate retrieval and
sparse skipping remain separate result classes.
