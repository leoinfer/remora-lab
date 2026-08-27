# Research status

| Area | Status | What is supported | What is not claimed |
| --- | --- | --- | --- |
| HAR native CPU | experimental | Rust GGUF readers and bounded dense/MoE paths | production-grade model coverage |
| HAR Vulkan | experimental | Rust Vulkan resource/dispatch layer and shader fixtures | end-to-end full-model throughput |
| R4X | research format | D32A geometry and parser checks | stable ecosystem compatibility |
| R4F | bring-up | architecture notes and admission vocabulary | complete Flash-Next implementation |
| R4KV | research codec | Rust profiles, pages, capture validation | frozen public wire compatibility |
| Effective context | hypothesis/experiment | accounting models and explicit caveats | dense attention at 10M tokens |
| MTP/speculation | experimental | scheduler and acceptance contracts | universal speedup |
| Expert residency | experimental | Rust page/residency accounting | full-model GPU residency proof |
| Laguna/HAR-X | historical input | sanitized inventory and open questions | source redistribution before license review |

Performance claims are tracked separately in [CLAIMS.md](CLAIMS.md). A result
without a public receipt remains historical or experimental.
