# Model support

The current public candidate is intentionally narrow:

- model-free toy and synthetic serving paths for scheduler tests;
- bounded native Rust dense Qwen-shaped and hybrid-MoE-shaped CPU paths when
  the caller supplies compatible GGUF files;
- experimental R4X row geometry and R4KV page/codec components;
- Vulkan resource and kernel seams with small fixtures.

“Qwen-shaped” describes the implemented equations and file metadata needed by
the bounded path. It does not claim ownership of the upstream model
architecture. See [PROVENANCE.md](../../PROVENANCE.md) for model attribution.

Unsupported model features must be rejected or reported as experimental. A
missing kernel never selects a hidden foreign backend.
