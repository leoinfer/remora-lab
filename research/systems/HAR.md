# HAR

**Status:** `PARTIALLY_IMPLEMENTED`

HAR is the production runtime of this candidate. Its host and execution code
is Rust-only. Bundled GPU program material is limited to required shader
source and SPIR-V; Vulkan and operating-system driver libraries are the
platform boundary reached through Rust bindings.

The production contract excludes Python, C or C++ host execution,
llama.cpp, GGML execution libraries, subprocess helpers, C ABI inference
backends, CMake-built HAR components, and hidden fallback paths. Research
notes can describe historical experiments, but cargo-built HAR does not load
them or invoke them during inference.

## Scope

The runtime includes native CPU paths, Rust Vulkan integration, model
metadata/tensor loading, serving and scheduling crates, certificate and
residency control surfaces, and reviewed R4KV format code. It does not bundle
model weights, checkpoints, tokenizers, or datasets.

See the [`HAR README`](../../har/README.md),
[`Rust-only audit`](../../PUBLIC_HAR_RELEASE_AUDIT.md), and the
[`runtime roadmap`](../roadmap/HAR_ROADMAP.md). The research relationship is
summarized in [`HERMES`](HERMES.md), [`REMORA`](REMORA.md), and the
[`idea index`](../../RESEARCH_IDEA_INDEX.md).

## Evidence boundary

Source inspection alone is insufficient for the runtime claim. The release
process requires dependency-tree inspection, executable/runtime tracing, the
Rust-only source gate, and reproducible cargo checks. This local candidate is
publication-blocked until the expanded-tree audits are complete.
