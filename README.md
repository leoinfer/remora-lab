# Local AI Research

This is the expanded local release candidate for an open-source local-AI
research stack. It is prepared as a fresh repository and is not published by
this workspace. The tree contains both the Rust runtime and a first-class,
public-safe research archive; no model weights, private experiment receipts,
machine identifiers, or copied upstream execution backends are included.

## HAR

`har/` is the production runtime. Its host and runtime code is Rust. GPU
shaders are the only bundled non-Rust program material; Vulkan and operating
system libraries are reached through Rust bindings. The runtime has one
execution policy: native Rust, with missing kernels and stale state rejected
closed. Python, C or C++ host code, llama.cpp, GGML execution libraries,
CMake-built components, subprocess helpers, and foreign inference backends
are not part of the HAR production tree.

HAR currently provides native CPU model paths, a Rust Vulkan layer, GGUF
metadata/tensor loading, serving/scheduling components, R4KV research code,
and small synthetic metadata fixtures. Model files are supplied by the
caller and are never bundled.

## Research stack

The [`research/`](research/) library is a first-class map of the local research
program: HERMES/REMORA mechanisms, ContextFold and effective context,
speculative decode, expert residency, R4X/R4KV/R4F formats, Flash-Next,
Laguna, HAR-X, and benchmark methodology. The canonical map is
[`RESEARCH_IDEA_INDEX.md`](RESEARCH_IDEA_INDEX.md) plus
[`research_idea_index.json`](research_idea_index.json). The inventory records
material deliberately omitted because its provenance, license, privacy, or
reproducibility status is not ready for publication.

`benchmarks/local-bench/` is the included MIT-licensed Rust estimator. It is a
research-only tool and is not a HAR runtime dependency; its optional external
measurement harness is outside the production boundary.

Research-only Python may be used in a separate offline workspace, but it is
not an inference dependency and no Python source is currently required to
build or run HAR from this tree.

## Status and honesty

This is experimental research software. It does not claim to be faster than
llama.cpp. Some comparable paths have historically trailed it by several
tokens per second; those observations remain historical until a public,
reproducible benchmark is added. See [CLAIMS.md](CLAIMS.md),
[RESEARCH_STATUS.md](RESEARCH_STATUS.md), and the falsified-results record.

The “10M context” line is an effective-context research target, not a claim
that a dense attention cache or a single model file has been run at ten
million tokens. Flash-Next is a bring-up track, not a completed full-model
generation result.

## Build and audit

From the repository root:

```text
cargo build --workspace --release
cargo test --workspace --locked
cargo run -p publication-audit -- .
```

For HAR-specific checks, run the Rust-only source gate and the native runtime
trace described in [har/README.md](har/README.md). From the repository root:

```text
rustc tools/check_rust_only_runtime.rs -o /tmp/har-rust-only-runtime
/tmp/har-rust-only-runtime har
```

The trace must be run with a caller-supplied model fixture on a machine with
the required Vulkan driver; this candidate contains no model payload.

The earlier code-centered private snapshot is not the complete research
release. This expanded candidate is local-only and has no remote configured;
publication remains blocked until the larger corpus passes its own audits.
The current runtime evidence is summarized in
[PUBLIC_HAR_RELEASE_AUDIT.md](PUBLIC_HAR_RELEASE_AUDIT.md), while the broader
candidate state is recorded in
[RESEARCH_CORPUS_STATUS.md](RESEARCH_CORPUS_STATUS.md).

Read [PROVENANCE.md](PROVENANCE.md) and
[docs/ACKNOWLEDGEMENTS.md](docs/ACKNOWLEDGEMENTS.md) before redistributing
derived work.
