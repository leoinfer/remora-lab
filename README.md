# REMORA Lab

REMORA Lab is an open local-AI systems research repository built around
hardware-specific inference on consumer machines. It explores how large
models can become more practical through Rust runtimes, quantization, memory
hierarchies, scheduling, speculation, and experimental GPU kernels.

The reference development machine currently centers on a 16 GB RDNA4 Radeon,
so many measured results are hardware-specific rather than portability claims.

## In 30 seconds

- **What:** an umbrella repository for the HAR runtime and related local-AI
  systems research.
- **Why:** to make constrained-hardware inference measurable, inspectable,
  and reproducible enough to improve one experiment at a time.
- **Current:** HAR is a Rust-only production runtime with a Rust Vulkan path;
  the surrounding research records formats, memory systems, schedulers,
  speculation, GPU experiments, falsified results, and open questions.
- **Not claimed:** universal portability, a finished full-model Flash-Next
  path, a dense 10M-token context, or a performance win over llama.cpp.

Start with the [research idea index](RESEARCH_IDEA_INDEX.md),
[implementation map](research/implementation-map.md), [claims ledger](CLAIMS.md),
[falsified work](research/falsified/), [methodology](docs/methodology.md),
[hardware profile](HARDWARE_PROFILE.md), and [provenance record](PROVENANCE.md).

This is an umbrella repository, not a taxonomy rewrite. HAR remains the
runtime. REMORA remains its own research and control family. The repository
keeps distinct lines of work visible: HERMES, REMORA, ContextFold,
multi-token prediction and speculation, MoE residency, R4X/R4KV/R4F, the
gfx1200/RDNA4 track, Flash-Next, Laguna, HAR-X, open problems, conjectures,
falsified experiments, and moonshot research.

## Research stance

High-upside ideas are written as testable hypotheses. An implausible result is
neither accepted because it is exciting nor rejected because it conflicts with
an expectation; the discrepancy becomes the experiment. Failed experiments
are research output when they expose a real mechanism or improve a falsifier.
Some work is original to this project, some is inspired or overlapping, and
the repository does not treat originality as proof of superiority.

**Do not reject a moonshot because it sounds impossible. Do not accept it
because it sounds exciting. Try to kill it.**

## HAR

`har/` is the production runtime. Its host and runtime code is Rust. GPU
shaders are the only bundled non-Rust program material; Vulkan and operating
system libraries are reached through Rust bindings. The runtime has one
execution policy: native Rust, with missing kernels and stale state rejected
closed. Python, C or C++ host code, llama.cpp, GGML execution libraries,
CMake-built components, subprocess helpers, and foreign inference backends
are not part of the HAR production tree.

HAR currently provides native CPU model paths, a Rust Vulkan layer, GGUF
metadata/tensor loading, serving and scheduling components, R4KV research
code, and small synthetic metadata fixtures. Model files are supplied by the
caller and are never bundled.

Research-only Python may be used in a separate offline workspace, but it is
not an inference dependency and no Python source is required to build or run
HAR from this repository.

## Research stack

The [`research/`](research/) library is a public research map for HERMES and
REMORA mechanisms, ContextFold and effective context, speculative decode,
expert residency, R4X/R4KV/R4F formats, Flash-Next, Laguna, HAR-X, and
benchmark methodology. The canonical map is
[`RESEARCH_IDEA_INDEX.md`](RESEARCH_IDEA_INDEX.md) plus
[`research_idea_index.json`](research_idea_index.json). The inventory records
material deliberately omitted because its provenance, license, privacy, or
reproducibility status is not ready for publication.

`benchmarks/local-bench/` is the included MIT-licensed Rust estimator. It is a
research-only tool and is not a HAR runtime dependency; its optional external
measurement harness is outside the production boundary.

## Hardware-specific research

This project was built primarily around a reference workstation rather than
around a promise of portable peak performance. The reference GPU is a
Sapphire NITRO+ Radeon RX 9060 XT OC 16 GB (RDNA 4 / `gfx1200`), and many
kernels, quantization decisions, memory policies, and benchmarks were designed
around that machine. Some ideas are architecture-independent, but measured
performance is not. Do not assume a result reported here will reproduce on
NVIDIA, Intel, another AMD architecture, or even another RX 9060 XT without
retuning. Exact public-safe specifications, software versions, overclock
configuration, and benchmark environment are recorded in
[`HARDWARE_PROFILE.md`](HARDWARE_PROFILE.md).

The profile is a phenotype, not a portability guarantee. Factory
specifications, live configuration, idle samples, bounded workload samples,
and historical workload observations are labeled separately. Future receipts
must identify the phenotype and preserve their own command, environment,
residency, correctness, and raw-output evidence.

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

The closure crosswalk is [`PUBLICATION_COVERAGE_MATRIX.md`](PUBLICATION_COVERAGE_MATRIX.md),
with the implementation navigation in
[`research/implementation-map.md`](research/implementation-map.md) and the
current Flash-Next status in
[`research/flash-next/CURRENT_CAMPAIGN.md`](research/flash-next/CURRENT_CAMPAIGN.md).

The artifact-level publication map is
[`TECHNICAL_ARTIFACT_INDEX.md`](TECHNICAL_ARTIFACT_INDEX.md); it links
implementations to executable lanes, receipts, historical reconstructions,
and explicit blockers.

## AI-assisted development

Implementation of this repository has been heavily AI-assisted. The owner
directs the questions, ideas, architecture, experiments, validation,
benchmarking, falsification, and final decisions; AI agents provide much of
the engineering workforce. The work is open so systems and GPU developers can
inspect it, criticize it, reproduce it, rewrite it, and improve it. AI
assistance is not evidence that an implementation or result is correct.

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
the required Vulkan driver; this repository contains no model payload.

The current runtime evidence is summarized in
[PUBLIC_HAR_RELEASE_AUDIT.md](PUBLIC_HAR_RELEASE_AUDIT.md), while the broader
candidate state is recorded in [RESEARCH_CORPUS_STATUS.md](RESEARCH_CORPUS_STATUS.md).

Read [PROVENANCE.md](PROVENANCE.md),
[docs/ACKNOWLEDGEMENTS.md](docs/ACKNOWLEDGEMENTS.md), and
[THIRD_PARTY.md](THIRD_PARTY.md) before redistributing derived work.
