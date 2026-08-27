# HAR public release audit — REMORA Lab

Status: `PUBLIC`

This audit covers the expanded code-plus-research candidate published in
`leoinfer/remora-lab`. The repository itself is the research publication; no
versioned release or Pages deployment has been created. The earlier
code-centered snapshot remains frozen and unchanged; it is not this
candidate.

The final closure crosswalk is [`PUBLICATION_COVERAGE_MATRIX.md`](PUBLICATION_COVERAGE_MATRIX.md)
and [`publication_coverage_matrix.json`](publication_coverage_matrix.json).
The current Flash-Next campaign is summarized in
[`research/flash-next/CURRENT_CAMPAIGN.md`](research/flash-next/CURRENT_CAMPAIGN.md).
The matrix contains 1,092 normalized records with `UNACCOUNTED = 0`, and
maps HAR, R4X, R4KV, and R4F research/implementation boundaries explicitly.

## Scope and publication gates

`har/` is the production runtime. Its host and runtime implementation is Rust;
the only bundled non-Rust program material is reviewed GPU shader source and
SPIR-V. Vulkan and operating-system libraries are reached through Rust
bindings. The production path contains no Python, C++, llama.cpp, GGML,
CMake-built HAR component, subprocess inference helper, C ABI execution
backend, or hidden foreign fallback.

The `research/` tree is a public-safe research library and is not a runtime
dependency. `benchmarks/local-bench/` is a separately licensed, research-only
Rust estimator. It is not a HAR runtime dependency and HAR never invokes its
optional external measurement harness.

## Hardware phenotype publication pass

The reference-machine phenotype is recorded in
[`HARDWARE_PROFILE.md`](HARDWARE_PROFILE.md) and
[`hardware_profile.json`](hardware_profile.json) as
`RX9060XT16-NITRO-GFX1200-RADV-2026.08.27-v1`. The profile is intentionally
machine-specific: the target is a Sapphire NITRO+ Radeon RX 9060 XT OC 16 GB
(RDNA 4 / `gfx1200`) with a Ryzen 7 3700X, CachyOS, and Mesa RADV. It does not
promise portability to another vendor, architecture, driver stack, memory
system, or nominally identical card.

The profile separates factory specifications from live controls, idle samples,
bounded Rust workload telemetry, historical full-workload observations, and
unknowns. Live checks established Sapphire PCI identity `1002:7590` with
subsystem `1da2:e493`, active Above-4G/ReBAR with a full 16-GiB-class VRAM BAR,
and the current GPU controls (`auto`/`BOOTUP_DEFAULT`, +230 MHz SCLK offset,
1,450 MHz MCLK control, -80 mV VDDGFX offset, 200 W cap). A fresh bounded
Rust Vulkan smoke passed; its small workload measured 1,554–2,354 MHz SCLK,
456 MHz MCLK, 18–19 W, and at most 36 C junction temperature. That is a
correctness/telemetry smoke, not a full-model performance result.

Historical private full-workload samples include 3,407–3,651 MHz SCLK and
separate approximately 3.72 GHz observations, but their phase timestamps or
sensor source are not sufficient for a current sustained-clock claim. The
current full-model sustained clock, power, and thermal envelope remain
unknown. Current live GPU `auto`/CPU `schedutil` state also differs from the
historical tuned preset that used GPU compute/high, CPU `performance`, PCIe
ASPM performance, and `iommu=pt`.

The mounted removable HDD was checked read-only during the broader corpus
pass; no separately cleared idea manuscript was found and no HDD content was
copied. The profile publishes no hostname, username, home path, serial,
UUID, MAC/IP address, private repository, model path, token, or key.

The expanded publication audit passed after the profile and documentation
changes:

```text
PUBLICATION_AUDIT PASS: 623 files, 4519194 bytes, no release-gate findings
```

The tree contains no model weights, checkpoints, tokenizer payloads, datasets,
raw receipts, screenshots, private paths, hostnames, or symlinks. The
machine-readable research index contains 280 distinct records and all indexed
public-record links resolve.

The mounted HDD was checked read-only for idea-bearing documentation, archive
member names, and manifests. No separately cleared idea manuscript was found;
model payloads, raw receipts, private archives, opaque source bundles,
model-derived plans, and unrelated backups remain excluded. The decision is
recorded in [`research/SOURCE_REGISTER.md`](research/SOURCE_REGISTER.md).

## Final adversarial privacy/de-anonymization audit

The final privacy gate covers every current tracked file, every blob and commit
reachable from the publication refs, commit author/committer/message metadata,
signed metadata, ref/tag visibility, and GitHub repository metadata. It checks
identity/contact markers, network and hardware identifiers, filesystem paths,
private keys, provider credential shapes, personal-content classes, and
temporary-path labels. Historical owner/path labels found during triage were
removed from the current corpus before the published history was finalized.

The Rust publication audit now includes conservative detectors and tests for
synthetic fake credentials, emails, IP/MAC/UUID/serial-shaped identifiers,
private-key headers, and suspicious assignments. Ordinary model token
terminology and PCI/software-version values remain accepted. The final result
is `PASS` with zero remaining sensitive findings; the detailed red-team report
is retained outside this repository and is not publication content.

## Rust-only validation

The following checks passed from this candidate:

```text
cargo fmt --all -- --check
cargo check --workspace --all-targets --locked --offline
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
cargo test --workspace --locked --offline
cargo build --workspace --release --locked --offline
rustc tools/check_rust_only_runtime.rs -o /tmp/har-rust-only-runtime
/tmp/har-rust-only-runtime har
cargo run -p publication-audit -- .
```

The Rust-only gate reported `RUST_ONLY_RUNTIME PASS: scanned 147 production
files`. The workspace tests passed with no failures; the serving crate has 54
passing tests and one intentionally ignored test requiring a caller-provided
GGUF. The separate MIT local-bench workspace also passed format, check,
strict Clippy, tests, and release build checks.

## Dependency and linked-object audit

The locked Cargo graph has no llama.cpp, GGML, Python, CMake, C/C++ execution,
or foreign inference package. The only loader edge requiring classification is:

```text
ash -> libloading
```

That edge is the Rust Vulkan loader boundary. HAR release binaries have no
direct `libvulkan` or `libstdc++` dependency; `readelf -d` shows only the
Rust/Linux baseline libraries (`libc`, `libm` where needed, `libgcc_s`, and the
ELF loader). The Vulkan smoke loader trace showed Rust dynamically loading
`libvulkan.so.1`, which selected the RADV driver. The driver loaded its own
system dependencies, including `libstdc++.so.6`; that is driver-owned system
code within the permitted OS/Vulkan boundary, not a HAR-linked C++ execution
backend.

## Runtime and GPU traces

`tools/make_tiny_gguf.rs` generated a synthetic Q4_0 caller fixture outside the
repository. `tools/trace_native_runtime.sh` loaded it with the release
`har-server`:

```text
NATIVE_RUNTIME_TRACE PASS
served token stream: [1, 2, 3, 0]
trace method: GDB execve/execveat catchpoints plus dynamic-loader trace
helper-process exec observed: no
```

The full `strace` process/file method was unavailable because `strace` is not
installed; this audit does not claim full syscall coverage. The GDB trace did
not observe a helper process, and the loader/result traces contained no
Python, CMake, llama.cpp, GGML, or C++ runtime dependency in the HAR process.

The direct Rust Vulkan smoke also passed against the checked-in SPIR-V:

```text
PASS_RUST_VULKAN sampler_token=777 q4_value=256
```

It exercised the RADV device, greedy sampling shader, Q4_K transfer/decode
path, timestamp queries, and queue submission. The observed device and driver
are execution-context facts, not a portability or throughput claim. The full
execution context, hostname, UTC timestamp, relevant working directories, and
command provenance are retained in the private audit companion outside this
candidate; they are deliberately not publication content.

## Research-corpus audit

The expanded candidate preserves:

- 39 HERMES mechanisms and 26 broader named families;
- 30 manifest ideas;
- 12 open problems, 10 conjectures, 16 formal checkers, and 28
  counterexamples;
- 96 experiment-queue entries; and
- 88 preserved idea-registry records.

The canonical map is [`research_idea_index.json`](research_idea_index.json),
with a human-readable view in [`RESEARCH_IDEA_INDEX.md`](RESEARCH_IDEA_INDEX.md).
Source, license, originality, and exclusion decisions are recorded in
[`PROVENANCE.md`](PROVENANCE.md), [`PUBLIC_RESEARCH_INVENTORY.md`](PUBLIC_RESEARCH_INVENTORY.md),
and [`research/SOURCE_REGISTER.md`](research/SOURCE_REGISTER.md). The
moonshot/anomaly policy and the invalidated gfx1200 sparse-matrix result are
preserved in [`docs/methodology.md`](docs/methodology.md) and
[`research/falsified/GFX1200_SPARSE_MATRIX_ANOMALY.md`](research/falsified/GFX1200_SPARSE_MATRIX_ANOMALY.md).

## Claims and review state

- No faster-than-llama.cpp claim is made.
- No full-model end-to-end or effective-10M-context claim is made.
- Flash-Next/R4F remains incomplete.
- Historical results remain labeled historical or experimental.
- Unsupported capability, stale state, and missing certificates fail closed.
- The final adversarial privacy/de-anonymization audit passed with zero
  remaining sensitive findings.
- The owner explicitly authorized publication in the final publication brief.

The repository is public. No versioned release or Pages deployment has been
created, and the runtime/model work remains experimental rather than a claim
of completion.
