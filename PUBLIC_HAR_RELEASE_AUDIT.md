# HAR public release audit — expanded candidate

Status: `LOCAL_ONLY_RELEASE_CANDIDATE_PENDING_REVIEW`

This audit covers the expanded code-plus-research candidate. It has not been
published, pushed, or connected to a remote. The earlier private
code-centered snapshot remains frozen and unchanged; it is not this candidate.

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

The expanded publication audit passed:

```text
PUBLICATION_AUDIT PASS: 614 files, 3105496 bytes, no release-gate findings
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
and [`research/SOURCE_REGISTER.md`](research/SOURCE_REGISTER.md).

## Claims and review state

- No faster-than-llama.cpp claim is made.
- No full-model end-to-end or effective-10M-context claim is made.
- Flash-Next/R4F remains incomplete.
- Historical results remain labeled historical or experimental.
- Unsupported capability, stale state, and missing certificates fail closed.
- Second human review and owner approval are still pending.

Publication remains blocked. No remote is configured for this candidate, and
no push or publication action is authorized by this audit.
