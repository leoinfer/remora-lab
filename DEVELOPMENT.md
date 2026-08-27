# Development

This project is developed as a clean-room publication candidate. The fresh
history must contain only reviewed source and documentation from the
allowlist; private worktree history and internal coordination records are not
copied.

AI-assisted development was used during the research and migration process.
Generated or suggested changes are subject to human review, compilation,
tests, provenance checks, and release-gate audits. AI assistance does not
change copyright ownership, license obligations, or the standard for a claim.

## Required checks

1. Format and build the workspace with the pinned lockfile.
2. Run all Rust tests, recording ignored model tests separately.
3. Run the HAR Rust-only source gate.
4. Inspect Cargo metadata and the resolved dependency tree.
5. Run a native executable under syscall tracing with a disposable caller
   fixture.
6. Run the publication audit, secret scan, license scan, and object scan.
7. Perform a second adversarial review against the denylist and claims ledger.

For any hardware result, also record the phenotype ID from
[`HARDWARE_PROFILE.md`](HARDWARE_PROFILE.md), exact GPU/CPU/driver identity,
factory-versus-configured clock state, observed telemetry, memory/storage
residency, and all performance-affecting environment and boot flags. Do not
reuse a historical receipt as a current result when those fields differ.

For a model-free runtime trace, compile `tools/make_tiny_gguf.rs` with the
Rust compiler, write the fixture outside the repository, and pass that path to
`tools/trace_native_runtime.sh`. The fixture is synthetic and must not be
treated as a model-quality result.

No check is allowed to turn a historical result into a current performance
claim. A benchmark receipt is publishable only when its source, hardware
context, command line, and raw output can be reproduced without private data.
