# Research-only tooling review

Python was reviewed as archaeology input, not as a production dependency.
The inspected families included Flash-Next/R4F probes and converters,
context and effective-context experiments, dense/MTP experiments, R4X
analysis, and Qwen-shaped measurement helpers.

The scripts are primarily model-specific converters, probe drivers, receipt
generators, and offline analysis. Several refer to local absolute paths, raw
receipts, model payloads, or nested foreign runtimes. They are therefore not
copied byte-for-byte into this candidate. The coverage matrix records each
research-tooling boundary as `PENDING_SANITIZATION` or `PENDING_PROVENANCE`
with a follow-up rather than implying that the source was lost.

The public candidate contains zero Python files. No Python process is started
by HAR, and no Python package is a Cargo dependency. Safe Rust replacements
and public equivalents are the reviewed runtime crates, the Rust
`benchmarks/local-bench` estimator, the Rust audit tools, and the bounded
fixtures described in the research notes.

Research-only tooling may be published later under an explicitly separate
path after file-level provenance, license, privacy, and reproducibility
review. It must never become an inference dependency.
