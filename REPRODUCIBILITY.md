# Reproducibility

The public reproducibility contract is indexed by
[`repro_manifest.json`](repro_manifest.json). Each declared lane has a README,
machine-readable manifest, expected outcome, executable command, and receipt or
an explicit blocked disposition.

The recovered lanes are listed in [`repro/README.md`](repro/README.md). The
R4X receipt is authoritative historical evidence, while its public command is
Rust-only and validates the D32A representation without loading excluded model
weights. The historical rows are kernel/prefill diagnostic rows/s; they are not
generation tokens/s.

Run the environment check and the lane validation from the repository root:

```sh
./repro/setup/verify-environment.sh
./repro/r4x/width-sweep/run_width_sweep.sh
cargo run --locked --release -p repro-audit -- .
```

The release workflow also executes the model-free R4KV, effective-context,
MTP, n-gram, and SWMMAC lanes. The machine-readable artifact and claim
crosswalk is [`technical_artifact_index.json`](technical_artifact_index.json).
The Qwen historical baseline, R4KV quality frontier, and Flash-Next full-model
directories are executable bounded-disposition lanes; their commands report
the missing authority and intentionally produce no throughput or quality
number.

The exact foreign historical executable, Python research helpers, C/C++ source,
and model payloads are not runtime dependencies of HAR. They remain either
outside the public tree or in clearly marked archival/reference material. The
production boundary is `har/`, which remains Rust plus shader/SPIR-V and OS/
Vulkan driver interfaces.
