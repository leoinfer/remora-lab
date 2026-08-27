# Quantization

HAR reads a small set of explicitly identified tensor layouts. Quantization
type identifiers come from the model-file format and are decoded by Rust
implementations in the workspace. The production path does not delegate to a
third-party inference library.

The CPU implementations are useful for deterministic correctness checks. The
Vulkan shaders are separate kernels with explicit block geometry and shader
identity. Any new format must add bounds tests, a byte-level fixture, a
dequantization comparison, and a provenance entry before it is admitted.

The included tests use small synthetic fixtures, not weight files. Caller
provided captures remain outside the publication tree and must be reviewed
before being used as evidence.
