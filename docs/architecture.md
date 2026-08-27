# Architecture

```text
caller-supplied model
        |
        v
  Rust model reader ----> Rust compiler/plan ----> Rust runtime policy
        |                                             |
        v                                             v
  Rust CPU kernels                         Rust scheduler/residency
                                                      |
                                             Rust Vulkan bindings
                                                      |
                                           Vulkan driver + shaders
```

HAR owns the runtime path inside the `har/` boundary. The root `research/`
tree is explanatory and historical; it cannot be imported by the runtime at
build or execution time.

The runtime deliberately has no foreign inference seam. A “reference” value
in a Rust test is a correctness oracle or comparison record. It is not an
alternative executable backend. Native policy admission rejects fallback
counters and reference-adapter invocations.

The model reader is bounded and format-aware. The compiler emits immutable
plans with model, hardware, and configuration identities. The scheduler owns
batching, page lifetimes, speculative acceptance, and telemetry. The Vulkan
layer owns resource and submission details while the driver performs the
platform-specific work.
