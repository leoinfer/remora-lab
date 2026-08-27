# Hardware-Aware Runtime (HAR)

HAR is a Rust-only production runtime for local model execution. This
directory is intentionally separate from the broader research notes at the
repository root.

The production boundary contains Rust crates, reviewed GPU shader source,
small synthetic metadata fixtures, and caller-supplied model-file readers. It does
not contain a C/C++ implementation, a Python runtime, a CMake component, a
llama.cpp or GGML execution library, a subprocess bridge, or a foreign
inference backend. GGUF/quantization numeric identifiers are file-format
compatibility data; they do not imply a linked implementation.

## Native path

The default runtime policy is `NativeRequired`. A fallback counter or a
reference-adapter invocation makes admission fail. Rust CPU implementations
are used for correctness and bounded serving paths; the Vulkan crate owns the
GPU boundary through `ash`, and the shaders are dispatched by Rust code.

`ash` uses a dynamic loader for the platform Vulkan library. A driver may load
its own system dependencies (including a C++ runtime) after HAR hands control
to Vulkan; those driver-owned libraries are not HAR host code, a linked
inference backend, or a fallback path. The release audit distinguishes this
permitted OS/Vulkan boundary from HAR's Rust dependency graph.

Run a small server demo without a model:

```text
cargo run -p har-serve --bin har-server -- --model toy --max-new 2
```

Run a caller-provided dense model:

```text
cargo run -p har-demo --bin har-native-qwen3 -- /path/to/model.gguf \
  --prompt "Hello" --max-new-tokens 1
```

The path above is illustrative. No model path is embedded in this tree.

## Gates

From the repository root:

```text
rustc tools/check_rust_only_runtime.rs -o /tmp/har-rust-only-runtime
/tmp/har-rust-only-runtime har
```

The optional runtime trace uses `strace` when available and a `gdb` fallback;
it requires a model argument. It runs the cargo-built `har-server` against a
caller-supplied Q4_0 GGUF and generates one token:

```text
bash tools/trace_native_runtime.sh /path/to/model.gguf
```

It builds the native Rust binary, traces process/file activity, and rejects
Python, CMake, llama, GGML, C++ runtime libraries, or extra process launches
in the HAR process. The GPU-driver process boundary is recorded separately.
The trace is an audit aid and is not a production runtime dependency.
