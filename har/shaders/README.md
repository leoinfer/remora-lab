# HAR shader assets

The `.comp` files are reviewed compute-shader sources. The matching `.spv`
files are generated Vulkan shader binaries used by the Rust smoke test. They
are the intentional non-Rust exception in the HAR production tree; all host
resource creation, dispatch, synchronization, and result validation remain
Rust code.

To regenerate a binary from its source with the Vulkan SDK:

```text
glslc -fshader-stage=compute har/shaders/greedy_argmax.comp \
  -o har/shaders/greedy_argmax.spv
```

Regeneration must be followed by the shader hash test and the Vulkan smoke
test on the target device. Do not add generated host code or a foreign kernel
runtime.
