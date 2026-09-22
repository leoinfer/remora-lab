# Host-KV production context lane

Bounded public disposition for huge-context production behaviour on the 27B
family.

```sh
./repro/host-kv/production-context-2026-09-22/run.sh
```

KV placement costs about 2.2x on decode at matched context, but the full window
cannot be device-resident: at 36.1 KB/token of `q8_0`, 262144 tokens needs ~9.0
GiB beside the weights. Two plausible fixes measured **null**: pinning the host
KV (+0%) and async host staging (0%, flag proven engaged). A Vulkan control on
the same shape decoded 5.38 t/s, i.e. on the host-KV path at full context ROCm
is faster than Vulkan — which kills the "port the Vulkan reuse kernel and the
speed comes back" plan for single-stream decode.

See [`research/host-kv/README.md`](../../../research/host-kv/README.md).
