# Qwen3.8-27B ROCm ladder lane

Bounded public disposition for the 27B ROCm side campaign.

```sh
./repro/qwen27b/rocm-ladder-2026-09-22/run.sh
```

Raw `K0`, MTP-accepted decode, and prefill are never blended here. The measured
HIP ladder is 17.35 raw at `K0`, and 29.19 / 34.25 / 33.90 accepted at
`K=1/2/4` in a quiet session, with `K=2` the practical optimum.

The committed historical **42.11 t/s** belongs to a different artifact
(`UD-Q3_K_XL`), a different backend (Vulkan), and a different session — do not
merge it with 34.25.

`iq4_nl` KV has no HIP flash-attention kernel: attention falls back to the CPU
silently (3.9x slower prefill) and the bridge then faults. `q4_0` KV works. See
[`research/qwen27b/README.md`](../../../research/qwen27b/README.md).
