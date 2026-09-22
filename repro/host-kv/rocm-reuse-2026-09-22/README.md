# Host-KV ROCm reuse lane

Bounded public disposition for the ROCm/HIP host-KV block-reuse kernel proof.

```sh
./repro/host-kv/rocm-reuse-2026-09-22/run.sh
```

Read the two arms separately. The pinned arm is the authoritative on-disk
receipt (13.997 to 244.944 GB/s logical across K=1..96); the registered-pageable
arm is the run-log-only arm and differs at K>=32 (4.50 vs 5.770 GB/s physical).
**They must not be averaged.**

The logical figures are block reuse, not wire speed. Physical host read stays at
the ~14 GB/s link roof through K=8 and then falls as the f32 kernel becomes
compute-bound; the Vulkan control held the roof to K=96 because it pays for
reuse with packed integer dots.

An unregistered pageable host buffer faults the GPU
(`Page not present or supervisor privilege`), so registration is a correctness
prerequisite. See [`research/host-kv/README.md`](../../../research/host-kv/README.md).
