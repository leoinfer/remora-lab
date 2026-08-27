# Benchmark methodology

The public benchmark harness is not complete in this candidate. The intended
receipt schema is:

```text
model identity + format
hardware/driver context
binary and source identity
command line and environment policy
prompt and generated-token count
warm-up and repetition policy
latency/tokens-per-second/energy
accepted and rejected speculative work
resident and transferred bytes
raw output digest and status
```

Comparisons must use the same model file, tokenizer behavior, prompt, output
length, sampling policy, and warm-up. Baselines are named explicitly. Until
these fields are public and reproducible, the result remains experimental or
historical.

## Exact hardware phenotype

This project is intentionally optimized around a reference workstation. A
benchmark receipt must include the phenotype ID
`RX9060XT16-NITRO-GFX1200-RADV-2026.08.27-v1` when it uses that machine, or a
new phenotype ID for another machine. The complete public-safe reference is
[`HARDWARE_PROFILE.md`](../HARDWARE_PROFILE.md).

Receipts must keep these categories separate:

- factory specification versus configured clock, power, firmware, and boot
  settings;
- observed idle values versus observed-under-load values;
- actual measured clock, power, temperature, PCIe, BAR, and residency samples
  versus requested settings; and
- current measurements versus historical settings or receipts.

The reference GPU is a Sapphire NITRO+ Radeon RX 9060 XT OC 16 GB (RDNA 4 /
gfx1200) with Mesa RADV. A kernel, quantization profile, scheduler policy,
workgroup width, cache layout, or VRAM/RAM/NVMe residency choice may be
deliberately specialized to that phenotype. A result from it is not evidence
for NVIDIA, Intel, another AMD architecture, or another nominally identical
card without a new capability probe, retuning, and matched correctness
measurement.

The current public Rust Vulkan smoke is a bounded correctness/telemetry check,
not a full-model performance receipt. Its current small-workload clock samples
must not be reported as sustained production clocks. Historical full-workload
clock observations are retained only as historical evidence and are not
interchangeable with the current live configuration.
