# Hardware support

The GPU design targets Vulkan-capable devices and keeps the driver boundary
outside the Rust source. Shader availability, subgroup behavior, memory
heaps, queue families, and timestamp support are probed at runtime.

The RDNA4/gfx1200 work is an experimental target profile, not a guarantee for
every device with a similar marketing name. The current reference phenotype is
the Sapphire NITRO+ Radeon RX 9060 XT OC 16 GB on Mesa RADV, recorded in
[`HARDWARE_PROFILE.md`](../../HARDWARE_PROFILE.md). Hardware specialization is
intentional: gfx1200-specific kernels, subgroup widths, sparse instructions,
workgroup shapes, register pressure, cache layouts, quantization, and
VRAM/RAM/NVMe residency may be selected because they fit that machine.

Public benchmark receipts must record the device and driver, capability probe,
configured power/clock and PCIe/BAR state, workload telemetry, and phenotype
ID in a privacy-safe, reproducible form. Factory clocks are not actual clocks;
requested offsets are not sustained clocks; and a bounded kernel smoke is not
a full-model result. Receipts must not include serial numbers, hostnames,
network identifiers, UUIDs, mount paths, or private model paths.

Portability is a separate research problem. No result in this tree should be
assumed to reproduce on another AMD architecture, NVIDIA, Intel, or even
another RX 9060 XT without retuning and matched correctness evidence.
