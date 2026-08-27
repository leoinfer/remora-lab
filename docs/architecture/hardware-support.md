# Hardware support

The GPU design targets Vulkan-capable devices and keeps the driver boundary
outside the Rust source. Shader availability, subgroup behavior, memory
heaps, queue families, and timestamp support are probed at runtime.

The RDNA4/gfx1200 work is an experimental target profile, not a guarantee for
every device with a similar marketing name. Public benchmark receipts must
record the device and driver in a privacy-safe, reproducible form, but must
not include serial numbers, hostnames, or network identifiers.
