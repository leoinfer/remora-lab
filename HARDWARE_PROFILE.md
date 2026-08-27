# Hardware phenotype profile

This is the public-safe, versioned description of the reference machine used
for much of the local-AI research. It is a reproducibility boundary, not a
claim that the runtime or its results are hardware-neutral.

Reference phenotype ID: `RX9060XT16-NITRO-GFX1200-RADV-2026.08.27-v1`

Profile version: `1`

Capture date: `2026-08-27` (UTC)

HAR runtime commit at the bounded hardware measurement:
`c035aedab138a57b9078321f3914973105d3be4c`

## Hardware-specific disclaimer

> Much of this research was designed, tuned, and measured around one specific
> workstation, particularly its AMD Radeon RX 9060 XT configuration.
> Hardware specialization is intentional.
> Results should not be assumed to reproduce on different GPUs, different
> vendors, different driver stacks, different memory systems, or even another
> nominally identical graphics card without retuning.
> Where a hardware-specific execution strategy materially outperforms a
> generic one on the target machine, this project generally prefers the
> specialized strategy.
> Portability and generalization are separate research problems and should not
> be inferred from a result obtained on the reference machine.

## Status vocabulary

- `FACTORY SPECIFICATION` — manufacturer-published board or component data;
  not a measurement of the current machine.
- `CONFIGURED SETTING` — a live control, boot policy, or selected software
  setting observed on the reference machine.
- `OBSERVED ...` — a live identity, capacity, software, or capability fact;
  the more specific labels below retain the relevant observation boundary.
- `OBSERVED IDLE` — a point-in-time low-power or idle sample.
- `OBSERVED UNDER LOAD` — a measured workload sample with a stated workload
  boundary; it is not automatically a sustained or production result.
- `HISTORICAL SETTING` — a setting or receipt from an earlier private run.
- `UNKNOWN` — not established by the public-safe capture and intentionally not
  inferred.

## Reference system summary

| Component | Public-safe reference value | Status |
| --- | --- | --- |
| GPU | Sapphire NITRO+ Radeon RX 9060 XT OC 16 GB, RDNA 4 / `gfx1200` | OBSERVED IDENTITY / FACTORY SPECIFICATION |
| CPU | AMD Ryzen 7 3700X, 8 cores / 16 threads | OBSERVED IDENTITY / FACTORY SPECIFICATION |
| Platform | ASUSTeK GA15DH, ROG Strix GA15DH_G15DH, board revision 1.0 | OBSERVED IDENTITY |
| Operating system | CachyOS rolling, x86_64 | OBSERVED IDENTITY |
| Kernel | `7.2.0-rc7-1-cachyos-rc` | OBSERVED SOFTWARE |
| Vulkan | Mesa RADV on Vulkan instance 1.4.357; device API 1.4.354 | OBSERVED SOFTWARE |
| Primary storage | SK hynix `HFS001TDE9X081N`, about 953.9 GB, PCIe Gen3 x4 | OBSERVED IDENTITY |
| Memory | OS reports 33,575,051,264 bytes total | OBSERVED CAPACITY; MODULE DETAILS UNKNOWN |

## GPU identity and factory specification

The board identity was checked against PCI/sysfs and Vulkan. The board-level
factory values below come from the [Sapphire NITRO+ product page](https://www.sapphiretech.com/en/consumer/nitro-radeon-rx-9060-xt-16g-gddr6).
The architectural/reference values are cross-checked against
[AMD's Radeon RX 9060 XT specification](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx9060xt.html).

| Field | Value | Status |
| --- | --- | --- |
| Board | Sapphire NITRO+ Radeon RX 9060 XT OC 16 GB | FACTORY SPECIFICATION |
| Sapphire board SKU | `11350-01-20G` | FACTORY SPECIFICATION |
| GPU family | AMD Navi 44 / Radeon RX 9060 XT, RDNA 4 / `gfx1200` | OBSERVED IDENTITY / FACTORY SPECIFICATION |
| PCI vendor/device | `1002:7590` | OBSERVED IDENTITY |
| Sapphire subsystem | `1da2:e493` | OBSERVED IDENTITY |
| PCI revision | `c0` | OBSERVED IDENTITY |
| VBIOS version | `113-P816G493-N07` | OBSERVED IDENTITY |
| Compute units | 32 | FACTORY SPECIFICATION |
| Stream processors | 2,048 | FACTORY SPECIFICATION |
| Ray accelerators | 32 | FACTORY SPECIFICATION |
| AI accelerators | 64 | FACTORY SPECIFICATION |
| ROPs / texture units | 64 / 128 | FACTORY SPECIFICATION |
| Infinity Cache | 32 MB | FACTORY SPECIFICATION |
| VRAM | 16 GB GDDR6, 128-bit bus | FACTORY SPECIFICATION |
| Effective memory data rate | 20 Gbps | FACTORY SPECIFICATION |
| Theoretical memory bandwidth | up to 320 GB/s | FACTORY SPECIFICATION |
| Maximum bus | PCIe 5.0 x16 | FACTORY SPECIFICATION |
| Sapphire game clock | up to 2,780 MHz | FACTORY SPECIFICATION |
| Sapphire boost clock | up to 3,320 MHz | FACTORY SPECIFICATION |
| Sapphire typical board power | 182 W | FACTORY SPECIFICATION |
| Vendor minimum PSU recommendation | 450 W | FACTORY SPECIFICATION; not the installed PSU |

AMD's reference-board page lists lower reference clocks and 160 W typical
board power than the Sapphire factory profile. That difference is expected:
the Sapphire board specification is the relevant factory profile for this
phenotype, while neither set of numbers is a measured sustained clock.

## GPU configuration and measurements

GPU overclocking is intentionally split into factory specification, current
controls, measured samples, and historical settings.

### Current configured settings

| Control | Observed value | Status |
| --- | --- | --- |
| Performance level | `auto` | CONFIGURED SETTING |
| Power profile | `BOOTUP_DEFAULT` | CONFIGURED SETTING |
| Board power cap | 200 W | CONFIGURED SETTING |
| SCLK offset | `+230 MHz` | CONFIGURED SETTING |
| MCLK control | 1,450 MHz controller setting | CONFIGURED SETTING |
| VDDGFX offset | `-80 mV` | CONFIGURED SETTING |
| DPM SCLK endpoints | 500 / 2,780 MHz; active intermediate sample 1,402 MHz at capture | CONFIGURED SETTING / OBSERVED IDLE |
| DPM MCLK table | 96 / 456 / 772 / 875 / 1,124 / 1,258 MHz | CONFIGURED SETTING |

These controls do not prove that the GPU reaches the requested values on every
workload. The performance level is elastic and the `rocm-smi` and sysfs
interfaces can sample different power states.

### Current observed samples

An idle/low-power sample reported approximately 18 W graphics package power,
34 C edge temperature, 36 C junction temperature, 36 C memory temperature,
456 MHz MCLK, and PCIe 8.0 GT/s x16. A separate live sysfs sample reported
PCIe 32.0 GT/s x16. Both are point-in-time power-state observations; the
width was x16 in both cases.

The current bounded Rust Vulkan smoke used the release binary and a tiny
synthetic sampler/Q4_K operation for 600 repetitions. It ran for about nine
seconds from `2026-08-27T11:09:42.458Z` through
`2026-08-27T11:09:51.436Z`, produced 40 telemetry samples, and exited
successfully. The sample range was:

| Measurement | Result | Status |
| --- | --- | --- |
| SCLK | 1,554–2,354 MHz; mean 2,030 MHz | OBSERVED UNDER LOAD |
| MCLK | 456 MHz throughout | OBSERVED UNDER LOAD |
| Graphics package power | 18–19 W; mean 18.5 W | OBSERVED UNDER LOAD |
| GPU busy | up to 23% | OBSERVED UNDER LOAD |
| Junction temperature | up to 36 C | OBSERVED UNDER LOAD |

This is a correctness and telemetry smoke, not a representative full-model
load. It is not a sustained-clock, throughput, or energy-efficiency claim.

### Historical workload observations

An owner-authorized private R4X full-model benchmark window on 2026-08-24
sampled SCLK values from 3,407 to 3,651 MHz in 82 of 117 raw samples at or
above 3,300 MHz. The raw file includes idle rows and does not carry a
timestamp on every phase row, so this profile does not assert a phase-specific
sustained boundary. MCLK 1,449 MHz appears in raw rows from the same window.

Separate historical decode documentation reports approximately 3.72 GHz SCLK
with MCLK parked at 456 MHz. A different profiler reports a sensor-source
discrepancy against the DPM caps. Those are historical observations, not
current settings and not a promise that every workload sustains them.

The current full-model sustained clock, current full-model power, and current
full-model thermal envelope are `UNKNOWN`; they were not remeasured as part of
this bounded publication pass.

## PCIe, BAR, and memory visibility

| Field | Value | Status |
| --- | --- | --- |
| GPU maximum link | PCIe 5.0 x16 / 32.0 GT/s x16 | FACTORY/CAPABILITY OBSERVATION |
| Live sysfs link sample | 32.0 GT/s x16 | OBSERVED IDLE/CONFIGURED STATE |
| Low-power telemetry link sample | 8.0 GT/s x16 | OBSERVED IDLE |
| Above-4G decoding | active in a firmware-safe read-only check | CONFIGURED SETTING |
| Resizable BAR | active; full 16-GiB-class VRAM BAR visible | CONFIGURED SETTING / OBSERVED CAPACITY |
| Driver-reported VRAM total | 17,095,983,104 bytes | OBSERVED CAPACITY |
| Driver-reported visible VRAM total | 17,095,983,104 bytes | OBSERVED CAPACITY |
| Kernel policy flags | `pci=realloc=on amdgpu.rebar=1` | CONFIGURED SETTING |

The PCIe link rate can fall to a low-power state while retaining x16 width.
The BAR and visible-memory observations are not a measured transfer-bandwidth
claim.

## Vulkan and architecture capabilities

The observed device was `AMD Radeon RX 9060 XT (RADV GFX1200)` using Mesa RADV.
`vulkaninfo --summary` reported 26 instance extensions; the full device
extension manifest is not frozen in this profile because the runtime performs
capability probing at startup. The bounded Rust probe observed the following
relevant capabilities:

| Capability | Result | Status |
| --- | --- | --- |
| Vulkan instance / device API | 1.4.357 / 1.4.354 | OBSERVED SOFTWARE |
| Driver | Mesa RADV 26.2.1, driver version 26.2.1 | OBSERVED SOFTWARE |
| Relevant device extensions | `VK_KHR_cooperative_matrix`, `VK_KHR_shader_bfloat16` present | OBSERVED SOFTWARE |
| Subgroup | reported 64; range 32–64; Wave32 and Wave64 supported | OBSERVED UNDER LOAD / CAPABILITY PROBE |
| Integer dot product | available | OBSERVED CAPABILITY |
| FP16 type | available | OBSERVED CAPABILITY |
| Cooperative matrix | available | OBSERVED CAPABILITY |
| Timeline semaphore / synchronization2 | available / available | OBSERVED CAPABILITY |
| Subgroup-size control | available | OBSERVED CAPABILITY |
| Timestamp period | 10 ns | OBSERVED CAPABILITY |

These capabilities explain why gfx1200-specific kernels, subgroup widths,
cooperative-matrix experiments, register-pressure choices, and shader layouts
are part of the research. They do not imply portability to another GPU.
The smoke also emitted RADV's testing-only warning that it is not a conformant
Vulkan implementation; this profile makes no Vulkan conformance or performance
certification claim.

## CPU

The processor identity and live policy were read from CPU topology, cpufreq,
SMT, and microcode interfaces. The factory values are from the
[AMD Ryzen 7 3700X specification](https://www.amd.com/en/support/downloads/drivers.html/processors/ryzen/ryzen-3000-series/amd-ryzen-7-3700x.html).

| Field | Value | Status |
| --- | --- | --- |
| Processor | AMD Ryzen 7 3700X | OBSERVED IDENTITY |
| Cores / threads | 8 / 16 | OBSERVED TOPOLOGY / FACTORY SPECIFICATION |
| NUMA | one NUMA node | OBSERVED TOPOLOGY |
| Base / maximum boost | 3.6 / up to 4.4 GHz | FACTORY SPECIFICATION |
| Default TDP / max temperature | 65 W / 95 C | FACTORY SPECIFICATION |
| Frequency policy | 2.2–4.4 GHz | CONFIGURED SETTING |
| Scaling driver / governor | `acpi-cpufreq` / `schedutil` | CONFIGURED SETTING |
| Boost | enabled | CONFIGURED SETTING |
| SMT | enabled | CONFIGURED SETTING |
| Microcode | `0x8701034` | OBSERVED SOFTWARE |
| ISA subset observed | AVX2, FMA, F16C, BMI1/BMI2, AES, SHA | OBSERVED CAPABILITY |
| PBO / CPU overclock | not established | UNKNOWN |

An earlier historical performance preset used the `performance` CPU governor;
that is not the current live setting and must not be mixed with this phenotype.

## Memory

The operating system reported 33,575,051,264 bytes of total memory. DIMM
manufacturer, part number, installed-module count, rated speed, actual speed,
timings, channel mode, and XMP/DOCP state were not exposed by the
public-safe unprivileged capture and remain `UNKNOWN`.

Historical planning notes sometimes describe this platform as 32 GB DDR4,
but that note is not promoted to an exact current DIMM specification. Future
receipts must record the memory fields only when they are directly measured or
otherwise owner-cleared for publication.

## Storage and residency

| Device or layer | Public-safe value | Status |
| --- | --- | --- |
| Primary NVMe | SK hynix `HFS001TDE9X081N`, about 953.9 GB, firmware `41720C20` | OBSERVED IDENTITY |
| Primary link | PCIe 8.0 GT/s x4 (Gen3 x4) | OBSERVED IDENTITY |
| Filesystem | Btrfs user-data subvolume; `noatime`, zstd compression, SSD/discard options observed | CONFIGURED SETTING |
| Removable HDD | Toshiba `MQ01UBD050`, about 465.8 GB, USB rotational | OBSERVED IDENTITY |
| Swap layer | zram, about 31.3 GB configured | CONFIGURED SETTING |
| Model streaming location for each benchmark | must be recorded per receipt; not inferred from the mount table | UNKNOWN until receipted |

The removable HDD was inspected read-only for idea-bearing documentation,
manifests, and archive member names. No separately cleared idea manuscript was
found, and no HDD content was copied into this candidate. Serial numbers,
filesystem UUIDs, device UUIDs, and mount paths are deliberately omitted.

## Platform and firmware

| Field | Value | Status |
| --- | --- | --- |
| Board vendor/model | ASUSTeK GA15DH | OBSERVED IDENTITY |
| Product | ROG Strix GA15DH_G15DH | OBSERVED IDENTITY |
| Board/product revision | 1.0 | OBSERVED IDENTITY |
| BIOS | American Megatrends Inc. `GA15DH.303`, 2020-11-18 | OBSERVED FIRMWARE |
| Actual PSU capacity/model | not established | UNKNOWN |
| Actual cooler | not established | UNKNOWN |
| Actual chassis/airflow | not established | UNKNOWN |

The Sapphire vendor minimum PSU recommendation is 450 W; it is not evidence
of the installed PSU. No unverified 550 W, cooler, or chassis description is
published as fact.

## Software and exact benchmark environment

| Component | Captured value | Status |
| --- | --- | --- |
| OS | CachyOS rolling, Arch-like, x86_64 | OBSERVED SOFTWARE |
| Kernel | `7.2.0-rc7-1-cachyos-rc #1 SMP PREEMPT_DYNAMIC Tue, 11 Aug 2026 08:04:41 +0000` | OBSERVED SOFTWARE |
| Mesa / RADV | Mesa 26.2.1 / RADV, driver info `Mesa 26.2.1-arch3.1` | OBSERVED SOFTWARE |
| System LLVM | package version 22.1.8 | OBSERVED SOFTWARE |
| Rust | `rustc 1.97.1`, host `x86_64-unknown-linux-gnu`, Rust LLVM 22.1.6 | OBSERVED SOFTWARE |
| Cargo | 1.97.1 | OBSERVED SOFTWARE |
| Vulkan loader | 1.4.357.0 | OBSERVED SOFTWARE |
| glslc | 2026.3, target SPIR-V 1.0 | OBSERVED SOFTWARE |
| glslangValidator | 11:16.4.0, GLSL 4.60 | OBSERVED SOFTWARE |
| SPIRV-Tools | 2026.3 / Vulkan SDK 1.4.357.0 | OBSERVED SOFTWARE |
| HAR workspace version | 0.1.0 | OBSERVED SOURCE |
| R4KV public crate | 0.1.0 at the captured runtime commit; experimental | OBSERVED SOURCE |
| R4X / R4F execution version | no separate canonical production binary in the bounded smoke; historical versions not asserted | UNKNOWN / NOT USED |
| HAR build policy | `--release --locked --offline`; thin LTO, one codegen unit, panic abort, stripped symbols | CONFIGURED SETTING |
| Current selected override environment | no `AMD`, `RADV`, `MESA`, `VK_`, `RUST`, `CARGO`, `LD_`, or `HAR_` override variables observed in the capture shell | CONFIGURED SETTING |
| Current performance boot policy | `pci=realloc=on amdgpu.rebar=1`; no current `iommu=pt` or PCIe-ASPM-performance flag observed | CONFIGURED SETTING |

The bounded public smoke was the Rust binary
`har-rust-vulkan-smoke`, built from the captured HAR commit and run against
the checked-in `har/shaders` SPIR-V. It returned token `777` from the sampler
fixture and value `256` from the Q4_K fixture. This validates a native Rust
Vulkan path and its correctness fixtures; it is not a model throughput result.

Historical foreign-runtime receipts are not treated as current HAR benchmark
receipts. A historical tuned preset used GPU compute/high, CPU
`performance`, PCIe ASPM performance, and `iommu=pt`; the live profile above
does not use those settings. Every future result must carry its own phenotype
ID and exact environment.

## What this profile does not claim

- It does not claim that a different AMD, RDNA, NVIDIA, or Intel device will
  reproduce the result.
- It does not claim that another RX 9060 XT, even with the same marketing
  name, will reproduce the result without retuning.
- It does not claim a current full-model sustained clock, power, thermal, or
  throughput envelope.
- It does not claim that the factory clock, configured offset, and observed
  clock are interchangeable.
- It does not claim that storage, RAM, VRAM, CPU scheduling, or residency
  behavior is portable across nominally similar systems.

## Public evidence links

- [Benchmark receipt requirements](docs/benchmarks.md)
- [HAR release audit and native runtime trace](PUBLIC_HAR_RELEASE_AUDIT.md)
- [HAR hardware boundary](docs/architecture/hardware-support.md)
- [Hardware specialization methodology](docs/methodology.md)
- [Claims ledger](CLAIMS.md)

No raw private hardware receipt, host identifier, serial, UUID, model path,
or private repository reference is published by this profile.
