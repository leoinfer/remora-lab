# Research methodology

Every result gets an evidence class:

- `verified` — independently rerunnable with public inputs and a checked
  invariant;
- `experimental` — implementation or bounded evidence exists, but coverage
  or reproducibility is incomplete;
- `historical` — useful context retained from an earlier private experiment;
- `invalidated` — the proposed claim failed an explicit acceptance gate.

The claims ledger is the authority for language used in public documents.
Performance is measured only after correctness, model identity, warm-up,
prompt, token count, and resource accounting are fixed. Narrow-kernel wins,
synthetic models, and reduced byte counts are not silently promoted to
end-to-end generation claims.

## Hardware specialization

Much of this research was designed, tuned, and measured around one specific
workstation, particularly its AMD Radeon RX 9060 XT configuration. Hardware
specialization is intentional. Results should not be assumed to reproduce on
different GPUs, vendors, driver stacks, memory systems, or even another
nominally identical graphics card without retuning.

The reference phenotype is
`RX9060XT16-NITRO-GFX1200-RADV-2026.08.27-v1`. The exact public-safe system
description is [HARDWARE_PROFILE.md](../HARDWARE_PROFILE.md), and its
machine-readable form is [hardware_profile.json](../hardware_profile.json).
Where a hardware-specific execution strategy materially outperforms a
generic one on the target machine, this project generally prefers the
specialized strategy. Portability and generalization are separate research
problems and should not be inferred from a result obtained on the reference
machine.

Examples of intentional specialization include gfx1200-specific kernels,
RDNA4 subgroup and data layouts, sparse-instruction experiments, workgroup
widths, register-pressure choices, cache layouts, VRAM-driven quantization,
and residency policies spanning VRAM, system RAM, and NVMe. Scheduler choices
also depend on the observed CPU policy, PCIe/BAR state, driver behavior, and
the resident/warm/streamed/cold memory phenotype.

A reference-machine result is valid only within the exact environment recorded
by its receipt. A cross-hardware result is a separate experiment requiring a
new device identity, driver and shader capability probe, memory/PCIe state,
retuning, and matched correctness and workload evidence.

## Moonshots and anomalous results

This project deliberately investigates results that appear implausible,
including results that conflict with vendor specifications, conventional
performance models, or my own previous conclusions.

An implausible result is not treated as proof of a breakthrough, but it is
also not discarded merely because an existing number says it should be
impossible. Instead, the discrepancy becomes the experiment.

For example, an early gfx1200 sparse-matrix experiment appeared to indicate
multi-POPS INT4 throughput despite an expected hardware envelope around the
advertised ~821 sparse INT4 TOPS. Rather than publishing the apparent result
as a discovery or rejecting it from intuition alone, I continued constructing
stronger known-answer tests. Those tests eventually showed that the apparent
useful throughput was invalid: repeated accumulator behavior had caused the
benchmark to overcount useful committed work.

The failed result remains documented because the investigation exposed real
instruction-level behavior and improved the benchmark methodology. See the
dedicated [`gfx1200 sparse-matrix anomaly`](../research/falsified/GFX1200_SPARSE_MATRIX_ANOMALY.md)
record and the [`claims ledger`](../CLAIMS.md).

The general rule is:

**Do not reject a moonshot because it sounds impossible. Do not accept it
because it sounds exciting. Try to kill it.**
