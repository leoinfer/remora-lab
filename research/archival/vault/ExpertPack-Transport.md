---
title: ExpertPack Transport
type: architecture
status: design-gated
---

# ExpertPack Transport

## H11 definition

An ExpertPack is a reversible physical pallet for one canonical expert:
`[gate bytes][alignment][up bytes][alignment][down bytes]`, with metadata and
checksums. It changes physical organization and transport batching only; it
must not dequantize, requantize, compress, or change tensor math.

## Hypothesis

The current Qwen path performs independent gate/up/down source lookup, slot
selection, staging, copy-region creation, and deferred Vulkan work. A contiguous
record may reduce handling, region count, descriptor/view churn, submissions,
and synchronization. This remains a hypothesis until counters and end-to-end
energy/timing prove it.

## Exact variants

| Variant | Isolation target |
|---|---|
| P0 | current separate projection transport |
| P1 | contiguous source record, three destination regions |
| P2 | one contiguous destination record with tensor views |
| P3 | scatter/gather, one submission per layer |
| P4 | CPU-gathered layer staging packet |
| P5 | offline prepacked layer packet |

No variant may hide CPU copying or change slot policy. Every variant requires
source/destination hashes, alignment validation, zero Q2 failure counters, and
matched route/output correctness.

## Current boundary

XP0 measured the current path: 3 source-span operations per logical expert,
three CPU copies/regions per uploaded expert, and 697 transport-attributed
record batches/submissions in the 8-token all-layer audit. XP1 validated all
30,720 source spans and falsified the assumption that GGUF stores contiguous
expert triplets. XP2 payload/source-byte examples are supported for ordinary
mixed and exceptional all-Q8_0 experts, but V5 found stale embedded
`metadata_bytes` in all four retained records; the full record-format PASS is
withheld pending offline regeneration. No speed, energy or physical-load claim
exists.

Source reports: `[local path omitted]`,
`QWEN_EXPERTPACK_SOURCE_LAYOUT_AUDIT.md`, and
`QWEN_EXPERTPACK_FORMAT_CERTIFICATE.md`; V5 evidence is in the immutable
static-swarm verifier drop. The next gate is one-expert P0–P5
microbenchmarking with unchanged slot policy, but it remains paused behind
post-B0 dense phase/byte/resource closure.
