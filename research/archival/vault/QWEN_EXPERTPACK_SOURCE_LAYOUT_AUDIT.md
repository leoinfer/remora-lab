# H11 ExpertPack XP1 — Qwen routed source-layout audit

**Status:** PASS for static source-map validation; no packing or transport
claim yet.

## Identity

- Model: `[local path omitted]`
- File size: `31,843,777,504` bytes
- Model SHA-256: `f6b6c6d5cfa6f00d964eeb7add28eb14ce7481734d506b90681007678cd2c484`
- Source map: `[local path omitted]`
- Source-map SHA-256: `bd48b7b63ea006300d80b7da5b1ebb2f4da7ae288ba2d70c59958e6e402e5bfc`
- Generator: `[local path omitted]`

The generator read the canonical GGUF tensor metadata and independently
SHA-256 hashed every one of the 30,720 projection/expert spans (40 layers × 3
projections × 256 experts). It validated bounds and global overlap while
preserving each file offset, size, type, alignment and 4 KiB page range.

## Validation

| Check | Result |
|---|---:|
| Routed layers represented | 40 / 40 |
| Experts represented | 10,240 / 10,240 triplets |
| Projection tensors | 120 / 120 |
| Expert spans | 30,720 |
| Exceptional layer 1 represented | yes, all Q8_0 |
| Ordinary mixed layers represented | yes, 39 layers |
| Span bounds | PASS |
| Global overlap | none |
| Independent span hashes | PASS |
| Projection expert contiguity | PASS for all projections |

## Geometry distribution

- Ordinary layers 0, 2–39: gate Q6_K `860,160` B/expert, up Q6_K
  `860,160` B/expert, down Q8_0 `1,114,112` B/expert; triplet
  `2,834,432` B.
- Exceptional layer 1: gate/up/down Q8_0 `1,114,112` B/expert; triplet
  `3,342,336` B.
- Projection types across the model: 78 Q6_K and 42 Q8_0 tensors.
- Total canonical routed source bytes: `29,154,607,104` B (`27.15234375`
  GiB). This is source payload, not runtime transfer per token.

## Physical layout

For every layer, projections are individually expert-contiguous, but a gate/up/
down triplet is not sequential in the GGUF file. The file order is:

```text
down experts → gate experts → up experts
```

Representative ordinary layer 0 offsets/ranges:

| Projection | file offset | file bytes | expert bytes |
|---|---:|---:|---:|
| down | 1,118,433,760 | 285,212,672 | 1,114,112 |
| gate | 1,404,760,544 | 220,200,960 | 860,160 |
| up | 1,628,180,960 | 220,200,960 | 860,160 |

Representative exceptional layer 1 offsets/ranges:

| Projection | file offset | file bytes | expert bytes |
|---|---:|---:|---:|
| down | 1,885,820,128 | 285,212,672 | 1,114,112 |
| gate | 2,172,146,912 | 285,212,672 | 1,114,112 |
| up | 2,460,579,040 | 285,212,672 | 1,114,112 |

The ordinary and exceptional projection offsets have the same qualitative
order and gaps. In ordinary layers, gate→up is `223,420,416` B and gate→down
is `-286,326,784` B; in layer 1 gate→up is `288,432,128` B and gate→down is
`-286,326,784` B. There is no source-faithful contiguous triplet record.

Tensor starts and expert boundaries are not generally 4 KiB aligned. Adjacent
expert spans in every projection share one boundary page in the map's 4 KiB
page calculation (`adjacent_expert_shared_4k_pages=255` across 255 adjacent
pairs). This indicates some page-level locality for sequential same-projection
access, but selected top-8 IDs can be non-sequential and the three projections
remain far apart.

## Runtime interpretation

The Q2 manager does not perform three file syscalls per request: its canonical
GGUF tensors are host/mmap-accessible and `source_span()` performs pointer
arithmetic plus bounds checks. The physical cost can still be page faults and
memory reads when pages are cold. XP0 measured three source-span operations per
logical routed expert and three independent staging copies for every changed
expert.

This map falsifies the narrow assumption that the current source is already an
expert-triplet pallet. It does **not** yet prove fragmented disk layout is the
end-to-end bottleneck: the model is often page cached, and XP0 exposed CPU
preparation, region handling, graph submissions, and synchronization as
separate possible costs.

## ExpertPack design consequence

A source-faithful exact pack can gather `[gate][up][down]` per expert, but it
will read three distant projection spans. A prepacked file could make one
expert sequentially readable; a runtime CPU gather may simply move the cost
from three staging copies to an additional gather copy. Both must be measured
as separate P1/P4/P5 variants. Layer 1 must never inherit ordinary Q6_K sizes.

## Reproduction

```bash
[local path omitted]/huggingface/bin/python \
  [local path omitted] \
  [local path omitted] \
  --output [local path omitted]
```

The full 33.82 MiB JSON source map remains outside the vault and is indexed in
[[Evidence/Artifact-Register]].
