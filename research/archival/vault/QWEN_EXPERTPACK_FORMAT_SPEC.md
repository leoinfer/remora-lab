# H11 ExpertPack exact format specification

**Status:** XP2 design; implementation not yet connected to Qwen runtime.

## Contract

An ExpertPack record contains the canonical quantized bytes of exactly one
Qwen routed expert. It changes physical organization only. `pack` reads the
GGUF source spans; `unpack` returns the original gate, up and down byte strings.
No dequantization, requantization, lossless compression, bit-width change,
scale rewrite, or mathematical transformation is permitted.

The normative identity is:

```text
unpack(pack(model, layer, expert).gate) == source_gate_bytes
unpack(pack(model, layer, expert).up)   == source_up_bytes
unpack(pack(model, layer, expert).down) == source_down_bytes
```

All three checksums and the whole payload checksum must match. A record with a
wrong model hash, layer/expert, type, shape, offset, size, alignment, or checksum
is invalid.

## Binary record

All integers are little-endian. Header and payload fields are aligned to 256
bytes unless a field explicitly says otherwise.

```text
[256-byte fixed header]
[UTF-8 canonical JSON metadata, padded to header_bytes]
[gate payload at stored_offsets.gate]
[padding]
[up payload at stored_offsets.up]
[padding]
[down payload at stored_offsets.down]
[tail padding]
```

The fixed header contains `magic="QWENEXPK"`, `format_version=1`,
`header_bytes`, `record_bytes`, `metadata_bytes`, `layer`, `expert_id`,
`geometry_class`, `layout`, and the SHA-256 of the canonical metadata. The
canonical JSON metadata is validated by `qwen_expertpack.schema.json`.

Metadata required for each projection:

- GGUF tensor name and Q6_K/Q8_0 type;
- shape and expert axis;
- source file offset and source byte size;
- source expert offset and size;
- stored offset and stored size;
- alignment;
- per-projection SHA-256;
- source and destination slot layout;
- record payload SHA-256 and flags.

## Layout 1 — source-faithful expert pallet

Payload order is `[gate][up][down]`, with each payload copied as one canonical
span and each start rounded to 256 bytes. This layout is easy to inspect and
supports one record read, but a runtime gather still has to read the three
separate GGUF source regions unless the pack was prepared offline.

For Qwen Q2 all current expert payload sizes are multiples of 256, so the
alignment overhead for an offline record is header/padding only. The ordinary
mixed record payload is `2,834,432` B; layer 1's exceptional Q8_0 record is
`3,342,336` B.

## Layout 2 — GPU-native slot pallet

The payload order and offsets are selected to match a destination slot buffer
and tensor views. The canonical bytes remain unchanged. The metadata records
projection offsets, view shapes, row/column strides, and destination tensor
buffer identity. A valid Layout 2 candidate must prove that Vulkan views point
at the exact canonical Q6_K/Q8_0 spans without a hidden conversion.

The initial candidate uses aligned `[gate][up][down]` offsets with the actual
source expert strides. A later candidate may use the runtime's physical slot
order, but it must report padding and cannot assume Layout 2 is faster.

## Destination and lifetime

The record is immutable after checksum validation. A runtime destination may be
one contiguous per-expert slot record or three views into a layer packet. The
source record must remain alive through the Vulkan fence that consumes its
copy. Slot repurpose is forbidden until the existing Q2 fence contract passes.

## Exact variants under test

| ID | Description |
|---|---|
| P0 | Existing separate source spans, staging regions, and copy regions |
| P1 | Offline/source contiguous record, three destination regions |
| P2 | Offline/source contiguous record, one contiguous destination record/views |
| P3 | Existing canonical spans, one scatter/gather submission per layer |
| P4 | Runtime CPU-gathered layer packet, one large transfer |
| P5 | Offline prepacked layer packet, no runtime gather |

P1/P2 test physical packing; P3 tests submission batching; P4 tests DMA
packet size against an extra CPU copy; P5 is the exact packing upper bound.
They must not be collapsed into one ExpertPack switch.

## Required proof artifacts

`expertpack_roundtrip.py` creates records from the canonical source map and
checks gate/up/down and complete payload identity for an ordinary mixed expert
and exceptional all-Q8_0 expert. Full-model packing is not performed until the
one-expert gate passes. The result is a byte proof, not a performance claim.
