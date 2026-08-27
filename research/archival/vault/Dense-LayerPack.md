---
title: Dense LayerPack
type: architecture
status: design-only
---

# Dense LayerPack

LayerPack is the proposed exact transport format for grouped dense Q8 tensor
ranges. It is not an MoE ExpertPack. Each record preserves the GGUF source
name, offset, length, type, shape, destination offset, alignment, and checksum.
Source ranges must round-trip byte-for-byte; no hidden dequantization or
requantization is allowed.

The controlled modes are P0 current GGUF access, P1 source grouping, P2
GPU-native destination layout, P3 scatter/gather, P4 CPU gathering, P5 offline
prepack, and P6 measured multi-layer groups. Double/triple buffering and fence
ownership are explicit. Larger packs are not presumed faster.

Design authority: `[local path omitted]`.
