---
title: Dense RAM/VRAM co-design
type: architecture
status: planning
---

# Dense RAM/VRAM co-design

The exact Qwen3.6-27B Q8_K_XL tensor payload is 35,765,489,664 B while the
current GPU reports 17,095,983,104 B total VRAM. Placement is therefore a
role/reuse problem, not a maximum-NGL-only problem.

Planner roles are permanent small state/norm/control tensors, candidate
resident complete skeleton blocks, and streamed large FFN/attention groups.
Activations should remain on GPU unless measured host ping-pong wins. Plans
must reserve allocator overhead, recurrent/KV state, activations, descriptors,
two staging slots, and safety margin. Direct host/GTT and explicit copied VRAM
working sets are separate controls.

No free-VRAM boundary, stock residency, PCIe traffic, or overlap measurement is
currently established. See `[local path omitted]`.
