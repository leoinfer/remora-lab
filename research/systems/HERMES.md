# HERMES-V4

**Status:** `PROPOSED` integrated architecture

HERMES-V4 is the consolidated architecture atlas for consumer-hardware local
inference. It combines correctness modes, compact expert skeletons, staged
MoE movement, residual quantization, route scouting, multi-token prediction,
expert-major execution, resident expert atlases, asynchronous lanes,
telemetry, resource-aware scheduling, and viability-style control.

The complete H01–H39 record is [`HERMES_V4_COMPLETE_IDEA_ATLAS.md`](../archival/authoritative/HERMES_V4_COMPLETE_IDEA_ATLAS.md).
The individual cards are under [`ideas/atlas`](../ideas/atlas/). The atlas is
a research architecture and vocabulary, not a claim that every mechanism is
implemented or jointly validated.

## How to read it

H01–H19 focus on runtime, routing, residency, and execution. H20–H34 extend
the control model with resource allocation, viability, autotuning,
certificates, and distilled reasoning control. H35–H39 are program-level
tracks and the integrated architecture. Each card retains its source evidence
and next falsifier; the normalized index status is only a navigation label.

HAR implements selected Rust surfaces that can host parts of this program.
The source atlas remains research-only and is not linked into the production
binary.
