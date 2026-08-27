# B0 first-use repeatability diagnosis

**Status:** `EXPERIMENTAL`

This record tracks a repeatability diagnosis for first-use behavior in a
Qwen-shaped Vulkan/MTP control. The historical note reports that clean
interleaved exact pairs were not reproduced reliably; that observation is a
diagnostic boundary, not a performance or quality claim.

The preserved source entry is in [`IDEA_REGISTRY.jsonl`](../../archival/vault/IDEA_REGISTRY.jsonl).
The next useful experiment is a fresh, model-identified control with explicit
warm-up, process, allocator, and trace state. No model payload or raw receipt
is part of this candidate.
