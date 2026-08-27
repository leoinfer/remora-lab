---
title: Exact block verification
type: architecture
status: open-gated
---

# Exact block verification

A candidate block is verified by the complete original Q8 target. The required
semantics are target-authoritative hidden/recurrent/KV state, candidate causal
positions, accepted-prefix commit, and rejected-suffix discard. Raw target
positions, verified positions, draft positions, and committed tokens remain
separate denominators.

Qwen3.6-27B’s hybrid recurrent state makes this an open gate: current static
source support is not a certificate for multi-position verification or rollback.
The post-B0 production-fix baseline is now deterministic under its named
contract, but its physical-byte roofline is incomplete and it supplies no
Link/Verify state certificate. The first live V0 should compare sequential
target steps, causal K-position verification, streamed LayerPack verification,
and double-buffered submission only after the complete resource and state gate
review permits it.
