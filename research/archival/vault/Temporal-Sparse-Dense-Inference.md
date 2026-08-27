---
title: Temporal-Sparse Dense Inference
type: architecture
status: hypothesis
---

# Temporal-sparse dense inference

The hypothesis is that a dense model may expose a cheap temporal draft path
from selected original layers while the full model verifies candidate blocks in
bursts. “Sparse” refers to draft execution across time/layers; it does not
change the authoritative target or imply a sparse trained model.

For the initial campaign, no target token may bypass full Q8 verification. Any
future high-confidence commit without immediate verification is a separate
**APPROXIMATE / NOT TARGET-EXACT / FUTURE RESEARCH** mode.

The dependency graph shows Qwen3.6-27B is hybrid: recurrent conv/SSM state and
full-attention KV state have different rollback requirements. Agreement, state
divergence, accepted-prefix length, and draft latency must be measured rather
than inferred from resident bytes.
