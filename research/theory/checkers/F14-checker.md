---
id: F14
status: OPEN_PROBLEM
source: ../../archival/authoritative/REMORA_FORMALIZATION_QUEUE.md
originality_status: unknown
---

## F14 — Source-level invariant checker

**Status: `EXPERIMENTALLY TESTABLE`; no production edits required**

- **Input:** read-only source paths and architecture contracts.
- **Check:** presence of original-ID→slot mapping, fence/epoch guard, source range bounds, per-position ID tensor, fail-closed counters, no exact claim from approximate flag.
- **Adversarial cases:** rank-as-slot offset, one route vector broadcast to K rows, reset before fence, zero-fill after allocation failure.
- **Output:** source invariant report with file/symbol/line evidence.
- **Certificate:** `REMORA-SRCINV-001`.
- **Affected:** H02/H10/H11/H13/H33, N01, RSSO LayerPack, ExpertPack.
