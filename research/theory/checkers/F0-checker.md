---
id: F0
status: OPEN_PROBLEM
source: ../../archival/authoritative/REMORA_FORMALIZATION_QUEUE.md
originality_status: unknown
---

## F0 — Source and identifier audit

**Status: `MACHINE-CHECKED` / `BLOCKED` for missing ranking**

- **Input:** source manifest of all audited paths, SHA-256, line counts, status timestamps.
- **Check:** required names `H01–H39`, `N01–N26`, manifest ideas `1–30`, `TBEH` appear exactly once in the crosswalk; required source ranking path exists or emits `BLOCKED`.
- **Adversarial cases:** duplicate MARC acronym, absent AutoSurgeon artifact, stale active-state field.
- **Output:** source-audit JSON and identifier crosswalk.
- **Certificate:** `REMORA-SRC-001`.
- **Affected:** all families; N26.
