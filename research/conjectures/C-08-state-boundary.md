---
id: C-08
status: CONJECTURE
source: ../archival/authoritative/REMORA_NEW_CONJECTURES.md
originality_status: unknown
---

## C-08 — Exactness Requires a State Boundary, Not Just an Authority Endpoint

**Status: `DERIVED UNDER ASSUMPTIONS`**

### Claim

Every exact/verified module interface must expose both an authority output and an authority state boundary. An interface that returns only logits/tokens cannot support exact continuation for a stateful host.

### Why stronger

It is a direct synthesis of RSSO recurrent state, MARC-Symbiote refresh, Qwen B0 drift, and HERMES fence/state contracts.

### Counterexample search

Use two states with equal current argmax token but different next-step logits. Any token-only interface falsely claims continuation equivalence.

### Cheapest decisive test

Two-state finite recurrence exhaustive checker.

### Certificate

`output_hash`, `state_hash`, `state_schema_version`, `commit_prefix`, `discard_suffix`, and authority root are mandatory.

### Affected ideas

H02/H08/H09/H13/H25/H29/H33; manifest `6–10`, `22`, `25–27`; N09/N18/N24/N25.

---
