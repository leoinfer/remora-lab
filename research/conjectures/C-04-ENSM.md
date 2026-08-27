---
id: C-04
status: CONJECTURE
source: ../archival/authoritative/REMORA_NEW_CONJECTURES.md
originality_status: unknown
---

## C-04 — Epoch-Namespace State Machine (ENSM)

**Status: `CONJECTURED`, motivated by `COUNTEREXAMPLE FOUND`**

### Claim

Every mutable transport/state object should be namespaced by an epoch that includes both logical decode epoch and runtime allocation/graph generation. A payload, slot, staging slice, cache artifact, or recurrent snapshot from epoch `e` cannot satisfy a consumer requiring `(e, generation)` unless its certificate explicitly proves compatibility.

```text
namespace = (model_root, graph_generation, decode_epoch, sequence_id)
```

### Why stronger

It combines Q3 stale-arena repair, HERMES NT-store publication, Qwen Q2 fence/thrash, dense Qwen host-buffer/memory drift, and dependency-versioned cognition. A monotonic decode epoch alone may be insufficient when allocator/graph generation changes without a new logical token.

### Counterexample search

Replay a same-process cache/graph-reuse sequence with an unchanged token stream but changed allocation generation. Any reused buffer without a generation match is a deliberate stale-state injection.

### Cheapest decisive test

Static source-level invariant checker over artifact metadata and a finite scheduler model.

### Certificate

Every copy, state read, publication, and cache hit carries source and consumer namespaces plus fence sequence. Fail closed on unknown generation.

### Affected ideas

H02/H10/H13/H14/H29/H33; manifest `13`, `22`, `25`, `26`; N01/N05/N09/N26.

---
