---
id: C-01
status: CONJECTURE
source: ../archival/authoritative/REMORA_NEW_CONJECTURES.md
originality_status: unknown
---

## C-01 — Causal-Closure Cache (CCC)

**Status: `CONJECTURED`**

### Claim

Exact reuse is best implemented as a content-addressed cache over the **transitive causal closure** of an artifact, not over prompt text, token ID, model hash, or semantic fingerprint alone.

```text
key = MerkleRoot(
  model/source roots,
  token/input roots,
  recurrent/KV state root,
  route/slot/pack root,
  precision/backend/driver contract,
  graph/function version,
  sampler/RNG state,
  external-state version
)
```

### Why stronger

It joins REMORA Reclaim/Refrigerator, dependency-versioned cognition, H10 stable slots, ExpertPack/LayerPack, and MARC-Symbiote hardware state into one exactness primitive. Approximate cache hits may use a relaxed key only as drafts.

### Counterexample search

Delete each key field in turn and search for a pair of artifacts that collide but differ in tensor/state/output. The dense Qwen prompt-cache controls and HERMES stale-arena history are seed counterexamples.

### Cheapest decisive test

CPU/property-based Merkle-key checker with synthetic recurrent state and mutable pack/driver versions. No model inference required.

### Certificate

`CCC-v1` records closure members, root, artifact hash, validation rule, and exact/approximate mode. The checker requires a miss on every changed causal leaf.

### Affected ideas

Manifest `12–15`, `26`; H10, H27, H30, H33; N04/N05/N09/N25; ExpertPack and LayerPack.

---
