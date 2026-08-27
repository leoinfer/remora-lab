# REMORA ContextFold formal specification

Status: **provisional static specification**. No runtime implementation is
authorized in the active Qwen worktree.

## 1. Authority state

For committed token position `t`, the authority state is

```
S_t = (H_t, KV_t, R_t, P_t, G_t, M_t, E_t)
S_(t+1) = F_theta(S_t, y_(t+1))
```

`H` is the exact committed token history; `KV` is attention-layer state; `R`
is recurrent/GDN/SSM state; `P` contains position/sequence metadata; `G` is
sampler/RNG state when the runtime contract requires it; `M` identifies model,
quantization, kernel and runtime; `E` is execution epoch/namespace.

A representation is exact only relative to a declared authority contract.  A
reconstructed K/V tensor with a different recurrent state is not the same
`S_t` for a hybrid model.

## 2. Tiers and representations

Logical physical tiers:

- **HOT:** exact executable state resident in VRAM;
- **WARM:** exact state or lossless ContextPack in RAM;
- **COLD:** exact token/checkpoint/ContextPack artifacts on NVMe, never required
  synchronously in the ordinary per-token path unless a live design proves it.

Representation lattice:

```
CF-L0 tokens/provenance
  -> CF-L1 replay checkpoint + suffix
  -> CF-L2 lossless ContextPack / CF-L3 base + exact residual
  -> CF-L4 exact RAM materialization
  -> CF-L5 exact VRAM materialization
```

The controller may promote or reclaim a block only after verifying the complete
causal dependency closure.  The ContextPack record is specified in
`CONTEXTPACK_SCHEMA.json` and `CONTEXTPACK_FORMAT.md`.

## 3. Validity

For artifact `a` requested by state `S_t`:

```
Valid(a,S_t) =
  ModelMatch(a,S_t)
  AND PrefixMatch(a,S_t)
  AND PositionMatch(a,S_t)
  AND StateGenerationMatch(a,S_t)
  AND RuntimeGenerationMatch(a,S_t)
  AND ExecutionEpochMatch(a,S_t)
  AND NotExpired(a)
```

Token equality is insufficient.  A ContextRoot is computed over model/runtime
roots, prefix/parent roots, token positions, layer/state kind, precision/codec,
recurrent generation and epoch.  Shared immutable storage is legal only when
roots match.

## 4. Exact execution modes

**Exact mode** processes every attention block eventually, regardless of tier or
order.  It uses online `(m,l,o)` partials and merges all blocks.  Tiering can
change location and order but not the state authority.  Numerical differences
must be classified as one of the declared exactness classes:

- `BIT-EXACT`;
- `NUMERICALLY EXACT UNDER DECLARED CONTRACT`;
- `LOSSLESS STATE, DIFFERENT ARITHMETIC ORDER`;
- `CERTIFIED`;
- `APPROXIMATE`.

**Certified-skip mode** may omit a block only if a verified finite-precision
certificate covers the declared attention/output/state boundary.  A heuristic
attention mass threshold or next-token match is not such a certificate.

**Approximate mode** includes retrieval, summaries, lossy eviction and sparse
attention.  It is reported separately and cannot be used to support an exact
claim.

## 5. Heterogeneous-layer rule

For Qwen3.6-27B local metadata, the main trunk contains 48 recurrent layers and
16 full-attention layers.  The attention KV formula applies only to the latter.
`R_t` is independently versioned and materialized.  Candidate/speculative
queries are legal together only when they share a compatible prefix and have
valid recurrent snapshots; the source audit does not provide that certificate
for a new cold streaming path.

## 6. Promotion/reclaim policy

For block `b`, a policy may use

```
V(b) = p_access C_miss - C_promote - C_contention
       - C_eviction - C_memory_pressure
promote iff V(b) > 0
slack(b) = horizon_tokens / token_rate - ready_time
```

Signals include recency, mass history, query similarity, layer behavior,
prompt/retrieval markers, MTP future queries, and repeated-prefix identity.
Recency is a baseline, not an optimality assumption.  The reclaim simulator
implements a shadow-price sensitivity model and exposes deadline slack.
