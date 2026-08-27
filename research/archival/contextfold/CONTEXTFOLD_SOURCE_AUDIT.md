# REMORA ContextFold source audit

**Scope.** Read-only audit of `[local path omitted]` at
`da004c4a3388164531923c3cd54bf5df79b2ba04`. No source file was modified. The
main post-B0 lane was left running and its worktree remained clean at audit
time. This lane did not use a GPU, launch inference, or load a model.

## Findings

The active target already has substantially more context machinery than a
blank runtime:

- ordinary per-layer K/V cache tensors and a ring/cell allocator;
- K/V type selection, including quantized cache types with Flash Attention
  constraints;
- CPU KV placement with `--no-kv-offload` and backend placement with
  `offload_kqv`;
- unified sequence streams, sequence IDs, cell sharing/copy, deletion/keep,
  context shifting, and state serialization;
- CPU/Vulkan Flash Attention implementations with tiled/partial online
  softmax-like accumulation and multi-row batching;
- separate recurrent R/S memory, GDN/SSM operators, bounded recurrent rollback
  snapshots, and a hybrid memory object that updates attention and recurrent
  state together;
- prompt-cache plumbing and target-specific DSA/DeepSeek-V4/ISWA cache paths in
  this source snapshot.

The audit did **not** find a generic paged-KV block table, a cross-tier HOT/WARM/
COLD context controller, an NVMe cold-KV protocol, or a ContextRoot/state-
generation validity namespace. Generic shader identifiers such as
`n_kv_blocks` are kernel tiling, not a persistent paged-attention block table.

## Exactness boundary

Existing state save/load is the closest prior representation. It serializes the
runtime's K/V and recurrent state, but the API does not establish the complete
ContextFold causal key (model hash, runtime generation, exact prefix root,
execution epoch, codec generation, and recurrent generation). Sequence copying
can share cells inside a context, but is not a cross-session authenticated
immutable prefix atlas. Existing Flash Attention can reuse resident K/V for
multiple query rows; a new block-stationary proposal is novel only if it avoids
additional cold-tier movement beyond that batching.

Qwen3.6-27B is hybrid: 48 recurrent layers and 16 ordinary full-attention
layers. A cold KV artifact cannot stand in for recurrent state. MTP/speculative
candidate queries also require a compatible recurrent snapshot and causal
prefix before they may be grouped.

The detailed feature-by-feature matrix is
`CONTEXTFOLD_EXISTING_FEATURE_MATRIX.csv`. The matrix uses source symbols and
line ranges as anchors; it is not a claim that every target-specific backend
combination has been live-tested.

## Audit method

1. Record the authority worktree and commit.
2. Search source and headers for allocation, type, placement, Flash Attention,
   recurrent, sequence, state, prompt, paging, and speculative symbols.
3. Read the controlling implementation files, not only command help.
4. Separate existing exact state/placement from proposed tiering.
5. Preserve negative findings (no generic paged table/NVMe KV path).

## Safety status

The only observed inference process was the already-running main Qwen server;
this lane did not attach to, signal, or reconfigure it. All subsequent artifacts
are scalar/static or bounded synthetic CPU tests.
