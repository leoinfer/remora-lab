# REMORA ContextFold executive report

## Scope and safety

This is a static/formal/source-audit/CPU-simulation lane. Authority source:
`[local path omitted]` at commit
`da004c4a3388164531923c3cd54bf5df79b2ba04`. No GPU was used, no model or
inference was launched, the active Qwen worktree was not modified, and no live
speedup is claimed. A pre-existing main-lane Qwen server was observed and left
untouched.

## Findings

1. **Existing context features:** resident KV cells, K/V quantization, CPU KV
   offload, backend KV placement, unified KV, Flash Attention with tiled
   partials, sequence sharing/copy/removal/shift, prompt cache/state
   serialization, SWA/ISWA, DSA/DeepSeek-specific caches, recurrent rollback,
   MTP outputs, and separate hybrid recurrent memory are already present.
   Generic paged block tables and NVMe cold-KV tiering were not found.
2. **State types:** Qwen local metadata gives 48 recurrent layers and 16
   full-attention layers, plus a separate MTP layer. Exact context state must
   include tokens, KV, recurrent/GDN/SSM state, positions, sampler/RNG where
   required, model/runtime identity, and execution epoch.
3. **Memory formula:** `tokens × attention_layers × 2 × kv_heads × head_dim ×
   bytes/element`. Qwen F16 ordinary KV is 65,536 B/token; at 262K it is 16
   GiB binary. Q8_0 and Q4_0 byte-model rows are 8.5 GiB and 4.5 GiB.
   An 8K F16 hot window is 512 MiB; recurrent state is additional.
4. **Real fixtures:** no real KV, recurrent-state, activation, or attention
   trace fixture was available. Ratios and concentrations are not empirical.
5. **Lossless codec:** bounded synthetic round trips all passed. Best measured
   synthetic ratio was 184.09x (repeated pattern, bit-plane/zlib); real-KV ratio
   is UNMEASURED. The best synthetic pure-Python decoder was about 0.90 MB/s in the recorded run;
   encoded transfer plus decode loses against raw transfer under the declared
   3 GB/s example. This does not predict native codec performance.
6. **Context-RSSO:** against separately submitted query rows, 480/560 scalar
   sweep rows satisfy `(J-1)F>O`; at 1,024-token blocks and prior bandwidths
   J=2 already satisfies the algebraic inequality. Against an already-batched
   resident-KV baseline, 0/560 rows show additional value. No MTP acceptance
   or recurrent legality trace was obtained.
7. **Replay:** token/checkpoint storage reduces memory in the model, but 0
   tested rows met both a storage reduction and an assumed 10-second cold
   reconstruction deadline at 100 replay tokens/s.
8. **Certificates:** exact-zero underflow and local half-ulp tests pass, but
   certificate composition through the complete model/recurrent authority is
   false. No block may be skipped from a heuristic mass threshold.
9. **Prefix/UARC/reclaim:** union residual savings and causal shared-prefix
   savings are positive in synthetic sensitivity regions; they require real
   overlap/root rates. Reclaim selects hot VRAM or ContextPack depending
   deadline/use probability; this is a policy model only.
10. **Modeled capacity:** with an 8K hot window, the scalar oracle reports a
    maximum modeled VRAM reduction of about 96.6% relative to full Qwen F16 KV
    (capacity-only). A 96.2% total-storage reduction appears only in the
    low-latency-unusable replay/token-archive row; with compression ratio 1.0,
    hot+cold exact storage has no total-memory reduction.
11. **Strongest counterexample:** at 262K, the 16 GiB F16 attention KV scan
    needs roughly 2.6 seconds at the prior 6.4 GB/s PCIe/GTT lower bound before
    compute/synchronization; direct NVMe per token is rejected.

## Gates

- Passed: bounded ContextPack/codec byte reconstruction; mathematical online
  softmax proof; scalar RSSO break-even against separate queries; modeled hot
  VRAM capacity value.
- Failed or not established: real compression, compressed transfer economics,
  useful replay deadline, Context-RSSO beyond existing batching, complete
  certified elimination, and strong baseline superiority.
- Narrowed: exact streaming is a capacity architecture with cold RAM/replay
  materialization, not an immediate efficient interactive decode path.

Detailed machine-readable decisions are in `CONTEXTFOLD_GATE_VERDICTS.json`.
