# EXTREME QUANT STATE V1

Lane: R4X extreme mixed precision (RDNA4 / gfx1200 / RX 9060 XT).
Date: 2026-08-24 (in progress; updated as receipts land).

## Mission
ONE mixed-precision R4X model. Precision profiles B1/T2/I3/D32A/I6/I8 assigned
per tensor by measured marginal quality-per-byte. D32A stays frozen as anchor.
QAT rescue map falls out of the PTQ frontier.

## Ground truth
- Model: Qwen3.8-27B (`Qwen3_5ForConditionalGeneration`), sha256 a0ee1ee6...
  - Text weights bf16 = 50.9 GiB. FFN 31.9 GiB (61.6%), GDN linears 10.4 GiB
    (20%), embed+lm_head (untied) 4.74 GiB (9.2%), full-attn 3.13 GiB, MTP 0.6 GiB.
  - Hybrid: 48 GDN layers + 16 full-attn (every 4th), 64 layers total, MTP seam
    = layer blk.64 (eh_proj/enorm/hnorm/shared_head_norm + one transformer layer).
- Release anchor: `[model payload omitted]`
  sha256 ead567f63dc1b1f774fccc6000385b62d5e57d1ac26cb26831cf8592cca049b4,
  uniform R4X_D32A on 506 tensors @ exactly 4.5 bpw + 360 F32 smalls =
  **4.503 effective bpw, 14.321 GiB payload** (15,358,213,024 B file).
- Dynamic-V3 control: UD-Q4_K_S (75bc9c8a...) same size class; its map is
  imatrix-driven folklore (embedding Q3_K, lm_head Q6_K, GDN/attn Q5/Q8 islands,
  FFN mostly IQ4_XS) with ZERO measured quality justification on record.

## R4X extreme profile family (implemented this lane, branch `extreme/mixed-precision`)
All profiles keep the D32A philosophy: 256-weight superblock, FP16 scale per
sub-block, little-endian bit packing, nibble-style decode d*(q-offset).

| profile | id | layout | bytes/256w | eff bpw | weight relRMSE (Gaussian ref) |
|---|---|---|---|---|---|
| R4X-B1 | 43 | sign bit/w + 8×fp16 s (g32) | 48  | 1.50 | ~0.60 |
| R4X-T2 | 44 | ternary 2-bit codes {0,+1,−1} + scales | 80  | 2.50 | ~0.50 |
| R4X-I3 | 45 | int3 offset-4 clipped ±3 + scales | 112 | 3.50 | ~0.23 |
| R4X-D32A | 36 | int4 offset-8 (anchor) | 144 | 4.50 | ~0.097 |
| R4X-I6 | 46 | int6 twos-complement + 4×fp16 (g64) | 200 | 6.25 | ~0.022 |
| R4X-I8 | 47 | int8 + 4×fp16 (g64) | 264 | 8.25 | ~0.006 |

Implemented end-to-end in the lane worktree `[local path omitted]`
(based on rc-snapshot tag `rc-snapshot-20260824` = d8ff61325):
ggml types/traits/dequant/CPU vec_dot/get_rows, Vulkan structs + dequant funcs +
generated mul_mat_vec pipelines + get_rows, gguf-py constants, loader ftype.
Round-trip validated encoder↔C-decoder semantics for all six profiles.
Build: `runtime/build-extreme-vk` (llama-cli, llama-perplexity built).

## Converter
`codec/r4x_extreme_convert.py`: binary surgery on the release template (header +
KV verbatim), requantizes D32A tensors from bf16 HF sources reapplying Qwen3.5
value transforms (GDN head permutations, norm +1). Precision map = JSON rules.
Multiprocess (~55 tensors/min).

## Weight-space census (running)
`sensitivity/run_census.py` → artifacts/EXTREME_TENSOR_SENSITIVITY_V1.json.
All text tensors × all profiles: rel_rmse, cosine, max_err, exact bytes/bpw.

## Allocator
`alloc/solver.py`: marginal-utility descent (greedy knapsack). Weight-space
frontier on partial census shows near-LINEAR cost growth 4.5→2.0 bpw — no cliff
visible to weight MSE alone, confirming the real cliff must be located by
end-to-end behavioral evals (KL/PPL/continuation), which are queued.

## GPU protocol
Sibling session's server gracefully stopped by owner approval; flock protocol on
temporary GPU lock file honored; identical relaunch script saved at
runs/sibling_server_restart.sh. RADV staging-pool quirk documented:
DSV4_STAGING_MB knob exists; uploads thrash when layers spill to CPU — run full
offload (-ngl 99).

## Eval ladder status
- L0 weight proxies: RUNNING (census)
- L2 logits/PPL/KL (wikitext-2, 512ctx): anchor + UD-Q4_K_S control RUNNING;
  candidates via build-extreme-vk binaries
- L1/L3/L4: k1-capture + deterministic continuation harnesses identified, queued
- imatrix capture for activation-aware census: queued behind baselines

## Open questions this lane must answer empirically
1. Does ANY large family tolerate T2/B1 functionally? (weight MSE says no cliff;
   behavior may differ both ways)
2. Where is the REAL cliff? (long-context/GDN state suspected first failure point)
3. Do activation-aware scale selection + islands beat uniform downgrades enough
   to justify dispatch complexity?
