# Alice campaign

**Campaign window: 2026-09-19 → 2026-09-22 (≈3 days of active work).**

The broader local-inference program is much older; the Alice campaign itself is
not. The earliest dated Alice receipt in the research corpus is a loader
bring-up at `2026-09-19T11:05:00Z`, and the latest receipt that loads the Alice
artifact is `2026-09-22T19:05:22Z`. Nothing in this file should be read as a
months-long campaign, and no pre-2026-09-19 Alice work exists in the searched
corpus.

**Evidence class:** everything below is `MEASURED` or `HISTORICAL` on the
reference machine unless a row says otherwise. No number here is a portability
claim, and no row blends raw decode with MTP-accepted decode or prefill.

## What Alice is

Alice is `AliceAI-Foundation-80B-A3B-Base` — a hybrid linear-attention + MoE
model with a custom `alice_ai` architecture. It is **not** a Qwen derivative,
and it is **not** the same architecture family as Flash-Next or Qwen3.8-27B.

| Property | Value | Class |
| --- | --- | --- |
| Blocks | 48 = 36 KDA linear-attention + 12 full-attention, full attention every 4th block | `MEASURED` |
| Attention | KDA gated delta-rule, 32 K/V heads × 128, conv kernel 4, fp32 recurrent state; full-attention blocks use 16 Q / 2 KV heads, head_dim 256, partial RoPE | `MEASURED` |
| Residuals | Depth-softmax-mix block residuals, block size 4 | `MEASURED` |
| Routed experts | 512 per layer, top-10, plus 1 shared expert; `expert_ff` 512 | `MEASURED` |
| Hidden / vocab / context | 2048 / 129024 / 262144 | `MEASURED` |
| MTP | 1 hidden layer declared; **its weights are absent from the source checkpoint** | `MEASURED` |
| Parameter count | 79,635,961,472 (79.64 B) in llama.cpp accounting | `MEASURED` |

The parameter count has an unresolved second reading: a server load log prints
`81.29 B` for the same artifact. Both are receipts; the 1.65 B difference is
unaccounted for, so this file quotes 79.64 B as the counted value and records
the discrepancy instead of picking one silently.

## Artifact

The deployed artifact is a mixed-quantization GGUF built from the source
checkpoint. It is expert-dominated: the expert bank is ~93 % of the file.

| Field | Value |
| --- | --- |
| Bytes | 39,913,721,760 B (37.1726 GiB) |
| Tensors | 1,287 = 52 `Q4_K` + 44 `Q3_K` + 614 `Q8_0` + 577 `F32` |
| Effective bits/weight | 4.003 |
| Expert bank | 34.7227 GiB (93.41 % of artifact bytes) |
| Expert effective bpw | 3.862 (77.309 B expert parameters) |
| Per-role types | routed `ffn_down_exps` = `Q4_K`; routed `ffn_gate_up_exps` = `Q3_K` (4 of 48 blocks upgraded); trunk, KDA, norms, shared expert, embeddings, output = `Q8_0`; control/router-bias tensors = `F32` |

`DERIVED` note: only the 93.41 % share is arithmetic over two receipt fields;
every other cell is a receipt value.

## Residency and the explicit expert arena

At the deployed `-ncmoe 32` split, 32 expert layers stay on the host CPU
(23.15 GiB of expert weights, 0.723 GiB per layer) and 16 expert layers are
device-resident (11.57 GiB). The host tier is served by an **explicit arena**
over a packed `O_DIRECT` expert store, not by the OS page cache.

| Configuration | Raw decode | Class |
| --- | --- | --- |
| Cold mmap, no arena | 4.49 t/s | `MEASURED` |
| 8 GiB arena | 13.99 t/s | `MEASURED` |
| 12 GiB arena (shipped) | 14.93 t/s | `MEASURED` |
| 12 GiB arena, `-t 4` pinned to one CCX | 15.58 t/s | `MEASURED` |

The shipped 12 GiB configuration measured 11.98 GiB resident, a 95.308 % hit
rate, 2 evictions, `not_resident = 0`, `drop_pages = 1`, and greedy output
byte-identical to the non-arena path. A destructive parity test — overwriting
the whole mmap expert tensor with `0xFF` and re-running — produced
`max|diff| = 0.0`, which is the evidence that a miss is always filled from the
store rather than silently falling back to the mmap.

Measured host-tier roofline for the deployed configuration: 566.2 MB of host
expert bytes per token, 8.97 MB/token of NVMe traffic, a 28.09 GB/s CPU DRAM
ceiling, and 26.75 GB/s achieved by the deployed expert kernel (95.3 % of that
ceiling); the same kernel reads 66.9 GB/s when L3-resident, and device-local
reads measure 364.4 GB/s. Critical path: 64.2 ms/token.

**Arena hit-rate correction.** An earlier arena/cache hit-rate figure of
99.536 % was **declared invalid** by its own author because of a per-layer
unit-size bug; the corrected range is 98.545–98.771 % at 16 GiB, and the 12 GiB
production point measures 95.308 %. Separately, block-device sector counts were
compared against engine-accounted miss bytes and matched at a ratio of
**1.000016**, which is what makes the arena's accounting trustworthy rather
than self-reported.

## MTP correctness: a real bug, in Alice, not in shared infrastructure

MTP with `K > 1` initially failed rollback, and the known-answer test recorded
**276 of 682 checks failing**. Three candidate causes were falsified (dense-KV
handling, stale pending state, draft-context KV contamination) before the
actual defect was localized:

```text
defect: the Alice-local recurrent conv snapshot plane convention
        snapshot offset used min(slot, n_seq_tokens)
        instead of n_seq_tokens - slot
consequence: after a rejection, conv state and S-state were restored
        from different steps
scope: ALICE_AI only - the shared gated-delta-net op was already correct
```

The shared delta-net operator was proven correct independently: 682 checks, 0
failures. After the Alice-local fix the KAT reached **0 of 682 failures**, and
the corrected lane demonstrated greedy parity at **K = 0, 2, 3, 4** with live
rejection behaviour observed. The invariant the KAT now protects is the
plane-equals-rollback-depth contract: the snapshot plane written for token `j`
must be the plane the rollback reader selects for depth `j`.

**Do not widen this.** Later production MTP claims are narrower than they look:
the corrected-lane acceptance receipts are per-`K` acceptance vectors from a
single session family, the `K=1` accepted-throughput figure (18.739 t/s
accepted versus 9.218 t/s at `K=0`, 2.03×) is retained as `HISTORICAL` and
explicitly **not reproducible** (reproducible abort on longer runs, silent
divergence on shorter ones), and `K=2` is the best point in the merged campaign
report. The measured `K` cost curve shows why: `K=1` costs ~0.98× a forward,
while `K=2` and `K=4` cost 1.94× and 3.55× for 2.55× and 2.79× span. MTP on
Alice is a correctness result with a narrow, receipted speed case, not a
multiplier.

## Backend policy: ROCm/HIP is production, Vulkan is the reference

The campaign's standing directive is:

```text
ROCm/HIP = production and performance backend
Vulkan   = parity oracle, debug path, mechanism donor
```

New optimizations target HIP first unless a capability genuinely does not exist
there. The historical numbers are kept, labelled by backend, because they are
what the mechanism work was proven against.

| Record | Value | Backend | Class |
| --- | --- | --- | --- |
| Historical config of record | **18.430 t/s** decode-only | Vulkan | `HISTORICAL` |
| Independent re-anchor | 18.360 t/s (reproduces 18.430 to 0.4 %) | Vulkan | `MEASURED` |
| Current clean raw `K0` | 15.374 / 15.422 t/s | ROCm/HIP | `MEASURED` |
| Same comparison | 18.360 t/s | Vulkan | `MEASURED` |

`18.430 t/s` is the **historical Vulkan configuration of record**. It is not a
current HIP number and must not be presented as one. The Vulkan gain came from
execution/submission work — batching Vulkan→host split-input readbacks into a
single synchronization boundary, which took synchronization boundaries from
100.8 to 36.9 per token, waits down ~33 %, and submissions down ~5.7 %.

The HIP path is real but not finished: the HIP build faults in the MoE expert
path above a device-resident expert-layer threshold (0 faults at `-ncmoe`
36/40/44/48), and an earlier "ROCm is broken" framing was itself retracted once
the threshold and the cross-commit nature of the speed comparison were
understood.

## Prefill: a very large hot-run gain, and why it is not a stable baseline

| Configuration | Prompt-eval rate | Class |
| --- | --- | --- |
| `pp2048` / `ub4096`, third (hot) request of a warm-repeat | **662.744 t/s** | `MEASURED` (hot run) |
| Same arm, first (cold) request | 91.345 t/s | `MEASURED` |
| Same arm, second (warm) request | 245.614 t/s | `MEASURED` |
| `ub512` controlled A/B band, hot | ~486–533 t/s | `MEASURED` |
| `ub2048` control | unstable, spread 84–298 % | `MEASURED` (unusable as a control) |

662.744 t/s is a **real measured hot run in the high-throughput regime**: it is
the third request of a three-repetition warm-repeat in one session, with the
same prompt shape and `cache_prompt=false`. It is **not** a universally
reproducible stable baseline, and it must not be advertised as one.

A `71.105 t/s` baseline is sometimes placed next to it to produce a "9.3×"
headline. That baseline comes from a **different session, different `ubatch`,
and a different offload configuration**, so the ratio is cross-session and
inadmissible. The defensible statement is: within one session, the same arm
moved from 91.345 to 662.744 t/s as caches warmed, and the controlled `ub512`
A/B setting sits in the ~486–533 t/s band with a tight control spread.

`ub2048` is retained as a negative control: its spread makes it useless for
attribution. `ub8192` at context 16384 OOMs before the server is ready, which
is the measured ceiling of that ladder.

## FreeToken-inspired staging and overlap: mechanism works, production does not

This was a **mechanism-donor investigation**, not a switch to FreeToken.

Standalone HIP probe (731 MiB H2D copy against a calibrated ~60 ms spin
kernel):

```text
kernel only            ~56 ms
copy only              ~55 ms
same-stream copy+GPU   ~110 ms
separate copy stream   exposed 0.00-0.60 ms
```

So the physical copy/compute overlap mechanism itself works: on a separate
stream the copy is fully hidden.

The production experiment was a **null**. A controlled interleaved A/B/A/B/A/B
at `ub512` measured control hot median **515.858 pp/s** against F2 **507.085
pp/s** — a 1.7 % deficit, not a win.

Why it failed, with instruments:

- **Synchronization is not the cost.** Per-MoE-layer id readbacks followed by
  backend/device synchronization were instrumented and priced at **~0.2 % of a
  prefill pass**. The hypothesis that these drains starve the copy stream was
  refuted by measurement.
- **Staging destinations alias.** Adjacent host-MoE staging destinations are
  observed to alias, because the allocator sees a single logical copy. Naive
  overlap is therefore unsafe: the next layer can overwrite bytes still being
  consumed.
- **The coarse fix is memory-infeasible.** Duplicating every split input
  (`sched->n_copies = 2`, all copies marked output) failed allocation at
  **49,326.56 MiB (~48.2 GiB)** — an OOM, twice.
- **The targeted fix also failed.** Restricting the second slot to host-MoE
  expert staging tensors is the right shape, but the candidate arms did not
  load: two allocator failures and then a segfault before readiness. Only
  control arms produced numbers.

The event-ordering requirement for any future attempt is recorded: consumer
finishes slot A → event → only then may the copy stream overwrite/reuse A.
Until a candidate arm actually loads and beats its control, **there is no
staging speedup on this campaign.**

## Expert-major grouping: a real local mechanism win

Grouping routed experts into expert-major batches raises the useful work per
GEMM by a large factor locally:

| Measurement | Value | Class |
| --- | --- | --- |
| Single-layer HIP expert-major grouping, B=4096 | 0.185 → **6.115 TMAC/s** (33.0×) | `MEASURED` |
| Same, after copy-cost correction | lower bound ~43× | `DERIVED` |
| Route-materialization helper rewrite (HELPER2) | removed 19.3 % / 29.5 % of the grouped step at B=4096 / 5796 | `MEASURED` |
| Same-kernel parity gate | 32/32 bit-exact at every batch size | `MEASURED` |
| Deployed-configuration gate | `REVERT` (see below) | `MEASURED` |

The parity result matters: an earlier apparent numerical delta of `1.14e-2` was
**not** a grouping error. It was the runtime's own kernel selection — the
1-row path dispatches to MMVQ while `B >= 32` dispatches to MMQ on this GPU —
and a same-kernel replica arm proved bit-exactness at every batch size.

**Local mechanism win ≠ full-model multiplier.** The grouped expert GEMM is
~44.6 % of the relevant step, so the Amdahl ceiling for this lever is ~1.65×
(1.28× in the conservative form); the deployed-configuration gate rejected
promotion anyway. Nothing here licenses a full-model claim.

Preserved negative: forcing prefill MoE onto the CPU path measured **-85.1 %**
(581.3 → 87.1 t/s at `pp4096`). That path is retired and should not be revisited
without a fundamentally different mechanism.

## Workhorse packaging

Alice is being packaged as an actual local worker, not only a benchmark: an
OpenAI-compatible local server with one-command start/stop/status, health
checks, logs, and a production/experiment separation, wired into an
agent-harness selector as a local model.

One caveat is load-bearing. The current Alice artifact is a **base checkpoint**
and its GGUF header carries **no `tokenizer.chat_template` key at all** (all 36
metadata keys were parsed). It is a completion endpoint. Server-side rendering
plus a client-side native-format renderer makes it usable as a worker, but it
must not be described as an instruction-tuned agent model.

## What is not claimed

- No portability claim: every number is one reference machine.
- No quality claim for the mixed-quantization artifact; it has no imatrix, and
  no teacher-KL, perplexity, or task-quality gate was captured.
- No MTP multiplier: the corrected lane proves parity, not economics.
- No staging/overlap win: the production A/B is a null.
- No end-to-end effect from the expert-major grouping result.
- No instruction-following or agent capability from the base checkpoint.

## Receipts and boundaries

The receipts behind this file are campaign-local JSON/log artifacts that are
not part of this repository: the identity and build receipts, the arena
verification and destructive parity receipts, the recurrent-snapshot KAT
receipt, the acceptance-lane vectors, the prefill arm receipts, the overlap
probe receipt and A/B analysis, the grouping attribution receipts, and the
runtime ledger. They are summarized here with their exact values and are
recorded as a bounded disposition lane in
[`repro/alice/`](../../repro/alice/). Model weights, container bytes, and the
executing runtime branch are excluded from publication.
