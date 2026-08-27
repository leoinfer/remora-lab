# H11 ExpertPack XP0 — current Qwen Q2 transport path audit

**Date:** 2026-08-03
**Status:** XP0 measured audit PASS/PARTIAL; no ExpertPack implementation or
speed claim.

## Authority and controls

- Runtime: `[local path omitted]`, branch
  `qwen-compact-q2`.
- Q2 code commits: `45195fc23` (phase timestamps), `55ba33a07` (energy phase
  timestamps), `6b0111465` (matched performance harness); XP0 counters are in
  the current dirty diff on top of these commits.
- Model: `[local path omitted]`, SHA-256
  `f6b6c6d5cfa6f00d964eeb7add28eb14ce7481734d506b90681007678cd2c484`.
- GPU: AMD Radeon RX 9060 XT / RADV GFX1200; queue flags unset;
  `DSV4_STAGING_MB=1024`; all 40 routed layers selected; Q6_K/Q8_0 mixed
  source authority remains host/mmap.
- Fixed audit command:

```bash
cd [local path omitted]
env -u RADV_EXPERIMENTAL -u GGML_VK_ALLOW_GRAPHICS_QUEUE \
  -u DSV4_QWEN_Q2_VERBOSE LD_LIBRARY_PATH="$PWD/build-q2/bin" \
  DSV4_STAGING_MB=1024 DSV4_QWEN_PERF_TOKENS=8 \
  DSV4_QWEN_PERF_ORDER=C DSV4_QWEN_FULLCORE_NGL=40 \
  DSV4_QWEN_PROMPT='ExpertPack XP0 current transport operation audit.' \
  ./build-q2/bin/qwen-q2-performance
```

Raw output: `raw/xp0-current-transport-counters-8-quiet.log`; the verbose
comparison is `raw/xp0-current-transport-counters-8-verbose.log`. The quiet
run uses the XP0 instrumentation and gates per-route diagnostic `fprintf`
without changing the route, slot, copy, or MMID mechanism. Existing certified
performance logs predating this audit retain their original logging state and
are not mixed with XP0 timings.

## Stage ledger

| Stage | Existing path | XP0 measurement / finding | Scope limit |
|---|---|---|---|
| 1. Router top-k available | Qwen `build_moe_ffn()` produces `ffn_moe_topk-*`; scheduler callback bridge identifies selected layers. | `route_prepare_calls=680` for 17 epochs × 40 layers; top-k is one position. | Callback/scheduler time outside manager is not separately timed. |
| 2. Expert source lookup | `on_topk()` keeps original IDs; `source_span()` does pointer + bounds arithmetic for each projection. | `source_span_lookups=16,320` = 3 × 5,440 logical requests; hits still re-lookup all three spans. | Attached process-level page-fault run records 1,807,570 minor / 361,700 major deltas; not per-expert attribution. |
| 3. Gate/up/down lookup | Three independent `source_span()` calls, independently strided. | No triplet lookup object; exactly three projection operations per routed expert. | This is the first direct ExpertPack candidate boundary. |
| 4. Hit/miss | Associative original-ID→slot map; `m_upload_all` can force hit re-upload. | `hits=2,039`, `uploads=3,401`, `changed_slots=3,401` for 5,440 requests. | Slot policy deliberately held fixed; no route-policy ablation. |
| 5. Slot selection | Free slot then oldest unoccupied slot; fence checks pending copies before repurpose. | All fail-closed counters zero; 17 fence sequences. | No alternative policy tested in XP0. |
| 6. Staging allocation | `stage_copy()` calls `dsv4_vk_staging_alloc()` per projection and IDs. | Manager `staging_allocations=10,883`, same as manager copy regions. Vulkan total is 10,900 including 17 initialization transfers. | Allocator lock/metadata subtime not separately split. |
| 7. CPU copy/mapping | `std::memcpy(staging, src, bytes)` for each changed gate/up/down and each ID vector. | `cpu_memcpy_calls=10,883`, `cpu_memcpy_bytes=9,697,318,144`. | No CPU gather; current path copies directly into separate staging ranges. |
| 8. Vulkan copy-region construction | `dsv4_vk_defer_copy()` appends one `VkBufferCopy` record per manager copy. | Manager `copy_regions=10,883`; Vulkan `vk_deferred_regions=10,900`, `vk_deferred_bytes=9,697,326,848`. | Region recording is counted; Vulkan driver internal copies are not. |
| 9. Command-buffer recording | `ggml_vk_dsv4_drain_deferred_copies()` emits all pending copy regions into the current compute context and barriers. | `vk_record_batches=697`, approximately 40 layer batches × 17 epochs plus epoch-boundary activity. | A batch is not necessarily a newly allocated command-buffer object. |
| 10. Queue submission | `ggml_vk_submit()` is attributed when a deferred-copy flag is pending. | `vk_queue_submissions=697`; no separate per-region submits. | This is transport-attributed submit count, not every Vulkan submission in the process. |
| 11. Synchronization/fence | `begin_decode()` calls `dsv4_vk_drain_now()` then resets the epoch; scheduler synchronizes graph ranges; optional verify drains. | `sync_calls=17`, `sync_us_total=1,899`, last `sync_us=4`; explicit drain submissions/waits were zero because graph submission consumed pending copies. | Manager timing does not include all scheduler fence wait time; this is a known instrumentation boundary. |
| 12. Tensor views | Four persistent GGML objects/layer: gate/up/down/IDs; destination strides match source. | `tensor_objects_constructed=160` for 40 layers; no per-route tensor construction. | Descriptor updates and driver descriptor-cache work are not exposed by current API. |
| 13. `MUL_MAT_ID` | Qwen graph passes local IDs to native Vulkan MMID with persistent compact tensors. | Correctness gate PASS; 8 generated tokens hash `bcbd5cc525785f6e`. | Shader GPU duration requires Vulkan timestamp/perf logger correlation. |
| 14. Slot lifetime | Pending copies are fenced at decode boundary; `slot_pending_copy` prevents premature repurpose. | zero repurpose-before-fence and stale-slot counters. | No deliberate fence violation in this audit; existing fence oracle remains authority. |

## Measured aggregate

The run had 9 prompt tokens + 8 generated tokens = 17 epochs and 5,440 routed
expert requests. The average across all epochs was:

- 640.176 manager copy regions/epoch;
- 570,430,479 logical bytes/epoch (`~544.0 MiB`);
- 259,166.6 µs route preparation/epoch with per-route diagnostics disabled;
- 40.0 route callbacks/epoch;
- 41.0 transport-attributed record batches/submissions/epoch;
- 3 source-span operations per logical request;
- the measured aggregate is 3,401 / 680 = 5.0015 uploaded experts per
  callback;
- 16.0044 copy regions/callback, including one ID copy plus three projection
  copies per uploaded expert.

The `QWEN_Q2_STATS` line is PASS and every required fail-closed counter is zero:
staging, missing lookup, unmapped route, premature repurpose, stale read,
source/destination OOB, upload, route, and precision counters.

## Exact timing audit of `upload_us`

The `m_last_upload_us` timer begins at entry to `on_topk()` after the selected
layer is found and ends after the IDs `dsv4_vk_defer_copy()` call. It includes:

1. top-k device-to-host read when the top-k tensor is not host-resident;
2. vector allocation and ID validation;
3. associative slot lookup/eviction;
4. three source pointer/bounds lookups per routed expert;
5. host `memcpy` into staging for changed projections;
6. staging allocator and deferred-copy vector operations;
7. local-ID staging/copy;
8. any enabled per-route diagnostic `fprintf` calls.

It excludes the later Vulkan command-buffer recording, actual queue submit,
GPU DMA completion, scheduler fence wait, `MUL_MAT_ID`, and MoE output. Thus the
historical aggregate `upload_us` field is a **CPU route-preparation plus
recording/bookkeeping timer**, not end-to-end upload latency and not DMA time.
`route_prepare_us` is the accumulated version of that same scope. The manager
`sync_us` field is only the explicit drain call's scope and is not the full
scheduler critical-path fence time. Any future XP comparison must report these
scopes separately.

## What XP0 establishes

1. The current path performs three physically separate source span operations
   and, for every uploaded expert, three CPU copies, three staging allocations,
   and three Vulkan copy regions. IDs add a fourth per-route region.
2. Current changed-slot persistence reduces expert uploads, but it does not
   coalesce a gate/up/down triplet. A one-record source or destination layout
   would directly test this handling cost without changing canonical bytes.
3. The route path is not one queue submit per projection: pending regions are
   batched into approximately one transport-attributed graph submission per
   layer range. Submission reduction and physical packing must therefore be
   measured independently (P2/P3/P4/P5).
4. Tensor objects are persistent (160 constructed once for 40 layers); generic
   per-token tensor construction is not the primary exposed count. Descriptor
   updates remain unmeasured and must not be claimed as the bottleneck.
5. The dominant measured exposed handling is CPU preparation and many separate
   copy regions. In the attached page-fault run, the qwen process accumulated
   approximately 1,807,570 minor and 361,700 major faults over model load plus
   the 8-token case. This is an observed process-level signal, not yet assigned
   to individual expert spans; GPU idle gaps, actual DMA completion, and MMID
   GPU time remain open instrumentation items.

## Required XP0 follow-up

- page-fault-capable sampler evidence is attached at
  `raw/xp0-pagefaults-8.samples.jsonl` (minor delta 1,807,570; major delta
  361,700; model-load plus case; not a per-expert attribution);
- correlate Vulkan timestamp/perf logger output with the 697 transport batches;
- add explicit descriptor-update/tensor-view counters only if the backend API
  permits it without changing execution;
- proceed to XP1 source-map and XP2 reversible format design; do not call the
  current path transport-bound by disk layout until the source map is read.
