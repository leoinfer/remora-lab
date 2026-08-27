# ContextFold counterexamples and kill cases

1. **262K Qwen F16 capacity.** The ordinary attention KV alone is about 16 GiB,
   while the prior placement map reports only 619 MiB free at NGL26 and 3,244
   MiB at NGL20. Entire-KV VRAM residency is therefore a theoretical baseline,
   not a feasible limited-VRAM deployment.
2. **Cold bandwidth.** A 262K F16 scan is about 17.18 GB decimal. At the prior
   6.4 GB/s PCIe/GTT measurement its transfer lower bound is about 2.6 s per
   one-token scan (before compute and synchronization). At 6.9 GB/s RAM it is
   about 2.5 s. K-query batching divides movement only across accepted queries,
   not to zero.
3. **Compression without entropy.** Random authority bytes selected raw at
   ratio 1.0. Base plus raw residual can increase size. Synthetic repeated
   patterns compressed dramatically, but that is not real KV evidence.
4. **Codec decode bottleneck.** The best synthetic bit-plane result had a pure
   Python decode rate near 0.90 MB/s in the recorded run; `encoded/bandwidth + decode` was slower
   than raw transfer even with a large ratio.
5. **Replay latency.** Token archive plus checkpoint uses little memory but at
   an assumed 100 replay tokens/s has no tested Qwen case meeting both storage
   reduction and a 10-second cold reconstruction deadline.
6. **Key-first all-survive.** When every value block survives, key-first adds a
   pass and overhead; unverified lower survival is approximate.
7. **RSSO no novelty.** Existing batching already applies multiple query rows to
   resident KV. A stationary model beats separately submitted transfers, but
   not the modeled already-batched baseline.
8. **MTP/recurrent legality.** Speculative acceptance alone does not establish
   compatible recurrent snapshots; grouping illegal candidate positions breaks
   the state authority.
9. **Exact-zero scarcity.** A mass threshold, next-token match, or bad norm bound
   is not a certificate. The prototype's authority-boundary composition is
   explicitly unproven.
10. **NVMe dependency.** If every generated token synchronously reads NVMe,
    queueing and device latency dominate; the proposal is narrowed to warm RAM
    or replay materialization ahead of the critical path.

These counterexamples are preserved as negative evidence.  They support a
capacity-only interpretation, not a live speed claim.
