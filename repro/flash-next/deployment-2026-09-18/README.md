# Flash-Next deployment record, 2026-09-18 — bounded disposition

This lane records the 2026-09-18 deployment measurements for Qwen3.8 Flash-Next
on the reference machine: coherent end-to-end generation, the deployment canary
and sustained-generation results, the 262,144-token Q8 KV context pass, the
matched control/prefetch pair, the route distribution, and the V2 hot-region
experiment.

Run from the repository root:

```sh
./repro/flash-next/deployment-2026-09-18/run.sh
```

The command prints the bounded disposition and exits successfully. No model is
loaded. The lane does not invoke Python, C++, llama.cpp, GGML, CMake, a
subprocess helper, or a foreign execution backend.

**Status: `EXCLUDED_WEIGHTS_DATA`.** The measurements are real and the sanitized
values are in [`sanitized_receipt.json`](sanitized_receipt.json), but the
reproduction inputs are deliberately excluded from this repository: the model
weights, the quantized scaffold container, the research runtime branch that
executed the runs, the raw run logs, and the host identity. Because of that,
this lane is a bounded disposition rather than a runnable public benchmark and
is not part of the model-free release gate.

What the numbers do and do not support:

- The canary verdict is a **judgement over printed outputs**; the harness has no
  automated answer grader. `returncode: 0` is the machine-checked part.
- The 262,144-token context pass is a **capacity and coherence smoke**, not a
  long-context quality result. The 384K profile was defined but never run.
- The control/prefetch pair is one **matched pair** of runs, executed
  back-to-back, with machine-wide device counters and no prefetch emission
  counters. The ratios are an observed pair, not a controlled experiment with
  counter-balancing.
- The V2 hot-region experiment is a **migration configuration**: the cold 58% of
  the expert population and the entire non-expert core still run on the
  temporary Q2 scaffold.
- No BF16 capability-retention, benchmark-retention, or final-V2 quality result
  is claimed anywhere in this lane.

The dated narrative and defect record live in
[`research/flash-next/CURRENT_RESEARCH_LOG.md`](../../../research/flash-next/CURRENT_RESEARCH_LOG.md);
the storage-path result with its full caveat list is
[`research/flash-next/PREFETCH_RESULT.md`](../../../research/flash-next/PREFETCH_RESULT.md);
the models behind the numbers are documented in
[`research/moe-residency/README.md`](../../../research/moe-residency/README.md).
