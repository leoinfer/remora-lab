# Flash-Next representation panel, 2026-09-18 — bounded disposition

This lane records the tensor- and activation-level representation measurements
taken on 2026-09-18: the base codec comparison (raw ternary T1, LS-reencoded T1,
the Q2_0 donor scaffold, and T1 plus a dense ternary T2 plane), the equal-added-
byte correction panel, and the Q8_0 residual island on layer 0 / expert 283.

Run from the repository root:

```sh
./repro/flash-next/representation-panel/run.sh
```

The command prints the bounded disposition and exits successfully. No model is
loaded. The lane does not invoke Python, C++, llama.cpp, GGML, CMake, a
subprocess helper, or a foreign execution backend.

**Status: `EXCLUDED_WEIGHTS_DATA`.** The measurements are real and the sanitized
values are in [`sanitized_receipt.json`](sanitized_receipt.json), but the
reproduction inputs are excluded: the BF16 source tree, the quantized container,
the panel scripts, the captured activation sets, and the raw run receipts.
Because of that, this lane is a bounded disposition rather than a runnable
public benchmark and is not part of the model-free release gate.

What the numbers do and do not support:

- Every value is a **fidelity** measurement: weight relRMS against the BF16
  authority tensor, or activation-space relative output error on captured MoE
  inputs. Neither is capability evidence.
- The teacher-KL probe for these candidates was never captured, so these panels
  say nothing about end-to-end model quality.
- The residual island is **off-budget**: it costs about 4.86× the equal-byte
  budget used by the correction panel, so it is mechanism evidence only.
- The equal-byte panel verifies its own byte constraint; that verification is
  part of the receipt.

Metric definitions, per-slice caveats, and two defects found while publishing
this material are recorded in
[`research/representation/README.md`](../../../research/representation/README.md).
