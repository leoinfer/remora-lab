# Falsified and bounded results

This record prevents attractive but unsupported numbers from becoming folklore.

## SWMMAC multi-POPS

The proposed multi-POPS result is invalidated. The acceptance contract did not
establish a valid end-to-end measurement, and the accounting was not strong
enough to support the headline. The result is retained only as a research
question: define the unit, publish the kernel, publish the exact hardware
configuration, and compare against a public baseline under identical work.

## Faster-than-baseline claims

Several local experiments optimized a narrow kernel or a small resident slice.
That does not establish faster generation. Historical observations include
paths that trailed llama.cpp by several tokens per second. The public project
therefore makes no speedup claim until a new receipt contains the model
identity, prompt, warm-up policy, token count, hardware context, command,
binary identity, and raw output.

## Effective context

The “10M” target is an effective-context hypothesis based on storage,
compression, retrieval, and quality budgeting. It is not a run of dense
attention over ten million positions. Any future claim must report the active
representation, exact quality gate, recovery behavior, and the fraction of
context that was actually attended.

## Flash-Next / R4F

The bring-up work established enough interfaces to continue investigation but
not enough to claim full-model generation. Missing evidence includes complete
weight coverage, numerically checked recurrent state transitions, recovery
after rejected speculative work, and a public end-to-end receipt.

## MTP and expert residency

Acceptance telemetry and expert-read accounting are useful instrumentation,
not speedup proofs. A lower read count can coexist with worse latency,
contention, or quality. Future benchmarks must report all of those dimensions.
