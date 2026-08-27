# Benchmark methodology

The public benchmark harness is not complete in this candidate. The intended
receipt schema is:

```text
model identity + format
hardware/driver context
binary and source identity
command line and environment policy
prompt and generated-token count
warm-up and repetition policy
latency/tokens-per-second/energy
accepted and rejected speculative work
resident and transferred bytes
raw output digest and status
```

Comparisons must use the same model file, tokenizer behavior, prompt, output
length, sampling policy, and warm-up. Baselines are named explicitly. Until
these fields are public and reproducible, the result remains experimental or
historical.
