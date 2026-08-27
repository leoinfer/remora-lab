# Qwen historical decode baseline — bounded disposition

This lane records the result of the public-availability audit for the
historical Qwen decode anchors commonly summarized as approximately 20.6 and
33.8 tokens/s. The exact model artifact, executor build, and authoritative
receipt were not cleared for publication, so this lane deliberately does not
restate either number as a reproducible benchmark.

Run from the repository root:

```sh
./repro/qwen27b/historical-baseline/run.sh
```

The command prints the machine-readable disposition and exits successfully.
That success means the evidence gap is recorded and bounded; it is not a
successful model benchmark. No model weights are loaded, and the lane invokes
no Python, C++, llama.cpp, GGML, CMake, subprocess helper, or foreign runtime.

The public historical summary remains useful as provenance, but an exact
tokens/s claim requires a cleared model hash, executor identity, prompt,
timing receipt, and correctness gate.
