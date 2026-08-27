# Flash-Next full-model generation — bounded disposition

The public repository contains the R4F/Flash-Next format and bring-up
documentation, but the full-model first-token and generation gates are not
closed. This lane makes that boundary executable and discoverable without
turning an active research campaign into a public performance claim.

Run from the repository root:

```sh
./repro/flash-next/full-model/run.sh
```

The command prints the blocked provenance receipt and exits successfully. No
model is loaded. The lane does not invoke Python, C++, llama.cpp, GGML, CMake,
a subprocess helper, or a foreign execution backend.

The current campaign record remains the place to follow bring-up work. A
future release can replace this disposition only after the exact model,
first-token correctness gate, generation receipt, and runtime boundary are
publicly cleared.
