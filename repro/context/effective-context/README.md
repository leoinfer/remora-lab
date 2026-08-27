# Effective-context accounting and shortcut-failure lane

This is the public, model-free reproduction boundary for the effective-context
research. It creates deterministic address/value records over a ten-million
position address space, encodes sorted address deltas with unsigned LEB128,
decodes them exactly, performs a lexical lookup probe, and runs an intentionally
weak prefix shortcut failure fixture.

Run from the repository root:

```sh
./repro/context/effective-context/run.sh
```

The result is an accounting and adversarial-test receipt. It is not dense
transformer attention at ten million tokens, a dense 10M-token KV cache, or a
semantic retrieval benchmark. `semantic_retrieval.r10` is intentionally null.
