# R4KV storage and page known-answer lane

This lane is a model-free Rust reproduction of the public R4KV storage
contract. It checks the five declared K/V profiles, deterministic codec
round-trip statistics, byte accounting, page identity validation, and
fail-closed corruption handling.

Run it from the repository root:

```sh
./repro/r4kv/storage/run.sh
```

The command invokes `tools/repro-harness` directly through Cargo. It does not
load model weights, invoke Python or C++, link llama.cpp/GGML, or call a
foreign execution backend.

The fixture contains 4096 deterministic `f32` values (`seed=0`) and a 256-byte
page body. The profile names and byte counts are the public Rust profile
definitions. The positive-vector KL value is only a synthetic codec proxy;
there is no perplexity, attention-quality, GPU-parity, or long-context claim.

The recovered receipt is [`sanitized_receipt.json`](sanitized_receipt.json).
The missing model-quality frontier remains an explicit gap: this lane does not
turn a codec KAT into a K6/V4 or K4/V3 model-quality result.
