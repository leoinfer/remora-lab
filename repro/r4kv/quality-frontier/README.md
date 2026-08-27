# R4KV model-quality frontier — bounded disposition

The public R4KV storage lane is fully reproducible, but it is not a model
quality experiment. This companion lane records the missing quality frontier
for K6V4, K4V4, and K4V3 rather than promoting storage arithmetic or a
synthetic proxy KL into perplexity or attention-quality evidence.

Run from the repository root:

```sh
./repro/r4kv/quality-frontier/run.sh
```

The command prints the blocked provenance receipt and exits successfully. It
does not load weights or invoke Python, C++, llama.cpp, GGML, CMake, a
subprocess helper, or a foreign execution backend.

For the reproducible portion, use
[`../storage/run.sh`](../storage/run.sh). A real frontier still needs a
cleared model, tokenizer, evaluator, prompt/data protocol, raw metrics, and a
known-answer or parity gate.
