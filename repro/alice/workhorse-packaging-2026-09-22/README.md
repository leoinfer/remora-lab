# Alice workhorse packaging lane

Bounded public disposition for Alice as a packaged local worker rather than a
benchmark only.

```sh
./repro/alice/workhorse-packaging-2026-09-22/run.sh
```

The packaging is real: an OpenAI-compatible local endpoint, one-command
start/stop/status, health checks, logs, and agent-harness integration, running
the artifact at its native 262144 context with `q8_0` KV in host RAM.

The boundary is equally real: the artifact is a **base checkpoint** and its GGUF
header carries **no `tokenizer.chat_template` key at all** (all 36 metadata keys
were parsed). It is a completion endpoint, and it must not be described as a
properly instruction-trained agent model.
