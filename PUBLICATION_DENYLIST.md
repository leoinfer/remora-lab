# Publication denylist

The following are excluded from the fresh public history:

- private worktree history, swarm/agent handoffs, prompts, scratch files, and
  internal ledgers;
- absolute local paths, usernames, hostnames, serial numbers, IP addresses,
  credentials, access tokens, and environment dumps;
- model weights, checkpoints, tokenizer payloads, datasets, caches, and large
  binary artifacts;
- C, C++, CMake, Python, llama.cpp, GGML, or foreign inference components in
  the HAR production path;
- copied upstream repositories, generated vendor trees, and files with
  unresolved provenance or license obligations;
- benchmark claims whose command, hardware context, raw output, or baseline
  cannot be reproduced publicly.

The denylist is intentionally explicit so that a future maintainer can audit
an apparently harmless file name or archive before adding it.
