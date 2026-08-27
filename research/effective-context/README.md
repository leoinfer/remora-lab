# Effective-context research

The local program explored whether storage tiers, KV compression, retrieval,
and quality-aware reconstruction could provide more usable history than a
plain dense cache under the same hardware budget. The public conclusion is a
hypothesis, not a ten-million-position generation result.

The next public experiment should use synthetic pages, a declared quality
metric, an attended-position fraction, recovery tests, and a model-free
accounting harness. The `R4KV` and ContextFold notes are related but are not
interchangeable: a byte saving does not prove preserved model quality.
