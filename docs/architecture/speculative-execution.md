# Speculative execution

The serving layer models draft proposals, target verification, accepted-token
count, rejected work, and resource use as separate events. MTP or a draft
model is useful only when the acceptance policy and the extra work are
measured together.

The implementation has deterministic scheduler contracts and telemetry. It
does not assume that a larger draft horizon is faster. A release benchmark
must include a no-speculation baseline, the same prompt and model identity,
accepted and rejected token counts, latency, and memory/transfer accounting.
