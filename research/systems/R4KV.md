# R4KV

**Status:** `PARTIALLY_IMPLEMENTED`

R4KV is the compressed key/value-page research line. It records page geometry,
precision variants, hot/warm/cold state, restoration, transfer accounting,
and the boundary between a storage codec and a model-quality result. Names
such as K4/V3, K6/V4, and related variants identify experiments and should not
be read as a universal quality guarantee.

The reviewed Rust implementation is [`har/crates/r4kv`](../../har/crates/r4kv/).
The source specifications and audits are [`QWEN_EXPERTPACK_FORMAT_SPEC.md`](../archival/vault/QWEN_EXPERTPACK_FORMAT_SPEC.md),
[`QWEN_EXPERTPACK_FORMAT_CERTIFICATE.md`](../archival/vault/QWEN_EXPERTPACK_FORMAT_CERTIFICATE.md),
and the R4KV-related records in the [`idea index`](../../RESEARCH_IDEA_INDEX.md).

No model payload, private profile, or raw receipt is distributed. A codec can
be tested for round-trip and structural correctness without implying that a
full model or long-context workload has been validated.
