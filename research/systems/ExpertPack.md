# ExpertPack

**Status:** `PROPOSED`

ExpertPack is a transport and residency design for mixture-of-experts
weights. An expert capsule carries layout, precision, epoch, integrity, and
placement metadata so hot, warm, and cold copies can be moved or reused
without confusing storage presence with execution readiness.

The source specifications are [`ExpertPack-Transport.md`](../archival/vault/ExpertPack-Transport.md),
[`QWEN_EXPERTPACK_FORMAT_SPEC.md`](../archival/vault/QWEN_EXPERTPACK_FORMAT_SPEC.md),
and [`qwen_expertpack.schema.json`](../archival/vault/qwen_expertpack.schema.json).
Related ideas include expert-major batching, predictive residency, frozen
epochs, dual/tri-path joins, cost-triggered repacking, and value-of-residency.

The public candidate includes only reviewed Rust format/control surfaces and
schemas. It excludes model expert payloads, private residency traces, and
claims that a particular cache policy is optimal.
