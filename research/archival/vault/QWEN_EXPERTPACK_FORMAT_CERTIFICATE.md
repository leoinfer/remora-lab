# H11 XP2 exact ExpertPack format gate

**Status:** PASS for offline one-expert byte reversibility; NOT a transport or
performance certificate.

## Evidence

- Format: `QWEN_EXPERTPACK_FORMAT_SPEC.md`
- Schema: `qwen_expertpack.schema.json`
- Generator/verifier: `scripts/expertpack_roundtrip.py`
- Result: `raw/roundtrip/roundtrip-results.json`
- Result SHA-256: `see artifact manifest`; all four records report
  `roundtrip=PASS`.

The records use model SHA-256
`f6b6c6d5cfa6f00d964eeb7add28eb14ce7481734d506b90681007678cd2c484` and
canonical source-map offsets. The proof covers:

| Geometry | Layout 1 | Layout 2 | Payload |
|---|---|---|---:|
| ordinary layer 0 expert 0, Q6_K/Q6_K/Q8_0 | PASS | PASS | 2,834,432 B |
| exceptional layer 1 expert 0, Q8_0/Q8_0/Q8_0 | PASS | PASS | 3,342,336 B |

Each gate/up/down source span was copied and independently hashed; the unpacked
record bytes matched the source bytes exactly and the complete payload checksum
matched. The initial Layout 2 metadata uses the same canonical bytes and aligned
views as Layout 1; it has not been shown faster and is not yet connected to the
Vulkan destination slots.

This gate does not prove one-copy GPU delivery, fewer submissions, lower CPU
handling, lower joules/token, or full-model exactness. Those are XP3–XP6 gates.

## Repair addendum (2026-08-05T23:0xZ, witness lane session 019fd3fc)

The V5 verifier finding of stale embedded `metadata_bytes` is confirmed and repaired.

- Defect: v1 records embedded `metadata_bytes` computed before the final re-encode
  (stale by 24 bytes on all four records: header 1465 vs embedded 1441 layer-0;
  header 1468 vs embedded 1444 layer-1).
- Fix: `scripts/expertpack_roundtrip_v2.py` (fixed-point metadata length; original v1
  preserved at `scripts/expertpack_roundtrip.v1-stale-metadata.py`).
- Regenerated: `raw/roundtrip-v2/` (4 records + results + REPAIR_RECORD.md); v1 records
  preserved at `raw/roundtrip/`.
- Verified: embedded metadata_bytes == header meta_bytes == actual JSON length on all
  four v2 records; payload bytes byte-identical to v1 (whole_record_sha256 and
  projection_sha256 unchanged); record_bytes unchanged.
- Status: XP2 offline one-expert byte-reversibility gate RE-ESTABLISHED (record-format
  PASS). XP3–XP6 (transport/speed/energy/full-model) remain NOT certified.
