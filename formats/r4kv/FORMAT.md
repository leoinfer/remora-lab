# R4KV experimental format profile

R4KV is an experimental KV-page and compressed-group format implemented in
`har/crates/r4kv`. Its current Rust constants are authoritative for this
candidate.

## Group formats

Each 32-element quantization group carries its scale metadata and payload:

| ID | Name | Group bytes |
| ---: | --- | ---: |
| 0 | F16 | 64 |
| 1 | Q8 | 34 |
| 2 | Q6 | 26 |
| 3 | Q4 | 18 |
| 4 | Q3 | 14 |

The represented page header is 96 bytes and begins with the little-endian
magic `R4KV` and version 1. A page binds prefix digest, token/position range,
layer range, K/V formats, epoch, generation, payload length, and optional
sketch flags. Restore rejects mismatched identity or generation.

Profiles currently include F16/F16 reference, Q8/Q8 baseline, K6/V4, K4/V4,
and K4/V3 candidates. The byte arithmetic is not a quality guarantee. The
wire format remains experimental until an independent decoder and public
cross-language vectors exist.
