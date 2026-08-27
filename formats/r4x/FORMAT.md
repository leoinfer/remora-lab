# R4X experimental format profile

R4X is an experimental weight representation used by a research lane. This
document records the profile represented by the current Rust parser; it is not
a claim that the name or numeric identifier is an ecosystem standard.

## D32A profile

- one block represents 256 values;
- one block occupies 144 bytes;
- the represented block has eight little-endian half-precision scale values
  (16 bytes) followed by 128 packed four-bit values;
- rows use ceiling division for a partial final block;
- the current model-file type identifier used by the parser is 36.

The byte layout is:

```text
bytes 0..16    eight little-endian binary16 scales, one per 32 values
bytes 16..144  128 packed bytes; element 2i is the low nibble and 2i+1 the high nibble
```

The signed code for a nibble is `nibble - 8`, clamped to `[-8, 7]`. The
clean-room fixture encoder chooses each group scale as the binary16 round-to-
nearest-even representation of `max(abs(group)) / 7`; a zero group uses a zero
scale. The public geometry/encoding known-answer test is available through
[`repro/r4x/width-sweep/run_width_sweep.sh`](../../repro/r4x/width-sweep/run_width_sweep.sh).

The recovered full-model logical-prefill-row receipt is a separate historical
execution record. In that receipt, `W` means `n_prompt` (logical prefill rows),
not a shader workgroup or local size. See the
[`R4X logical-prefill-row lane`](../../repro/r4x/width-sweep/) for the exact source
commit, configuration, sanitized rows, malformed `ubatch=4096` boundary, and
the explicit distinction between diagnostic rows/s and generation tokens/s.

The parser checks checked arithmetic, row-window bounds, and the 256-value /
144-byte geometry. The public tests use synthetic data; model-derived capture
fragments and distributable model weights are not included. Full-model
compatibility and cross-implementation byte agreement remain open work.
