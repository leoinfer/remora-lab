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

The parser checks checked arithmetic, row-window bounds, and the 256-value /
144-byte geometry. The public tests use synthetic data; model-derived capture
fragments and distributable model weights are not included. Full-model
compatibility and cross-implementation byte agreement remain open work.
