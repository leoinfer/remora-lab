# Shadow-price scheduling

**Status:** `PROPOSED`

Shadow-price scheduling assigns an explicit opportunity cost to scarce VRAM,
RAM, bandwidth, verification time, and reserve capacity. Policies can then
compare transfer, compute, residency, and recovery decisions in a common
ledger rather than optimizing one subsystem in isolation.

The formal checker target is [`F13`](../../theory/checkers/F13-checker.md).
The source concepts are preserved in the HERMES atlas and REMORA research
notes; no optimality claim is made.
