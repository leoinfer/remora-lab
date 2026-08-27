# R4F experimental format profile

R4F is the experimental Flash-Next bring-up label. No stable public byte
format is frozen in this release candidate.

The intended container must bind model identity, tensor layout, recurrent or
attention state layout, quantization profile, and recovery metadata. A loader
must reject unknown versions and incomplete coverage. This document is a
design boundary only; use [docs/flash-next.md](../../docs/flash-next.md) for
the readiness criteria.
