# DSpark / MTP

**Status:** `DEFERRED` pending model-shaped validation

DSpark/MTP is the speculative-decoding line: predict future tokens, verify a
variable horizon, and charge the controller for both accepted and rejected
work. The retained designs include elastic horizon depth, sidecar or
layer-pack predictors, expert-major multi-position execution, small branch
trees, and the accepted-token objective.

The research source set is [`DSpark-MTP-Architecture.md`](../archival/vault/DSpark-MTP-Architecture.md),
[`DSPARK_ADAPTIVE_HORIZON.md`](../archival/vault/DSPARK_ADAPTIVE_HORIZON.md),
[`Progressive-Future-Materialization.md`](../archival/vault/Progressive-Future-Materialization.md),
and the [`mtp-speculation`](../mtp-speculation/) notes. Official tensor
definitions, acceptance quality, and full-model end-to-end evidence remain
gaps; see the negative and missing-data records.

The production runtime may expose Rust scheduling contracts, but no Python
or C++ drafter is a runtime dependency.
