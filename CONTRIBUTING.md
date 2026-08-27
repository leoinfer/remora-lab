# Contributing

Keep production host/runtime code in Rust. GPU shader source and documented
driver/system boundaries are allowed; external inference implementations are
not.

Every new research artifact needs a source identifier, license/provenance
record, privacy review, and a status in the claims ledger. Do not commit model
weights, checkpoints, private logs, screenshots, local paths, credentials,
machine serials, or raw benchmark output that cannot be reproduced publicly.

Changes that affect format bytes, runtime admission, model loading, memory
accounting, or performance claims require focused tests and an update to the
relevant specification or falsified-results record.
