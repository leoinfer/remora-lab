# Memory hierarchy

The memory model is explicit:

1. Model storage is caller-owned and read through bounded Rust I/O.
2. Host-memory pages are represented by typed residency records.
3. Device-memory pages are admitted only when capacity and generation checks
   pass.
4. KV representations carry prefix, position, layer, format, epoch, and
   generation identities.
5. Eviction and reconstruction are observable; stale or ambiguous state fails
   closed.

The `har-residency`, `har-storage`, `har-kv`, and `har-contextfold` crates
implement these contracts. Their accounting is not a promise that every
model can fit in a given device budget. It is an auditable control plane for
deciding what may be resident.
