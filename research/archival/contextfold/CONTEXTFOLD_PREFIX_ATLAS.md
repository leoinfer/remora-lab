# Causal shared-prefix atlas

An immutable ContextPack may be shared across sessions only when

```
ContextRoot = Merkle/Hash(model, runtime generation, exact prefix,
                           positions, layer, state kind, precision, codec,
                           recurrent generation, execution epoch)
```

roots must match exactly.  Equal text at different positions, different model
hashes, changed runtime kernels, or divergent recurrent generations are not
aliases.  The CPU simulator shows large modeled savings at high true shared
prefix rates, but those numbers are sensitivity results; no session trace was
available.  Text-only false matches are explicitly counted as rejected.

The current llama.cpp source has sequence/cell sharing and prompt caching, but
not this cross-session authenticated root/atlas protocol.  ContextFold would
add immutable lifetime/refcounts, dependency closure, hash verification, and
branch divergence handling rather than rename the existing sequence APIs.
