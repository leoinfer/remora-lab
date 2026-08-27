# ContextPack format (CF-L2 / CF-L3)

Status: **static schema and round-trip prototype; not a runtime format**.

A ContextPack is an immutable, content-addressed artifact for one exact state
slice.  Its causal identity includes model, runtime, prefix, position, layer,
state kind, authority precision, codec generation, recurrent state generation,
and execution epoch.  A token sequence by itself is not a valid key.

## Required record

`CONTEXTPACK_SCHEMA.json` defines the JSON record.  The payload is stored at
`base_offset` and the listed `residual_offsets`; the record carries raw and
encoded byte counts and an `exact_payload_hash`.  `state_kind` distinguishes
ordinary attention K/V from recurrent, GDN, SSM, and complete checkpoint state.

The initial prototype uses a simple payload hash and Python-side validation.
Production use would need a versioned binary header, authenticated integrity,
crash-safe commit protocol, range checks against the backing file, and a codec
registry pinned by hash.

## Validity contract

A block is reusable only when all of the following match the requesting state:

```
ModelMatch AND PrefixMatch AND PositionMatch AND StateGenerationMatch
AND RuntimeGenerationMatch AND ExecutionEpochMatch AND NotExpired
```

`contextpack_id` is an identifier, not an authority claim.  The payload hash
proves bytes were not changed; it does not prove that the bytes belong to the
right causal prefix.  `parent_root` and `dependencies` make the dependency
closure explicit.

## Representation transitions

CF-L0 token/provenance → CF-L1 replay checkpoint → CF-L2 lossless ContextPack or
CF-L3 base+exact-residual → CF-L4 RAM materialization → CF-L5 VRAM
materialization.  A controller may take any safe route, but every exact route
must reconstruct the same declared authority state.  Reconstructing only K/V
is insufficient for a hybrid model if the recurrent state, position metadata,
RNG/sampler state, or runtime/kernel identity is not restored.

## Round-trip evidence

`test_contextpack.py` creates a bounded synthetic payload, validates the record,
serializes/deserializes the record, and checks the payload hash.  This proves
schema/metadata round-trip only; it is not evidence of real-KV compression.
