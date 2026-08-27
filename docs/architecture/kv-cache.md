# KV cache and effective context

The cache is treated as a typed, identity-bound page graph. A page can be
stored in an exact representation, a compressed representation, or a
reconstructible cold tier. A restore requires matching prefix identity,
token/position range, layer range, formats, epoch, and generation.

R4KV is one experimental representation family. ContextFold is the policy and
orchestration vocabulary around representations; it does not turn an
unverified compression result into an exact cache. Quality and recovery
contracts remain explicit at admission.

“Effective context” means the amount of useful information retained under
those storage, quality, retrieval, and recovery constraints. It must not be
read as a dense-attention position count.
