# Ephemeral safetensor donor lifecycle

Remora’s resident-model donor is now handled in two distinct passes:

1. Every donor shard is scanned sequentially through its safetensors header.
   This records tensor names, shapes, dtypes, byte regions, and header hashes
   without materializing payload values or constructing a model.
2. A selected organ requests exact tensor names and optional contiguous slices.
   Only the requested pieces from one source shard at a time are materialized
   into a temporary staging `.safetensors` bundle. A verifier consumes it. The
   staging bundle is deleted only if the verifier returns `accepted: true`.

The original donor tree is immutable. The lifecycle refuses a staging path
inside the donor tree and refuses deletion outside `/tmp` or `/var/tmp`.
Existing historical extraction bundles are not retroactively deleted.

## Durable context after deletion

The receipt and append-only lifecycle log preserve enough information to
reconstruct or connect the part later:

- donor repository, pinned revision, source root, metadata/index hashes;
- source shard name, byte size, full shard SHA-256 for selected shards;
- safetensors header length/hash and payload data-region start;
- each parent tensor’s shape, dtype, payload size, relative offsets, absolute
  file offsets, and raw parent-payload SHA-256;
- exact slice coordinates and the decoded materialized tensor SHA-256;
- logical role, operation, reassembly group/order/axis, and dependencies;
- input/output/state contracts, normalization and residual assumptions;
- the verifier ID, acceptance reason/metrics, staging hash, deletion time, and
  explicit proof that the source was not mutated.

For a later recovery, re-read the pinned source shard, verify its file hash,
locate the parent tensor by name and offsets, apply the recorded slice, then
follow the recorded reassembly steps. The deleted staging file is not treated
as the source of truth.

## Real run

The first real lifecycle uses the selected layer-17/value-head-10 Qwen GDN
core. Ten pieces are read from `model-00051-of-00131.safetensors`:

```text
in_proj_qkv rows for q/k/v
in_proj_a and in_proj_b rows for head 10
conv1d rows for q/k/v
A_log and dt_bias rows for head 10
```

The convolution source is `[channels, 1, kernel]`; the receipt records the
required `squeeze(axis=1)` before concatenating q/k/v into `[384, kernel]`.
The verifier reloads the temporary bundle, executes the core, and checks
finite output plus full-sequence versus chunked recurrent-state equivalence.

Reproduce the accepted run:

```bash
python -m experiments.donor_ephemeral_stream \
  --source /home/leo/models/Qwen3.8-Flash-Next-BF16-source \
  --output results/qwen-safetensor-ephemeral-gdn-v4.json \
  --event-log ledger/donor-safetensor-lifecycle.jsonl \
  --staging-dir /tmp/remora-donor-safetensor-stream/qwen-gdn-core-004 \
  --stream-id qwen-gdn-core-ephemeral-004 \
  --delete-after-accept \
  --skip-header-sweep
```

Run the full header pass by omitting `--skip-header-sweep`. The committed
header result is `results/qwen-safetensor-shard-sweep-v1.json`.

The accepted run deleted one temporary staging bundle and deleted zero source
files. Earlier verifier failures remain as retained temporary diagnostic
bundles and separate result history; they were not silently treated as
accepted or cleaned up.

