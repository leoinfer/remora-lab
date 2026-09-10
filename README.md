# Remora-v0

Remora-v0 is a small, scratch-trained research instrument for testing whether
modular neural systems can accumulate experience and change locally without
discarding the whole model.

This repository is intentionally an experiment harness, not a claim that the
long-term Remora architecture is solved. The first implementation contains:

- an autoregressive character language model;
- a shared versioned latent language bus;
- a causal attention path, a recurrent gated-delta state path, and replaceable
  routed experts;
- a low-rank fast-plastic adapter for surgical updates;
- an explicit inherited/experienced evidence store with cluster-aware
  independence accounting;
- a trainable structured world-model prototype;
- factual assembly lineage plus a trainable ledger reader;
- controlled module replacement and resurrection-queue experiments.

The matched baseline is a conventional pre-norm causal Transformer using the
same tokenizer, streams, optimizer family, seed policy, and approximately the
same parameter count.

## Quick start

The commands below use the local ROCm environment discovered during the v0
audit. CPU is also supported.

```bash
cd /home/leo/research/remora-v0
/home/leo/.venvs/remora-rocm10/bin/python -m unittest discover -s tests -v
/home/leo/.venvs/remora-rocm10/bin/python -m train.train --model remora --steps 80 --run-id smoke-remora
/home/leo/.venvs/remora-rocm10/bin/python -m train.train --model baseline --steps 80 --run-id smoke-baseline
/home/leo/.venvs/remora-rocm10/bin/python -m experiments.run_experiments --quick
/home/leo/.venvs/remora-rocm10/bin/python -m experiments.donor_port
/home/leo/.venvs/remora-rocm10/bin/python -m experiments.specialization \
  --checkpoint checkpoints/remora-v0-scratch-seed7.pt
/home/leo/.venvs/remora-rocm10/bin/python -m experiments.credit_assignment \
  --checkpoint checkpoints/remora-v0-scratch-seed7.pt
/home/leo/.venvs/remora-rocm10/bin/python -m experiments.ship_of_theseus \
  --checkpoint checkpoints/remora-v0-scratch-seed7.pt
/home/leo/.venvs/remora-rocm10/bin/python -m experiments.donor_selection \
  --manifest results/donor-inspection.json
# Explicit payload extraction is opt-in and should write to /tmp, never the
# donor tree; this reads only the selected tensor names.
/home/leo/.venvs/remora-rocm10/bin/python -m experiments.donor_extract \
  --manifest results/donor-inspection.json \
  --selection results/donor-selection.json \
  --output /tmp/remora-v0-donor-candidate.safetensors \
  --allow-payload
/home/leo/.venvs/remora-rocm10/bin/python -m experiments.donor_response \
  --checkpoint checkpoints/remora-v0-scratch-seed7.pt \
  --device auto
/home/leo/.venvs/remora-rocm10/bin/python -m experiments.donor_response_lookup \
  --checkpoint checkpoints/remora-v0-scratch-seed7.pt \
  --device auto
/home/leo/.venvs/remora-rocm10/bin/python -m experiments.donor_activation
/home/leo/.venvs/remora-rocm10/bin/python -m experiments.profile_forward \
  --device auto
/home/leo/.venvs/remora-rocm10/bin/python -m experiments.aggregate_multiseed \
  --glob 'results/*multi*.json'
# Matched local Wikitext-2/repository-code training plus held-out verified
# parity adaptation; use a fresh resource check and legitimate GPU lock.
flock -n /tmp/remora-v0-gpu.lock \
  /home/leo/.venvs/remora-rocm10/bin/python -m experiments.transfer_benchmark \
  --wiki-root /home/leo/tmp/wikitext-2-raw \
  --device cuda --seeds 7 19 31 --steps 240 --adaptation-steps 120 \
  --batch-size 16 --seq-len 96 --max-eval-tokens 1536 --task-eval-count 16
```

For a longer bounded run, use `scripts/run_v0.sh`. It records the exact
command, seed, configuration, git revision, hardware context, metrics, and
checkpoint metadata under `results/`, `ledger/`, and `checkpoints/`.

## Research status

The authoritative experiment record is the append-only JSONL ledger at
`ledger/experiments.jsonl`. Claims in reports are labelled `MEASURED`,
`DERIVED`, `ESTIMATED`, `MODELED`, `HYPOTHESIS`, or `EXTERNAL`. Failed trials
are retained with the conditions under which resurrection could be sensible.
The first measured snapshot is `results/RESULTS_SUMMARY.md`.

The initial falsification criteria and the v0 architecture contract are in
`docs/architecture/V0_HYPOTHESIS.md` and
`docs/architecture/V0_CONTRACT.md`.
The working prior-art classifications are in `docs/prior_art/README.md`.

The existing GitHub-backed symbolic/control lab and its boundary are described
in `docs/PRIOR_LAB.md`; it is evidence context, not a source of pretrained
weights for this neural run.

Large temporary files can be redirected with `--output-root /tmp/remora-v0` on
the audited machine. No pretrained weights are used by the training scripts.

## Resident donor models

Remora has an explicit donor track, but a downloaded model is not silently
loaded or treated as a bag of interchangeable tensors. Run the read-only
manifest pass first:

```bash
/home/leo/.venvs/remora-rocm10/bin/python -m experiments.donor_inspection \
  --path /home/leo/models/Qwen3.8-Flash-Next-BF16-source \
  --output results/donor-inspection.json
```

This reads JSON metadata and safetensors headers only. It records tensor names,
shapes, dtypes, shard accounting, provenance, license state, and compatibility
with the v0 bus without materializing model weights. The current donor design
and the import ladder are documented in
`docs/donors/RESIDENT_MODEL_IMPORT.md`. Incompatible models such as the
360-GB Qwen3.8-Flash-Next source are expected to enter Remora first as frozen
teachers or through learned representation/response ports. Direct tensor grafts
are allowed only after explicit shape, tokenizer, positional, normalization,
license, and held-out compatibility checks.

An explicitly selected smaller resident model can be queried through the
runtime boundary, but it requires a separate model-load acknowledgment and a
resource-bounded GPU lock. For example, the local Nanbeige 3B artifact can be
probed with:

```bash
flock -n /tmp/remora-v0-gpu.lock \
  /home/leo/.venvs/remora-rocm10/bin/python -m experiments.resident_donor_query \
  --model-path /home/leo/models/nanbeige4.2-3b \
  --runtime-id nanbeige4.2-3b-local \
  --device cuda --max-new-tokens 8 --trust-remote-code --allow-model-load
# Chat-template-aware variant; raw prompting remains the default.
flock -n /tmp/remora-v0-gpu.lock \
  /home/leo/.venvs/remora-rocm10/bin/python -m experiments.resident_donor_query \
  --model-path /home/leo/models/nanbeige4.2-3b \
  --runtime-id nanbeige4.2-chat-template-local \
  --device cuda --max-new-tokens 8 --trust-remote-code \
  --chat-template --allow-model-load
flock -n /tmp/remora-v0-gpu.lock \
  /home/leo/.venvs/remora-rocm10/bin/python -m experiments.resident_donor_activation \
  --model-path /home/leo/models/nanbeige4.2-3b \
  --runtime-id nanbeige4.2-3b-local-activation \
  --layer-name model.layers.0 \
  --device cuda --trust-remote-code --allow-model-load
# Candidate utility test: donor labels are never trusted; the external parity
# oracle supplies labels and promotion remains false.
flock -n /tmp/remora-v0-gpu.lock \
  /home/leo/.venvs/remora-rocm10/bin/python -m experiments.resident_activation_transfer \
  --model-path /home/leo/models/nanbeige4.2-3b \
  --runtime-id nanbeige4.2-chat-template-transfer \
  --layer-name model.layers.0 --device cuda --trust-remote-code \
  --chat-template --allow-model-load
```

This path uses local-only files, manual bounded greedy decoding, external
verifier-gated response records, layer-scoped activation hooks, and never
promotes a Remora candidate. `--chat-template` applies the donor tokenizer's
versioned conversation template and records its SHA-256 in every response or
activation record. The Qwen3.8 source remains inspection/selection-only until a
runtime can expose a safe, budgeted query or activation hook; the new
activation-transfer experiment is the first real resident-donor utility gate.
