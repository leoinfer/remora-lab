# local-bench

`local-bench` is a Rust research tool for estimating model-serving
configurations without loading model weights. It reads GGUF header and tensor
metadata, combines that information with a user-supplied hardware phenotype,
and simulates placement, prefill, decode, KV, and speculation choices.

The estimator is not part of the HAR production runtime. Its `inspect`, `sim`,
`search`, and `calibrate` commands are offline. The optional `measure` command
is a research harness that may start a user-supplied external serving binary;
it is never called by HAR and is not required to build or run HAR.

## Quick start

```bash
cargo build --release

# inspect a model header only; no weights are loaded
./target/release/lb inspect <model.gguf>

# simulate with the illustrative profile, or provide your own JSON profile
./target/release/lb sim <model.gguf> example \
    ngl=40 threads=8 batch=2048 ctx=196608 \
    kv=q8_0 fa=on kvoff=off spec=mtp nmax=3

# use a separately reviewed calibration set when one is available
./target/release/lb calibrate <model.gguf> <hardware.json> \
    <anchors.json> <calibration.json>

./target/release/lb search <model.gguf> <hardware.json> \
    --ctx 196608 --cal <calibration.json>
```

Commands:

| command | purpose |
|---|---|
| `lb inspect <model.gguf>` | header-only tensor and attention inventory |
| `lb sim <model> <hw.json> [k=v ...]` | predict prefill/decode and name bottlenecks |
| `lb search <model> <hw.json> [flags]` | search a bounded configuration space |
| `lb calibrate <model> <hw.json> <anchors.json> [out.json]` | fit synthetic constants to reviewed anchors |
| `lb hw` | print an illustrative example profile |

## Model

The simulator tracks memory placement, bandwidth and compute ceilings, KV
geometry, flash-attention constraints, and accepted-token economics for
speculative decoding. It reports assumptions and bottlenecks with every
prediction. Synthetic values are examples only; they are not a performance
claim for HAR or for any particular host.

No model weights, calibration captures, machine telemetry, or raw benchmark
receipts are distributed in this publication. Put reviewed external inputs in
the data directory when running experiments locally.

## Layout

```
crates/lb-model/       GGUF header-only metadata
crates/lb-hardware/    user-supplied hardware phenotype schema
crates/lb-sim/         performance model and calibration
crates/lb-search/      bounded configuration search
crates/lb-cli/         `lb` binary and optional research harness
data/                  intentionally empty of experiment payloads
docs/DESIGN.md         model and evidence rules
```

Licensed under the MIT License in this directory. See the umbrella project’s
provenance and claims documents before treating any research output as an
external result.
