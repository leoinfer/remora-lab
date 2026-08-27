# local-bench design

## Purpose

Predict a serving configuration from model metadata, a hardware phenotype, and
an explicit workload. The tool does not load weights for estimation. It is a
research component under `benchmarks/`, not an inference dependency of HAR.

## Guarantees

1. `lb-model` reads the GGUF header, metadata, and tensor directory only.
2. Tensor byte accounting comes from the format’s block geometry and is
   reported per quantization class.
3. Placement, KV sizing, and working-set checks are explicit in the result.
4. Every prediction names the limiting resource and records assumptions.
5. Calibration is a separate step over user-supplied, reviewed anchors. No
   bundled receipt is treated as a public performance proof.

The optional `measure` command is deliberately outside HAR’s production
boundary. It can launch a user-selected serving process and query its local
HTTP endpoint so a researcher can create new anchors. That path is not used by
`har-server`, is not needed for a HAR build, and must be audited separately.

## Prediction model

For each accepted token, the estimator compares bandwidth and compute ceilings
for the host and accelerator portions of the configured placement. Speculative
decoding applies an explicit accepted-token expectation and draft cost. Prefill
uses a calibrated batch curve. Invalid memory, KV, attention, or speculation
combinations are reported as non-fitting instead of receiving a fabricated
throughput value.

The implementation carries research hypotheses about KV geometry, graph
overhead, residency, and speculation. Hypotheses remain labeled as such until
an independent, reproducible experiment supports them.

## External inputs

Hardware profiles and calibration anchors are intentionally not bundled. A
future experiment may add them only after removing local paths, model payloads,
credentials, raw receipts, and unverified claims, and after recording their
license and provenance in the umbrella project.
