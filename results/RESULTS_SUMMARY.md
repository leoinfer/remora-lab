# Remora-v0 measured results

Snapshot date: 2026-09-10. This is the first reproducible checkpoint, not a
claim of frontier capability. Results below distinguish measured observations
from derived comparisons, modeled mechanisms, and hypotheses.

## Execution context

The scratch runs used Python 3.14.7, PyTorch 2.13.0+rocm10.0.0, HIP
7.15.26333, an AMD Radeon RX 9060 XT, host `cachyos-x8664`, seed 7, and the
local synthetic character/code/math stream. Both models started from random
weights and used the same 400-step, batch-32, sequence-96 training budget.
The machine identity and resource state were checked through the shared
research-machine bridge before the GPU runs. No resident large model was
loaded.

## Scratch learning and matched baseline

| Result | Remora-v0 | Monolithic baseline |
| --- | ---: | ---: |
| Parameters | 1,682,137 | 1,675,008 |
| Final held-out loss | **0.281514** | 0.306059 |
| Final held-out perplexity | **1.3251** | 1.3581 |
| Held-out tokens/s | 121,557 | 1,042,263 |
| Total training wall time | 58.39 s | 11.54 s |

**MEASURED:** both models learned from scratch and passed the small overfit
test. On this one synthetic seed, Remora finished with lower held-out loss.

**DERIVED:** the current Remora implementation is approximately 8.6x slower
than the baseline on this held-out path. This is a real engineering failure
for scaling, not evidence against the modular thesis; the custom recurrent,
routing, and bus operations need profiling and kernel/layout work.

**LIMITATION:** this corpus and evaluation do not establish general language,
reasoning, or coding ability. Loss curves are in
`remora-v0-scratch-seed7-loss.svg` and `baseline-v0-scratch-seed7-loss.svg`.

## Lifetime evidence and three timescales

The controlled world encodes inherited rule Y as the default and experience
rule X when hidden condition `z=1`.

**MEASURED:** the inherited subnetwork was trained from random initialization
(loss 0.632383 -> 0.00000854). A naive local experience-adapter update learned
the target (`z=1 -> X`) but lost the tested old case (`z=0 -> Y`). Adding old-
stream rehearsal while updating only the adapter retained both cases. The
repair changed 194 of 582 world-model parameters (33.33%); the inherited path
was unchanged.

**MEASURED:** after copying the adapter into the slow consolidated path and
disabling retrieval, both the old and conditioned target accuracies were 1.0.
The archive retained 13 provenance records, including 12 experience episodes.

**MEASURED:** inherited and experienced evidence remain queryable separately:

- inherited-only `P(X | z=1)` = 0.3100;
- experienced-only `P(X | z=1)` = 0.7109;
- combined = 0.5250.

**MEASURED:** with 25 experience observations, duplicate evidence from one
program produced 2 effective clusters and `P(X)=0.5250`; four correlated
clusters produced 5 effective clusters and `P(X)=0.9427`; 25 independent runs
produced 25 effective clusters and `P(X)=0.9999999991`. The mechanism is
bounded cluster support, not a complete Bayesian treatment.

The first naive update is retained as `LIFETIME-EVIDENCE-001`/the result
failure history; the repaired arm is `LIFETIME-EVIDENCE-002`.

## Surprise, specialization, and credit assignment

**MEASURED mechanism test:** low surprise allocated 1 compute unit; an
isolated high surprise and 24 duplicate high-surprise observations each
allocated 4; 24 independent high-surprise clusters allocated 8. This is a
toy policy demonstrating cluster-aware allocation, not calibrated uncertainty.

**MEASURED/DERIVED:** the trained router's mean absolute task-selectivity shift
was 0.022379 versus 0.007001 for the fresh-router control, a 3.20x ratio. The
absolute effect is small, so this requires multi-seed and larger-task reruns.

**MEASURED:** expert ablations produced non-uniform target-loss deltas, and a
two-expert intervention had a nonzero interaction delta (-0.00620). Credit is
assigned by intervention effects rather than trusting self-attribution.

## Surgical consolidation and module replacement

**MEASURED FAILURE:** replacing a used expert and training only the new expert
on the target stream reduced target loss 5.0875 -> 0.8806 but raised old-task
loss 0.2870 -> 2.0864. This is retained as a negative result.

**MEASURED CONDITIONAL PASS:** adding old-stream rehearsal to the same local
replacement reduced target loss 5.1600 -> 0.4671 while old-task loss changed
0.2818 -> 0.3577. The replacement added 111,456 new-expert parameters and
updated only the declared expert scope, 6.482% of the 1,719,385-parameter
post-surgery model. The monolithic full-model control reached target loss
0.2869 but destroyed the old stream (loss 0.3064 -> 6.4532) and changed 100%
of its parameters.

**MEASURED:** the Ship-of-Theseus run performed four sequential expert
replacements. Local update fractions were 6.482%, 6.345%, 6.213%, and 6.087%.
The final fixed-old-stream loss was 0.3389 versus the initial 0.2818, below
the declared 2x retention gate, and final target loss was 0.3527. Every
original expert in the four-layer toy stack has a recorded replacement and
ancestry edge. This demonstrates a bounded toy surgery path, not indefinite
real-world capability preservation.

## Manual and resurrection

**MEASURED:** the factual assembly graph answered dependency, ancestry, and
minimum-affected-neighborhood queries. The learned reader predicted all graph
dependents for `language-v1` with no unsupported dependency; the graph remains
the authority and the reader cannot mutate it.

**MODELED/SYNTHETIC:** a candidate with old-state quality 0.41 versus a safe
baseline at 0.58 rose from queue priority 0.608 to 1.013 after a surrounding
state change. In the toy changed context it scored 0.76 versus 0.61. This
validates queue mechanics only; it is not a neural resurrection benchmark.

## Resident open-weight donor track

The local Qwen3.8-Flash-Next BF16 source is pinned and protected. The v0
inspection read JSON metadata, the index, and safetensors headers only:

- 131 shards, 360,000,192,888 bytes, 1,658 indexed tensors;
- index/header reconciliation succeeded;
- `weights_materialized=false`, `model_loader_called=false`,
  `get_tensor_called=false`, and no payload hashes were attempted;
- direct graft was rejected because the donor is a different model family,
  hidden width, layer count, head count, and vocabulary.

**MEASURED MECHANISM CONTROL:** a frozen synthetic teacher plus a trainable
`donor-port-v1` adapter reached 95.9635% held-out accuracy versus 92.8385% for
the raw-input control. This validates the port accounting, not Qwen transfer.

**MEASURED MECHANISM CONTROL:** the value-free selector groups header names
into complete architectural units and obeys a fixed 256 MiB budget. It chose:

- `model.language_model.layers.0.linear_attn` — 115,917,248 bytes;
- `model.language_model.layers.1.linear_attn` — 115,917,248 bytes;
- `model.language_model.hyper_connection_mixer` — 13,127,680 bytes;
- `model.language_model.layers.0.attn_hyper_connection` — 13,209,600 bytes.

Total selected payload: 258,171,776 bytes. Nothing was copied or promoted;
the registry state is `DONOR_OBSERVED -> CANDIDATE_EXTRACTED`, with
`frozen_teacher_or_activation_distillation` as the import mode.

**MEASURED MECHANISM CONTROL:** after a fresh resource check, the explicit
payload path materialized exactly those 25 selected tensors from only two
shards into `/tmp/remora-v0-qwen3.8-selected.safetensors`. Expected and
observed payload bytes both equaled 258,171,776; the standalone file is
258,175,320 bytes and has SHA-256
`b20a37abc72f7285e565821278012095d48aa66da0e11f636acc9cb4d707d141`. The
receipt is `donor-extraction.json`. No full Qwen model loader ran, and the
candidate remains unpromoted. This proves bounded selective reading, not that
the extracted Qwen mechanisms transfer to Remora.

**NEGATIVE INSTRUMENTATION RESULT:** the first selector implementation grouped
parameter leaves. A regression test caught it, the component-path grouping was
corrected, and the corrected run is recorded as `DONOR-SELECTION-003`.

No Qwen response or hidden-state execution/distillation has been run yet. That
requires an explicitly launched, resource-budgeted runtime and a fixed external
evaluator; only the bounded selected tensor payload was read in this phase.

## Reproduction

From the repository root:

```bash
/home/leo/.venvs/remora-rocm10/bin/python -m pytest -q
/home/leo/.venvs/remora-rocm10/bin/python -m train.train --model remora --steps 400 --run-id remora-v0-scratch-seed7
/home/leo/.venvs/remora-rocm10/bin/python -m train.train --model baseline --steps 400 --run-id baseline-v0-scratch-seed7
/home/leo/.venvs/remora-rocm10/bin/python -m experiments.run_experiments --quick
/home/leo/.venvs/remora-rocm10/bin/python -m experiments.donor_inspection \
  --path /home/leo/models/Qwen3.8-Flash-Next-BF16-source \
  --output results/donor-inspection.json
/home/leo/.venvs/remora-rocm10/bin/python -m experiments.donor_selection \
  --manifest results/donor-inspection.json \
  --output results/donor-selection.json
```

The append-only experiment ledger is `ledger/experiments.jsonl`; failures are
in `ledger/failures.jsonl`. Checkpoints are intentionally ignored by Git and
remain local because they are generated artifacts.

## Next scaling path chosen from evidence

1. Profile and optimize the Remora forward path before increasing model size;
   rerun the matched scratch comparison over at least three seeds.
2. Add a small real text/code/math corpus and held-out transfer tasks while
   preserving the synthetic causal tests.
3. Add a guarded donor-runtime client that can request only selected Qwen
   responses or activations, records prompt/runtime hashes, uses a separate
   GPU lock, and cannot promote candidates.
4. Run response distillation, then hidden-state port transfer, against fixed
   controls and an external verifier; measure latency, retention, transfer,
   and changed-parameter fraction.
5. Run direct tensor surgery only with a deliberately compatible small donor;
   keep Qwen as a frozen teacher/representation source until compatibility is
   demonstrated.
