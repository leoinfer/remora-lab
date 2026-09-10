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

**DERIVED:** the original reference implementation was approximately 8.6x
slower than the baseline on this held-out path. The subsequent CUDA scan
optimization reduced the measured forward gap, but this remains a real
engineering constraint for scaling; the custom routing and bus operations
still need profiling and kernel/layout work.

### Multi-seed rerun after the scan optimization

| Aggregate over seeds 7, 19, 31 | Remora-v0 | Monolithic baseline |
| --- | ---: | ---: |
| Mean final held-out loss | **0.291099** | 0.306373 |
| Population loss standard deviation | 0.000609 | 0.002838 |
| Mean training wall time | 48.57 s | 11.00 s |
| Mean held-out tokens/s | 211,505 | 1,015,924 |

**MEASURED:** Remora's final held-out loss was lower on all three paired
seeds. This is stronger than the original single-seed observation, but it is
still only a controlled synthetic stream, not a general capability result.

**DERIVED:** the scan-enabled Remora takes 4.42x the mean training wall time
and runs at 0.208x the baseline held-out throughput. The per-seed artifacts and
post-hoc aggregate are in `results/*multi*.json` and
`results/multiseed-summary.json`.

### Forward profile and optimization

**MEASURED:** on the RX 9060 XT with identical weights and inputs, the
reference Remora path took 18.533 ms per batch, while the associative scan
took 13.009 ms (29.8% faster) with maximum absolute logit difference
1.19e-6. Manual attention was faster than unconfigured SDPA on this driver;
the profile keeps both paths available for runtime-specific testing. Full
measurements are in `results/profile-forward.json`.

**MEASURED:** the first scan optimization was inference-only on this PyTorch
runtime because associative-scan backward rejected lifted trainable inputs.
An explicit affine-scan backward now passes reference-loop gradient parity.
The matched three-seed rerun after that fix reached mean Remora wall time
25.48 s versus 10.84 s for the baseline at the same 400-step budget, while
Remora's final held-out loss remained lower on every pair (0.282343 versus
0.306373). This reduces the training wall-time ratio from 4.42x to 2.35x;
the detailed artifacts are `results/*scanback*.json` and
`results/multiseed-scanback-summary.json`.

**LIMITATION:** this corpus and evaluation do not establish general language,
reasoning, or coding ability. Loss curves are in
`remora-v0-scratch-seed7-loss.svg` and `baseline-v0-scratch-seed7-loss.svg`.

## Real local text/code and continual parity transfer

The next run used local Wikitext-2 raw train/validation/test files plus
Remora's own Python source as the real code stream. The code and math task
splits are deterministic, disjoint, and externally verifiable; personal data
and resident model weights were not read. The benchmark used three paired
seeds (7, 19, 31), 240 scratch-training steps, 120 adaptation steps, batch 16,
sequence 96, and one fixed 1,536-token evaluation window per stream. Exact
task metrics use the first 16 examples of each predeclared split, with a
four-example free-running audit on parity.

| Base result, mean over seeds | Remora-v0 | Monolithic baseline |
| --- | ---: | ---: |
| Final Wikitext-2 validation loss | **2.4682** | 2.5911 |
| Final repository-code validation loss | **2.4184** | 2.6125 |
| Text loss gain / million training tokens | **6.6875** | 6.1925 |
| Code loss gain / million training tokens | **5.9942** | 5.9811 |

**MEASURED:** both scratch-trained models reduced held-out real-stream loss.
Remora was lower on the final text and code windows in this run, and its
loss-gain-per-token was slightly higher on both domains. This is a small local
corpus result, not evidence of broad language or coding competence.

| Continual adaptation arm, mean over seeds | New parity teacher-forced accuracy | New parity free-running audit | Old text loss | Old code loss | Changed fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen Remora control | 43.75% | 50.00% | 2.4682 | 2.4184 | 0.00% |
| Remora local adapter + rehearsal | **58.33%** | 50.00% | **2.9960** | **2.9047** | **2.19%** |
| Remora local adapter, target only | 62.50% | 50.00% | 3.1327 | 3.0537 | 2.19% |
| Frozen baseline control | 43.75% | 50.00% | 2.5911 | 2.6125 | 0.00% |
| Baseline full model + rehearsal | 52.08% | 50.00% | 4.4523 | 4.6225 | 100.00% |
| Baseline full model, target only | 56.25% | 50.00% | 12.5040 | 12.8709 | 99.29% |

**MEASURED CONDITIONAL PASS:** local Remora adaptation cleared the declared
nonzero-parity/update gate and changed 2.19% of the model parameters, while
the matched full-model controls changed approximately all parameters. Rehearsal
substantially reduced—but did not eliminate—old-stream loss. The target-only
local arm learned more on this tiny teacher-forced subset but retained less;
that tradeoff is why the rehearsal arm is the declared comparison.

**MEASURED LIMITATION:** free-running parity was only a four-example audit and
did not separate the arms. Exact code/math task accuracy remained zero at this
training budget. The result supports a surgical-update mechanism under a
controlled gate, not a claim of general reasoning or robust continual
learning. Full JSON, split hashes, per-seed histories, checkpoints, and the
append-only ledger entry are in `results/transfer-benchmark.json` and
`ledger/experiments.jsonl`.

## Strong adversarial lifetime comparison

The first continual-learning comparison used Remora local plasticity against
baseline full-model fine-tuning. `LIFETIME-COMPOUNDING-001+` replaces that
easy adversary with a conventional Transformer using rank-8 LoRA plus the same
old-task rehearsal policy. Both arms were scratch-trained for 120 shared base
steps, then exposed to five sequential tasks (`T1 -> T5`) over three paired
seeds (7, 19, 31). Remora had 1,682,137 parameters and 36,864 trainable
plastic parameters; the baseline had 1,675,008 base parameters, 1,711,872
parameters after LoRA insertion, and the same 36,864 trainable parameters.

The paired Remora-minus-Transformer-LoRA differences were:

| Stage | Primary accuracy | Shifted-interface accuracy | Wall time | Threshold-step difference |
| --- | ---: | ---: | ---: | ---: |
| T1 | +16.67 pp | -2.08 pp | +3.57 s | not reached |
| T2 | -14.58 pp | -27.08 pp | +4.51 s | not reached |
| T3 | 0.00 pp | -64.58 pp | +5.43 s | -20 steps* |
| T4 | 0.00 pp | -12.50 pp | +6.48 s | not reached |
| T5 | +18.75 pp | 0.00 pp | +7.62 s | not reached |

The T3 `-20` entry is a single threshold comparison from the declared tiny
task set; it is retained for auditability and is not treated as significance.

**MEASURED/DERIVED:** the Transformer adapter is a strong matched adversary.
The primary accuracy deltas are noisy and do not grow monotonically with age;
the shifted-interface deltas are mostly negative for Remora, and Remora's
wall-time penalty grows across stages. The one negative threshold-step entry
is not a general learning-velocity win because most fixed thresholds were not
reached and the task set is tiny. The full per-seed values, bootstrap/t
intervals, task matrices, split hashes, and curves are in
`results/lifetime-compounding-analysis-v1.json` and
`results/lifetime-compounding.json`.

**MEASURED NEGATIVE RESULT:** after normalizing by wall time and modeled
training FLOPs, this tranche provides no credible compounding advantage for
Remora. Positive small-task gain-per-token values exist in the receipt, but
they do not survive as a consistent capability advantage across stages or
interfaces. The current evidence therefore does not pass the scaling gate.

The experiment was predeclared to count the following against the thesis:
matched Transformer-LoRA retention or transfer, no increasing later-task
advantage, shifted-template collapse, and compute-normalized regression. All
remain live concerns rather than being hidden by a favorable baseline.

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

### Aged surgery against matched LoRA

`AGED-SURGERY-001+` repeats the meaningful replacement after five lifetime
stages. It replaces the used layer-1 expert with a 111,456-parameter SwiGLU
module and trains only that module. The adversary keeps the aged Transformer
attention LoRA frozen, inserts rank-64 LoRA factors at the corresponding
feed-forward block, and trains 110,592 parameters. Both arms use 60 steps,
the same target examples, the same task rehearsal, and the same base text/code
replay: 445,440 assimilation tokens.

**MEASURED NEGATIVE RESULT:** Remora's mean target-valid difference was
`+0.1875`, but its shifted-interface difference was `-0.1667`, unseen-task
difference was `-0.0625`, and old-task retention-ratio difference was
`-0.0688`. The Remora surgery was 6.24–8.62 seconds slower per seed. One seed
favored the local replacement on retention, two favored the Transformer;
there is no aged-surgery superiority claim. Full task matrices and changed
parameter receipts are in `results/aged-surgery-v1.json`; surgery checkpoints
remain local and are ignored by Git.

### Bus ablation and real resurrection

The shared learned bus has 42,049 parameters. In the scratch bus ablation, a
parameter-free direct channel-slice/zero-pad coupling matched the shared bus
at 0.0 paired primary/shifted accuracy difference on the first four stages;
shared-bus unseen-interface deltas were -0.1042, -0.0833, -0.0417, and
-0.1250. This does not establish that the bus adds abstraction. On the aged
path, removing it while retaining the other weights raised base validation
loss from text/code `2.9881/2.9633` to `3.4908/3.3862`, so the learned bus is
used by the current representation even though its transfer value is
unproven. The result and small comparison plots are in
`results/bus-ablation-v1.json` and `results/bus-ablation-v1-*.svg`.

The real resurrection run took the immutable target-only expert-replacement
failure, where old-task retention failed, and re-tested the same candidate
after model age changed from 0 to 5 and the policy changed to target plus old
rehearsal. Its queue priority rose from `0.5100` to `0.8925`; the actual trial
improved the target and passed the fixed old-loss gate. **MEASURED
CONDITIONAL PASS:** this is a context-dependent resurrection, not proof that
the candidate is universally good. Both the original failure and successful
retest remain in the ledger and `results/resurrection-real-*.json`.

The experienced Ship-of-Theseus run then replaced all four experts after the
five-stage checkpoint. Local update fractions were 6.482%, 6.345%, 6.213%,
and 6.087%; old loss moved from 3.4084 to 2.7192 and target loss from 3.4103
to 2.1184 under the declared rehearsal. The four-generation lineage is in
`results/ship-of-theseus-aged-v1.json`. This is a successful bounded toy chain,
not evidence that arbitrary capability can survive indefinite replacement.

## Manual and resurrection

**MEASURED:** the factual assembly graph answered dependency, ancestry, and
minimum-affected-neighborhood queries. The learned reader predicted all graph
dependents for `language-v1` with no unsupported dependency; the graph remains
the authority and the reader cannot mutate it.

**MEASURED:** `MANUAL-REAL-CHANGES-001+` replayed the actual aged
Ship-of-Theseus graph and bound the donor, consolidation, aged-surgery, and
resurrection receipts to it. Eight questions passed mechanically, including
which parameters changed at T3, all 24 consolidated experience IDs and their
independence clusters, the active dependents of `language-v1`, the affected
neighborhood of `expert-layer1-g2`, the aged-surgery regressions, and the
Qwen-graft ancestry/status. Candidate branches remained `CANDIDATE` or
`DORMANT`; none was promoted. This is a factual real-change manual receipt,
not a claim that a language-model manual has learned the whole graph. The
receipt is `results/manual-real-changes-v1.json`.

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

**MEASURED:** the guarded `donor-activation-v1` bundle round-tripped 16
synthetic hidden-state records, loaded only the 12 accepted records under a
2,304-byte budget, and produced zero activation/port drift. This validates the
on-disk interchange and byte gate, not donor semantics.

**MEASURED CONDITIONAL PASS:** `DONOR-RESPONSE-002` distilled a synthetic
eight-key lookup donor through the plastic islands. Same-interface held-out
accuracy was 100% for the adapter-rehearsal arm, with 36,864/1,682,137
parameters changed (2.19%) and old loss 0.2795 -> 0.4343. The target-only
adapter also reached 100% but old loss rose to 2.3350. The full-model rehearsal
control reached 100% with 99.72% of parameters changed and old loss 1.0982.
The shifted-template test was 0% for every trained arm, exposing protocol
brittleness.

**MEASURED FAILURE:** `DONOR-RESPONSE-001` used sparse random arithmetic
responses. Adapter-only accuracy was 0%; the full-model control reached only
4.6875% on unseen pairs and damaged the old stream. A small adapter cannot be
treated as a general reasoning distiller without a better-conditioned port,
more data, or a compatible response curriculum.

**MEASURED RUNTIME PROBE:** the explicit local Transformers boundary loaded the
resident Nanbeige 3B artifact at approximately 8.4 GB VRAM and generated eight
bounded greedy responses using the compatibility path. Its custom generation
helper was incompatible with the installed Transformers 5 cache API, so the
client used manual `use_cache=False` decoding. Zero of eight strict integer
answers passed the external verifier; no response was eligible for
distillation. Qwen3.8 remains un-loaded because its BF16 source is about 360
GB and is not a safe implicit load on this machine.

**MEASURED ACTIVATION PROBE:** the same explicit runtime hooked only
`model.layers.0`, recorded that Nanbeige invoked it twice, and selected the
final invocation for each of four prompts. Four BF16 width-3,072 vectors
(24,576 bytes) round-tripped through `donor-activation-v1`; the explicit port
cast produced finite `[4, 1, 96]` bus packets. No donor graph was imported and
no candidate was promoted. This validates surgical observation, not transfer
utility.

**MEASURED CHAT-TEMPLATE PROBE:** the resident runtime now applies the
donor tokenizer's own chat template when requested and records its template
hash. On Nanbeige the template hash was
`4819d36ae9e1491c0f323a3767fa1f86b34070b2849c720381e0657dd10ab21b`; the
bounded strict-integer probe still passed 0/8 responses. A chat-template
activation capture produced four finite layer-0 records and the same 24,576
payload bytes.

**MEASURED ACTIVATION-TRANSFER FAILURE:** `DONOR-ACTIVATION-TRANSFER-001`
captured 96 BF16 final-token records (589,824 payload bytes), trained only a
`TeacherPortAdapter(3072 -> 96)` plus a classifier, and used externally
computed parity labels. The candidate reached 53.125% on disjoint validation
pairs versus 50.0% for a prompt-byte control, but reached only 43.75% on the
shifted interface versus 56.25% for that control. The predeclared utility gate
failed; no donor-derived module was promoted. The port itself changed
304,707/309,412 parameters (98.48%), so the experiment does not yet support a
cheap-import claim. Records, bundle, result, and failure history are retained
in `results/resident-activation-transfer-*`.

### Direct Qwen neural-organ pilot

The donor addendum's primary objective was tested separately from response or
activation distillation. `QWEN-NEURAL-ORGAN-001+` built a header-derived
anatomy map for the pinned 131-shard Qwen source, then selected one finite
component: the layer-0 shared expert. Only four actual BF16 tensors were read
from `model-00003-of-00131.safetensors`: the shared expert's `gate_proj`,
`up_proj`, `down_proj`, and scalar shared gate. The selected payload was
9,835,520 bytes / 4,917,760 trained parameters; no full model loader ran.

**MEASURED:** the standalone organ exactly reproduced its explicit gate/up,
SiLU, down, and scalar-gate computation against an independent reference:
zero max absolute, mean, and relative-L2 error, cosine similarity 1.0, and
full top-k agreement. This proves function reproduction for the isolated
organ, not for the full Qwen block.

**MEASURED:** the Remora attachment was a wrapped frozen graft. All 4,917,760
donor parameters were preserved unchanged and frozen; 0 were analytically
transformed and 0 discarded. BF16-to-FP32 was a runtime storage conversion,
not learned reconstruction. Rank-8 input/output ports added 42,496 newly
trained parameters. Across seeds 7, 19, and 31, repair used 60 gradient steps,
184,320 total assimilation tokens (92,160 target + 92,160 rehearsal), and
18.70–19.62 seconds. The modeled trainable-update cost was 46.997 billion
FLOPs and modeled forward cost including the donor core was 7.264 trillion
FLOPs. Original Qwen training compute is **UNMEASURED**, so original compute
avoided / assimilation compute spent is **NOT_COMPUTABLE** rather than
invented.

The donor core was causally relevant with fixed ports: zero/random/shuffled
core interventions changed target loss by about 0.423 on average. But the
same-budget retrained controls were close (random minus actual target loss
`0.0360`, shuffled minus actual `0.0147`), and native Remora specialists were
stronger in this v0 task. The graft is therefore a genuine
`WRAPPED_GRAFT`/`FUNCTION_REPRODUCED` candidate, not a successful capability
assimilation claim and not promoted. The full parameter accounting,
functional-equivalence receipt, controls, hashes, and lineage are in
`results/qwen-neural-graft-analysis-v1.json`,
`results/qwen-neural-graft-v1.json`, and
`results/manual-real-changes-v1.json`.

### Current performance gate

The post-adversarial profile confirms the runtime cost remains material. On
the same RX 9060 XT batch, current Remora reference/scan/scan+SDPA paths took
13.22/13.62/13.57 ms, while the manual-attention baseline took 3.08 ms. The
current scan remained numerically equivalent (maximum logit delta
`1.19e-6`) but was not faster in this rerun. This is an engineering negative
result and blocks scaling until routing, expert batching, and small-kernel
fragmentation are improved. Receipt:
`results/profile-forward-post-adversarial-v1.json`.

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
# Strong matched lifetime adversary (three seeds; requires the local Wikitext-2 files)
flock -n /tmp/remora-v0-gpu.lock \
  /home/leo/.venvs/remora-rocm10/bin/python -m experiments.lifetime_compounding \
  --device cuda --seeds 7 19 31
/home/leo/.venvs/remora-rocm10/bin/python -m experiments.analyze_lifetime_compounding
# Real-change manual receipt and aged-model lineage
/home/leo/.venvs/remora-rocm10/bin/python -m experiments.manual_real_changes
flock -n /tmp/remora-v0-gpu.lock \
  /home/leo/.venvs/remora-rocm10/bin/python -m experiments.aged_surgery \
  --device cuda --seeds 7 19 31 --steps 60
flock -n /tmp/remora-v0-gpu.lock \
  /home/leo/.venvs/remora-rocm10/bin/python -m experiments.ship_of_theseus \
  --checkpoint checkpoints/lifetime-compounding-remora_local_rehearsal-seed7.pt \
  --seed 7 --steps-per-generation 20 --device cuda \
  --output results/ship-of-theseus-aged-v1.json
# Donor-organ pilot and explicit assimilation accounting
/home/leo/.venvs/remora-rocm10/bin/python -m experiments.donor_organ
flock -n /tmp/remora-v0-gpu.lock \
  /home/leo/.venvs/remora-rocm10/bin/python -m experiments.donor_graft \
  --checkpoint-dir checkpoints --payload results/qwen-neural-organ-layer0.safetensors \
  --device cuda --seeds 7 19 31 --steps 60
/home/leo/.venvs/remora-rocm10/bin/python -m experiments.analyze_donor_graft
# Current runtime profile; compare with the preserved profile-forward.json
flock -n /tmp/remora-v0-gpu.lock \
  /home/leo/.venvs/remora-rocm10/bin/python -m experiments.profile_forward \
  --device cuda --output results/profile-forward-post-adversarial-v1.json
```

The append-only experiment ledger is `ledger/experiments.jsonl`; failures are
in `ledger/failures.jsonl`. Checkpoints are intentionally ignored by Git and
remain local because they are generated artifacts.

## Next scaling path chosen from evidence

1. Keep the model at v0 size. The strong Transformer-LoRA adversary did not
   reveal a compounding Remora advantage, so the next step is architecture
   redesign/measurement, not scale-up.
2. Reduce runtime overhead: batch experts, remove avoidable per-branch kernel
   launches, and profile bus/routing layouts before repeating the central
   lifetime comparison.
3. Improve shifted-interface abstraction and repeat the multi-lifetime run
   with larger held-out task sets and more seeds; preserve the current failure
   gates.
4. For donor work, prioritize lower-repair/function-preserving conversions
   and a donor task where the source is independently strong. Do not claim
   compute avoidance until the donor training bill is measured or sourced.
5. Keep the Qwen organ as a candidate branch. Test subspace/low-rank extraction
   and stateful organs only after a bounded organ beats native and fresh
   controls at comparable assimilation cost.
