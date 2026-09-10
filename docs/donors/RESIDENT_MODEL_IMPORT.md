# Resident open-weight models as Remora donors

Status: design formalization plus a measured metadata/header inspection. No
Qwen3.8 tensor values have been loaded into Remora-v0 and no Qwen-derived
module has been promoted.

## Decision

Yes: a resident open-weight model can become a source of replaceable Remora
capability. The safe default is not cross-model weight copying. It is a
frozen donor interface whose outputs are converted into the common language
bus and then distilled into a candidate module under the external experiment
harness.

The import ladder is:

| Level | Mechanism | v0 decision |
| --- | --- | --- |
| 0 | Inspect config, license, index, shard headers, and tensor names | Implemented and safe by default |
| 1 | Query a frozen donor for checked responses or logits | Planned; response distillation is the first real transfer test |
| 2 | Read donor hidden states and learn a `donor-port-v1` adapter into `language-v1` | Adapter implemented; requires an explicit donor runtime |
| 3 | Graft named tensors or an expert | Only for a shape/semantics-compatible model and only with A/B tests |
| 4 | Keep a donor resident as an optional specialist at inference | Long-term; must be budgeted as a serving dependency, not hidden state |

“Useful” is an experimental result, not the donor’s self-description. A
candidate must improve a predeclared target or transfer suite against the
frozen Remora control, retain old-task performance, fit the compute/latency
budget, and pass provenance and licensing checks. A candidate remains
`CANDIDATE` until the harness promotes it.

## What the local Qwen source tells us

The local acquisition is protected and pinned to
`Qwen/Qwen3.8-Flash-Next`, revision
`f5d08274bafd880402bd16f5e3e6c514136ec06c`. Its existing acquisition report
records 131/131 shards, 360,000,192,888 shard bytes, 1,658 tensors, and a
successful index/header reconciliation. The local source manifest reports the
upstream SHA-256 verification and says that no model load or conversion was
performed.

Header inspection finds a multimodal `qwen4_exp` configuration with a 2,560
wide, 48-layer text stack, 24 query heads / 2 KV heads, 512 experts with 10
selected per token, and a 248,320-token vocabulary. The official repository
describes the architecture as Gated DeltaNet plus Qwen Sparse Attention,
four-way gated residual streams, and n-gram embedding tables. It also reports
125B main-model parameters plus 51B n-gram embeddings, with about 6B active per
token. These are precisely the kinds of subsystems worth studying, but they
are not directly compatible with the v0 character tokenizer, 192-wide hidden
state, 4 layers, 128-token vocabulary, or 96-wide language bus.

Therefore the first Qwen experiment must be one of:

1. query a separately launched, explicitly budgeted local runtime and distill
   verified responses into a small Remora specialist; or
2. expose selected hidden states from a compatible serving/runtime path and
   train a `TeacherPortAdapter(teacher_dim -> bus_dim)` while the donor is
   frozen.

The 360-GB BF16 source will not be loaded implicitly on the 16-GB GPU. The
existing quantized artifact is also a serving input, not permission to copy
opaque tensors into Remora. All donor execution must use a separate resource
check, legitimate GPU lock, bounded context/batch, and a reversible output
directory.

## Why direct tensor grafting is usually the wrong first move

Weight names are not semantic interfaces. A transplant can be invalid even
when two tensors have the same shape because of differences in tokenizer
coordinates, RoPE/position conventions, normalization placement, residual
width, expert routing, calibration, tied embeddings, or training scale.

For Qwen3.8 specifically, the local and upstream architecture metadata show
different model families, dimensions, vocabulary, layer count, routing, and
multimodal inputs. Direct graft status must consequently be
`INCOMPATIBLE_FOR_DIRECT_GRAFT`; the manifest recommends frozen-teacher or
activation distillation instead.

A direct graft becomes a candidate only if all of these are true:

- the source license permits the planned derivative and distribution;
- tensor shapes and dtype conversion are explicit and lossless enough for the
  experiment;
- tokenizer and output coordinates are identical or a declared adapter exists;
- positional, normalization, residual, and routing semantics are compatible;
- the module has a declared `inputs` / `outputs` contract on the common bus;
- pre/post behavior is measured on target, old, transfer, and adversarial sets;
- the change is reversible and its ancestry/content hashes are recorded.

## Candidate lifecycle and provenance

Every donor enters the assembly ledger with a separate identity from the
Remora module it may produce:

```text
DONOR_OBSERVED
    -> CANDIDATE_EXTRACTED
    -> FROZEN_EVALUATED
    -> PROMOTED | REJECTED | DORMANT
```

The record must retain:

- local path, upstream repository, pinned revision, and metadata hashes;
- license text/hash and any additional model-card restrictions;
- whether the donor was used as a black-box teacher, hidden-state teacher, or
  tensor source;
- tokenizer/template/runtime versions and prompt/data-set hash;
- selected donor components and the port adapter hash;
- candidate ancestry, changed parameter fraction, and training compute;
- control/candidate/promoted/rejected state and all held-out metrics;
- failures and the surrounding architecture state for resurrection.

`remora.donors.registry.DonorRegistry` stores this state separately from the
neural weights. It permits `DONOR_OBSERVED` and `CANDIDATE_EXTRACTED`, records
`FROZEN_EVALUATED`, and rejects a `PROMOTED` transition unless an external
decision is supplied.

Donor evidence is also kept distinct from Remora’s inherited prior and its
personal experience. “The donor predicts X” is provenance, not ground truth;
the external evaluator and later independent experience decide how much weight
it receives.

## Planned experiments

### D1: metadata and compatibility gate

Run `experiments.donor_inspection` against the local Qwen source. Falsify the
header-only claim if any tensor is materialized or if index/header mismatches
are silently ignored. This experiment is already implemented and is not a
knowledge-transfer result.

### D2: mechanism-only port transfer

Use a small frozen teacher with a different hidden width and a held-out task.
Train only `TeacherPortAdapter` plus one Remora specialist. Compare against a
raw-input control without teacher features. Measure target quality, old-task
retention, adapter parameters, wall time, and transfer to a
new task:

```bash
python -m experiments.donor_port --output results/donor-port.json
```

This validates the port and promotion accounting before paying for a large
donor query. It is deliberately labelled `MECHANISM_ONLY_SYNTHETIC_FROZEN_TEACHER`.

### D3: Qwen response distillation

Use an explicitly launched local Qwen runtime, fixed prompt set, deterministic
sampling, and an external verifier. Distill only accepted responses into a
candidate expert or adapter. Do not compare Qwen logits with Remora logits:
their vocabularies and token positions are not aligned. A failed verifier or
retention gate leaves the candidate dormant with a failure record.

### D4: Qwen representation distillation

If the runtime can expose hidden states, choose a small layer subset, collect
activations for the same text examples, and learn a low-rank projection into
the Remora bus. Test whether the projection transfers beyond the collection
prompts. Store activation statistics and hashes, not the full donor state, by
default.

### D5: compatible tensor surgery

Use a deliberately compatible small donor, not Qwen3.8, to test a named expert
graft. Freeze all unaffected modules, train only the replacement and declared
ports, and compare to full-model adaptation. This is the direct analogue of
the existing Remora module-replacement experiment.

## Prior-art classification

This is a research direction assembled from known mechanisms, not a claim that
Remora invented them:

- Knowledge distillation and strong-to-weak distillation: `RELATED`; they
  support importing behavior into a smaller student.
- FitNets/intermediate hints: `RELATED`; they support hidden-state ports with
  learned projections.
- LoRA/parameter-efficient adaptation: `COMPLEMENTARY`; it supplies a way to
  keep donor-derived updates local.
- Net2Net: `COMPLEMENTARY`; function-preserving surgery applies mainly when the
  source and target computation graphs are explicitly compatible.
- Progressive Neural Networks: `RELATED`; lateral columns motivate retaining
  old specialists while adding new ones, but do not solve cross-architecture
  donor extraction.
- Task arithmetic/model merging: `RELATED but risky`; it assumes meaningful
  alignment in weight space and must not be used as a default cross-family
  import path.

The open question for Remora is not whether distillation or adapters exist. It
is whether a common, versioned bus plus ledgered selection can turn useful
parts of heterogeneous donors into replaceable modules while preserving
lineage, evidence separation, and surgical update economics.

## Research sources

- Qwen3.8-Flash-Next model repository and architecture description:
  <https://github.com/QwenLM/Qwen3.8-Flash-Next>
- Qwen3.8-Flash-Next model card/files:
  <https://huggingface.co/Qwen/Qwen3.8-Flash-Next>
- Safetensors format and metadata/header access:
  <https://huggingface.co/docs/safetensors/index>
- Transformers sharded loading/offload behavior:
  <https://huggingface.co/docs/transformers/models>
- Distilling the Knowledge in a Neural Network:
  <https://arxiv.org/abs/1503.02531>
- FitNets: Hints for Thin Deep Nets:
  <https://arxiv.org/abs/1412.6550>
- Net2Net: Accelerating Learning via Knowledge Transfer:
  <https://arxiv.org/abs/1511.05641>
- LoRA: Low-Rank Adaptation of Large Language Models:
  <https://arxiv.org/abs/2106.09685>
- Progressive Neural Networks:
  <https://arxiv.org/abs/1606.04671>
- Editing Models with Task Arithmetic:
  <https://arxiv.org/abs/2212.04089>
