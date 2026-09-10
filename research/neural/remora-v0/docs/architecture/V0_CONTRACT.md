# Remora-v0 architecture contract

## Forward graph

```text
tokens -> embedding + position -> [RemoraBlock x L] -> norm -> vocabulary
                                      |
             +------------------------+-------------------------+
             | shared language bus (dense latent + confidence)   |
             |       /                 |                 \        |
             | causal attention   gated-delta state   routed experts |
             |                                               |     |
             +------------------------ fast-plastic adapter -+     |
                                      |
                                  residual stream
```

The bus is a versioned module with fixed tensor ports. Metadata such as module
identity, confidence, and provenance is represented outside the dense tensor in
the packet/ledger interfaces so the first neural comparison does not smuggle
non-differentiable strings into the training path.

## Module boundaries

Each replaceable neural component has a stable name, interface signature,
version, and parameter namespace. `ModuleRegistry` snapshots hashes and
dependencies. Replacing an expert with another implementation is therefore a
real state-dict surgery, not a file rename.

## Three timescales

- **Fast:** `FastPlasticAdapter` and recurrent block state.
- **Medium:** `RuleWorldModel` beliefs and hypotheses.
- **Slow:** `EpisodicArchive`, evidence lineage, and raw observations.

Consolidation trains a selected adapter from repeated archived experiences and
stores a provenance-linked consolidation record. It does not erase the raw
archive.

## Deliberate v0 limitations

- Character tokens and controlled synthetic streams are used to avoid data
  leakage and storage pressure.
- The world-model experiment uses a small structured condition vector alongside
  the language model; it is not presented as a complete learned world model.
- Cluster accounting uses bounded per-lineage support rather than a complete
  Bayesian treatment.
- The manual reader is a learned graph-query component evaluated against a
  factual graph oracle; it is not trusted to edit the graph.

## Foreign neural-organ boundary

Resident open-weight checkpoints are treated as versioned donor assemblies,
not as monolithic replacements. The donor path has explicit categories:
`DIRECT_GRAFT`, `FUNCTION_PRESERVING_CONVERSION`, `WRAPPED_GRAFT`,
`SUBMODULE_GRAFT`, `MECHANISM_RECONSTRUCTION`, `INSPIRATION_ONLY`, and
`DISTILLATION_CONTROL`. A candidate must first reproduce its selected donor
computation in the standalone Neural IR harness, then attach through a declared
Remora port. The donor core, transformed parameters, discarded parameters,
newly trained ports, tokens, steps, and compute are recorded separately.

The Qwen v0 pilot uses a frozen `WRAPPED_GRAFT` with exact selected trained
weights and rank-8 ports. It remains a candidate because function reproduction
and causal contribution do not by themselves establish retained capability or
compute avoidance against native and equal-budget controls.
