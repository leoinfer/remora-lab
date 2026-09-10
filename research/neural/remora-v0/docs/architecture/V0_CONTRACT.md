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
