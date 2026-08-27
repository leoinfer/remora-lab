# Research methodology

Every result gets an evidence class:

- `verified` — independently rerunnable with public inputs and a checked
  invariant;
- `experimental` — implementation or bounded evidence exists, but coverage
  or reproducibility is incomplete;
- `historical` — useful context retained from an earlier private experiment;
- `invalidated` — the proposed claim failed an explicit acceptance gate.

The claims ledger is the authority for language used in public documents.
Performance is measured only after correctness, model identity, warm-up,
prompt, token count, and resource accounting are fixed. Narrow-kernel wins,
synthetic models, and reduced byte counts are not silently promoted to
end-to-end generation claims.

## Moonshots and anomalous results

This project deliberately investigates results that appear implausible,
including results that conflict with vendor specifications, conventional
performance models, or my own previous conclusions.

An implausible result is not treated as proof of a breakthrough, but it is
also not discarded merely because an existing number says it should be
impossible. Instead, the discrepancy becomes the experiment.

For example, an early gfx1200 sparse-matrix experiment appeared to indicate
multi-POPS INT4 throughput despite an expected hardware envelope around the
advertised ~821 sparse INT4 TOPS. Rather than publishing the apparent result
as a discovery or rejecting it from intuition alone, I continued constructing
stronger known-answer tests. Those tests eventually showed that the apparent
useful throughput was invalid: repeated accumulator behavior had caused the
benchmark to overcount useful committed work.

The failed result remains documented because the investigation exposed real
instruction-level behavior and improved the benchmark methodology. See the
dedicated [`gfx1200 sparse-matrix anomaly`](../research/falsified/GFX1200_SPARSE_MATRIX_ANOMALY.md)
record and the [`claims ledger`](../CLAIMS.md).

The general rule is:

**Do not reject a moonshot because it sounds impossible. Do not accept it
because it sounds exciting. Try to kill it.**
