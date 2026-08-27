# gfx1200 sparse-matrix anomaly

**Status:** `FALSIFIED` as a useful-throughput claim

An early gfx1200 sparse-matrix experiment appeared to show multi-POPS INT4
throughput, far above the expected hardware envelope around the advertised
~821 sparse INT4 TOPS. The result was kept as an anomaly rather than accepted
as a breakthrough or rejected by intuition.

## Investigation

Stronger known-answer tests were constructed around the sparse matrix path.
They showed that repeated accumulator behavior caused the benchmark to count
work as useful and committed more than once. The apparent useful throughput
was therefore invalid as a throughput result.

The investigation still exposed real instruction-level behavior and improved
the benchmark methodology. It does not establish a multi-POPS capability,
vendor-specification violation, or end-to-end inference speedup.

## Publication disposition

The result is retained as negative knowledge. Any future sparse-throughput
claim must define useful committed work, use independent known-answer checks,
separate instruction activity from committed output, and publish the exact
kernel, hardware context, and accounting boundary.

Related records: [`C-008`](../../CLAIMS.md),
[`falsified-results.md`](../../docs/research/falsified-results.md), and the
[`moonshot methodology`](../../docs/methodology.md).
