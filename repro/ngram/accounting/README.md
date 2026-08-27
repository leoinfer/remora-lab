# N-gram speculation accounting lane

This lane reproduces the public bookkeeping boundary for workload-shaped
token replay. It uses deterministic warm-shell and template sequences, records
accepted tokens and trace digests, and reports throughput as deliberately not
measured.

Run from the repository root:

```sh
./repro/ngram/accounting/run.sh
```

The two workloads are exact synthetic sequences. They are not generic 27B
neural decode and do not reproduce historical n-gram tokens/s anchors.
