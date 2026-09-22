# Alice artifact and residency lane

Bounded public disposition for the Alice campaign's identity, artifact
composition, and host expert arena.

```sh
./repro/alice/artifact-and-residency-2026-09-22/run.sh
```

The lane prints the sanitized values recorded in
[`sanitized_receipt.json`](sanitized_receipt.json) and asserts no throughput.
Model weights, container bytes, and the executing runtime tree are excluded, so
this lane is not a runnable public benchmark. The narrative is in
[`research/alice/README.md`](../../../research/alice/README.md).

The invalidated 99.536% arena hit-rate figure is retained in the receipt only as
a correction record; the corrected range is 98.545-98.771% at 16 GiB.
