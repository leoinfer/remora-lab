# Alice MTP and staging lane

Bounded public disposition for the Alice MTP correctness fix and the
FreeToken-inspired staging/overlap programme.

```sh
./repro/alice/mtp-and-staging-2026-09-22/run.sh
```

Two separate conclusions live in this lane and must not be merged:

- **Correctness is fixed.** The `K > 1` rollback failure was an Alice-local
  recurrent conv snapshot plane convention (`min(slot, n_seq_tokens)` instead of
  `n_seq_tokens - slot`); the shared delta-net operator was already correct
  (682 checks, 0 failures). The KAT moved from 276/682 failing to 0/682 and the
  corrected lane shows greedy parity at `K = 0/2/3/4`.
- **Overlap is a null.** The copy/compute overlap mechanism works in isolation
  on a separate stream, but the production F2 A/B measured 515.858 pp/s control
  against 507.085 pp/s candidate, and the synchronization drains the design was
  built to hide price out at ~0.2% of a prefill pass.

See [`research/alice/README.md`](../../../research/alice/README.md) and
[`research/falsified/ALICE_CAMPAIGN_NEGATIVES.md`](../../../research/falsified/ALICE_CAMPAIGN_NEGATIVES.md).
