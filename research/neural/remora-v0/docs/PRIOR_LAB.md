# Relationship to the GitHub remora-lab checkout

The local GitHub checkout is:

```text
/home/leo/public-staging/local-ai-research-expanded
origin: https://github.com/leoinfer/remora-lab.git
```

Its `research/developmental_learning` package is the prior control plane used
for deterministic causal worlds, immutable raw episodes, clustered retrieval,
hypothesis ledgers, replay controls, and self-edit verification. At audit time
its existing test suite passed 219 tests. The current neural v0 workspace is
kept separate because that checkout contains a large pre-existing dirty/untracked
research campaign.

The neural workspace does not reuse pretrained weights or claim that the
existing symbolic results are neural evidence. It reuses the lab as a reference
for experiment hygiene and uses the same distinctions in a small trainable
world-model and language model. Future integration should happen as a reviewed
GitHub change after the neural artifacts have stable interfaces and receipts.
