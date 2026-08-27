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
