---
id: C-10
status: CONJECTURE
source: ../archival/authoritative/REMORA_NEW_CONJECTURES.md
originality_status: unknown
---

## C-10 — Certificate-First Autonomous Experimentation

**Status: `CONJECTURED`**

### Claim

An autonomous local research loop should be allowed to propose or run an experiment only when it can construct the verifier schema and failure-preservation path before execution. If no certificate design exists, the experiment is not yet executable research.

### Why stronger

It makes N26 a formal gate rather than a queue runner. It joins H33, Current-State, the failure ledgers, and the user instruction that empirical correlation cannot become proof.

### Counterexample search

Give a queue item with a speed metric but no exactness/timing denominator. The system must return `BLOCKED`, not run and average.

### Cheapest decisive test

CPU-only queue slice with one valid static checker, one missing artifact, and one injected failure. Require immutable evidence outputs and no promotion.

### Certificate

Experiment manifest, preconditions, allowed resources, raw artifact paths, verifier version/hash, decision, and invalidation reason.

### Affected ideas

N26, H02/H14/H29/H33, every manifest batch and the seven handoff packages.
