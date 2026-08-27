# Expanded research-corpus candidate

**STATUS: LOCAL ONLY — PUBLICATION BLOCKED pending independent review and owner approval**

This directory is the expanded, code-plus-research release candidate. It is
not the frozen code-centered snapshot previously pushed to the private
`leoinfer/local-ai-research` repository. That earlier snapshot remains private
and unchanged; it is intentionally treated as incomplete for the broader
publication scope.

This expanded candidate has no Git remote and has not been pushed. The larger
tree's privacy, secret, provenance, license, claim, build/test, dependency, and
runtime audits have now been rerun and pass; publication still waits for an
independent review and explicit owner approval.

## Recovered scope

The public-safe index currently covers:

- 39 HERMES-V4 mechanisms (`H01`–`H39`);
- 26 broader named families (`N01`–`N26`);
- 30 numbered REMORA manifest ideas, represented as `M01`–`M30`, plus the
  separate TBEH/PFM records;
- `OP-01`–`OP-12`, `C-01`–`C-10`, `F0`–`F15`, and `CE-01`–`CE-28`;
- all 96 `E001`–`E096` experiment queue entries; and
- 88 additional records from the preserved idea registry, including named
  systems and cross-family mechanisms.

The canonical machine-readable map is [`research_idea_index.json`](research_idea_index.json).
The source documents remain available under
[`research/archival/`](research/archival/), with section-level cards under
[`research/ideas/`](research/ideas/), [`research/open-problems/`](research/open-problems/),
[`research/conjectures/`](research/conjectures/),
[`research/theory/`](research/theory/), and
[`research/falsified/`](research/falsified/).

## HDD archaeology disposition

The mounted external research archive was inspected read-only for idea-bearing
documentation, manifests, and archive member names. Model weights, checkpoints,
raw receipts, caches, system backups, and opaque source archives were not
extracted or copied. The scan found archival confirmations of the R4X, MTP,
residency, and failure-ledger lines already represented in the local source
corpus, but no separate cleared idea manuscript that could be safely added
without file-level provenance review. The excluded source IDs and reasons are
recorded in [`research/SOURCE_REGISTER.md`](research/SOURCE_REGISTER.md) and
[`research_idea_index.json`](research_idea_index.json).

## Status vocabulary

Every indexed item uses one of the public statuses `IMPLEMENTED`,
`PARTIALLY_IMPLEMENTED`, `EXPERIMENTAL`, `TESTED`, `PROPOSED`, `HYPOTHESIS`,
`CONJECTURE`, `OPEN_PROBLEM`, `FALSIFIED`, `SUPERSEDED`, or `DEFERRED`.
Historical source labels such as `SUPPORTED`, `BLOCKED`, and `ORACLE_PENDING`
are preserved in the archival documents and mapped explicitly in the index.
