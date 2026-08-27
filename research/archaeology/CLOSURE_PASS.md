# Closure-pass archaeology record

**Capture date:** 2026-08-27

This record describes the search boundary and the disposition policy; it is
not a copy of the private machine inventory. The shared-machine identity was
verified through the read-only research bridge before HAR and hardware
archaeology. No hostname, private path, branch name, model path, machine
identifier, model weight, raw receipt, or opaque archive content is published
here.

## Search boundary

- The research root contained 68 top-level directories and 58 top-level files
  at capture time.
- Thirty-seven repository marker directories and 69 HAR worktree records were
  checked, including the required dense-profiler worktree and archived or
  prunable worktree metadata.
- The external research disk was inspected read-only: 444 files and six
  archives were inventoried. No separately cleared idea manuscript was found.
- The whole-machine pass included notes, reports, handoffs, state documents,
  receipts, abandoned implementations, current campaign material, generated
  summaries, historical bundles, model and build inventories, and upstream
  source markers.

The detailed normalized records are in
[`PUBLICATION_COVERAGE_MATRIX.md`](../../PUBLICATION_COVERAGE_MATRIX.md) and
[`publication_coverage_matrix.json`](../../publication_coverage_matrix.json).
The earlier research index and source register remain the readable
cross-checks for the included idea corpus.

## Disposition rule

Every normalized idea, source-register entry, root artifact-register entry,
source-tree/worktree inventory unit, HDD aggregate, current campaign asset,
and mandatory family crosswalk receives one allowed public disposition. Raw
payloads are not copied merely because they were found. Model weights,
checkpoints, datasets, tokenizer payloads, private archives, raw receipts,
build outputs, foreign execution trees, and unclear-license source remain
explicitly excluded or pending.

The closure invariant is `UNACCOUNTED = 0` within this declared boundary. It
means every discovered unit has a recorded decision; it does not mean every
unit is public or that every research idea has a production implementation.
