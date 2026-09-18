# Roadmap

## Current state (2026-09-18)

- Qwen3.8 Flash-Next generates coherent text end to end on the reference
  machine through the deployment lane (external llama.cpp-derived research
  branches), with 12/12 deployment canaries correct and a 262,144-token Q8 KV
  context passing.
- Route-aware prefetch is measured at 3–5× decode on a matched pair of runs
  over identical weights.
- The hottest 42% of expert units carry 90.2% of routed traffic, and that
  region has been rebuilt directly from BF16 (`q4_K` gate/up, `q4_0` down).
- The cold 58% and the non-expert core still run on the temporary Q2 scaffold.
- The native HAR/R4F lane's full-model generation gate remains open.

## Next

- Turn the high-route-mass region into a persistent hot/warm cache hit rather
  than a per-token stream, then measure non-thrashing resident throughput.
- Replace the remaining donor-derived regions with BF16-derived representations
  region by region, and re-run the canary and sustained-generation suites on the
  resulting bank.
- Close the final V2 quality question with a matched reference rather than a
  local-fidelity panel.
- Extend the context ladder beyond 262,144 tokens and record a 384K
  daily-driver result.
- Finish the real MTP artifact/load gate and measure real acceptance and paired
  `MTP_NET` under matched base/MTP conditions.
- Publish model-free reproduction lanes for each new measurement family as the
  receipts are sanitized.

## Later / conditional

- Evaluate the 80 and 250 tokens/s research targets only after the residency,
  representation, and MTP lanes close.
- Add public, reproducible end-to-end model fixtures behind a separate download
  step rather than storing weights.
- Stabilize R4KV versioning after independent decoder review.
- Re-run effective-context experiments with a public harness and explicit
  quality gates.
- Freeze a native R4F/Flash-Next container only after native full-model
  correctness and recovery tests exist.
- Publish research-only tools in a separate package only after each file has a
  license and provenance decision.
- Revisit omitted Laguna, HAR-X, and historical archive sources after an
  owner/license review.
