# Roadmap

## Current release candidate

- Rust-only HAR host/runtime tree assembled.
- Native CPU serving and bounded Vulkan paths retained.
- R4X/R4F/R4KV research boundaries documented as experimental.
- Clean-room inventory, provenance, claims, and falsified-results records
  added.
- Model payloads and private receipts excluded.

## Next

- Add public, reproducible end-to-end model fixtures under a separate download
  step rather than storing weights.
- Finish Vulkan generation for the supported model matrix and publish shader
  build identities.
- Stabilize R4KV versioning after independent decoder review.
- Re-run effective-context experiments with a public harness and explicit
  quality gates.
- Add a release benchmark suite that reports both wins and regressions against
  clearly named baselines.

## Later / conditional

- Publish research-only Python tools in a separate package only after each
  file has a license and provenance decision.
- Revisit omitted Laguna, HAR-X, and historical archive sources after an
  owner/license review.
- Freeze a Flash-Next container only after full-model correctness and recovery
  tests exist.
