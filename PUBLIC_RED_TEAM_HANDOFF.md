# Public red-team handoff

This is the review packet for a second, adversarial pass before publication.
It is intentionally separate from the implementation notes.

## Required checks

- inspect the complete fresh tree, not only the Git diff;
- check every source and archive decision against `PUBLICATION_ALLOWLIST.md`;
- search for private paths, usernames, hostnames, credentials, model payloads,
  screenshots, raw receipts, and hidden generated files;
- inspect Cargo metadata, dependency licenses, linked objects, and a syscall
  trace of a model-loading executable;
- verify that the HAR subtree has no Python, C/C++, CMake, foreign backend,
  subprocess, or hidden fallback path;
- test every `VERIFIED` claim and downgrade anything without a public receipt;
- check that model-architecture attribution is separate from HAR code credit;
- check that no file with unresolved provenance or license is present.

## Review state

The first pass is the implementation audit recorded by the local release
commands. A second adversarial pass must be rerun after the final tree is
frozen. Publication remains blocked until that pass reports zero findings and
the owner explicitly approves the public repository and identity.
