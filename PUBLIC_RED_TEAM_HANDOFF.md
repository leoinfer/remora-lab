# Public red-team handoff

This is the public-safe handoff for the completed adversarial publication
pass. It is intentionally separate from the implementation notes.

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

## Final audit state

The implementation audit and final adversarial pass were rerun against the
frozen publication tree. They reported zero remaining sensitive findings, and
the owner explicitly authorized publication of `leoinfer/remora-lab`. The
detailed audit companion is outside the repository by design.
