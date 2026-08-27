# Security

Do not report credentials, private model files, local paths, or unpublished
machine data in a public issue. Remove sensitive material and use the host's
private security-advisory channel once this repository has a public security
contact. Until then, keep the report local and do not publish it with the
source tree.

The runtime treats model files and compiled plans as untrusted input. New
loaders must retain bounds checks, checked arithmetic, explicit format
validation, and fail-closed behavior for unknown capabilities.
