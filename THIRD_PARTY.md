# Third-party notices

The HAR workspace uses Rust ecosystem crates through Cargo. The exact resolved
versions and checksums are recorded in `Cargo.lock`; release review must
regenerate this notice from current Cargo package metadata.

| Component | Use | License family | Distribution decision |
| --- | --- | --- | --- |
| `ash` | Rust Vulkan bindings | MIT / Apache-2.0 | Included as a dependency; Vulkan driver remains an OS boundary. |
| `serde`, `serde_json` | Rust serialization | MIT / Apache-2.0 | Included as dependencies. |
| `sha2`, `hex` | Rust hashing/encoding | MIT / Apache-2.0 | Included as dependencies. |
| `thiserror` | Rust error declarations | MIT / Apache-2.0 | Included as a dependency. |
| `libc` | Narrow OS memory/file boundary | MIT | Included only where the Rust binary needs OS calls. |
| `regex` | Native Rust argument/model helpers | MIT / Apache-2.0 | Included as a Rust dependency. |

No upstream inference implementation is vendored. In particular, a model
format identifier or an external correctness comparison is not a dependency
on an execution library. See [PROVENANCE.md](PROVENANCE.md) for the research
references that were studied but not copied.
