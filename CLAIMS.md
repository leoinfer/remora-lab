# Claims ledger

Statuses are deliberate: `VERIFIED` means supported by a reproducible local
check in this candidate; `EXPERIMENTAL` means code or a bounded test exists;
`HISTORICAL` means the observation is retained but its original receipt is not
part of the public tree; `INVALIDATED` means the proposed result failed its
own acceptance rule.

| ID | Statement | Status | Evidence boundary |
| --- | --- | --- | --- |
| C-001 | HAR has native Rust CPU paths for bounded dense and routed-MoE model operations. | EXPERIMENTAL | Source and unit tests; no bundled full model. |
| C-002 | HAR's production policy rejects fallback and foreign adapter invocation. | VERIFIED | `har-runtime` policy tests and Rust-only gate. |
| C-003 | HAR can build a Vulkan resource/dispatch layer through Rust bindings. | EXPERIMENTAL | Vulkan crate and shader fixtures; device-specific execution remains required. |
| C-004 | R4X D32A geometry is represented as 256 values in 144 bytes and checked by row-window tests. | EXPERIMENTAL | Parser geometry tests; no claim of ecosystem standardization. |
| C-005 | R4KV profiles reduce KV storage accounting relative to its F16 reference profile. | EXPERIMENTAL | Rust profile arithmetic and codec tests; model-quality impact is unmeasured here. |
| C-006 | A 10M effective-context target can be discussed as a memory/accounting hypothesis. | EXPERIMENTAL | Explicit accounting notes; not dense attention at 10M tokens. |
| C-007 | Historical comparable paths sometimes trailed llama.cpp by several tokens per second. | HISTORICAL | Retained as a caution; no historical private receipt is presented as public proof. |
| C-008 | The proposed SWMMAC multi-POPS result is accepted. | INVALIDATED | The original acceptance and accounting gates were not satisfied. |
| C-009 | MTP/speculative decode can expose acceptance and resource telemetry. | EXPERIMENTAL | Rust scheduler contracts; no universal speedup claim. |
| C-010 | Flash-Next/R4F is ready for full-model generation. | INVALIDATED | Bring-up exists, but full-model correctness and recovery gates are incomplete. |
| C-011 | HAR loads and runs models without Python, C++, llama.cpp, GGML, CMake, or a foreign backend in its runtime path. | VERIFIED FOR THIS CANDIDATE | Dependency metadata, Rust-only source gate, linked-object inspection, and syscall trace are required release evidence. |

Claims C-001 through C-006 and C-009 are research statements, not a promise
of production coverage. C-011 remains valid only while the release commands
in `DEVELOPMENT.md` have passed on the final tree; regenerate evidence when
that tree changes.
