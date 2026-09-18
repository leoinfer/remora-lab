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
| C-010 | The native Flash-Next/R4F path is ready for full-model generation. | INVALIDATED | Bring-up exists, but native correctness and recovery gates are incomplete. This applies to the native lane only; a separate external research runtime produces coherent generation (C-019). |
| C-011 | HAR loads and runs models without Python, C++, llama.cpp, GGML, CMake, or a foreign backend in its runtime path. | VERIFIED FOR THIS CANDIDATE | Dependency metadata, Rust-only source gate, linked-object inspection, and syscall trace are required release evidence. |
| C-012 | The reference-machine hardware phenotype is recorded with factory, configured, observed, historical, and unknown fields. | VERIFIED FOR THIS CANDIDATE | Read-only PCI/sysfs/Vulkan/CPU/storage/software capture, official board/CPU specifications, and reconciled historical receipts in `HARDWARE_PROFILE.md` and `hardware_profile.json`. This is not a portability claim. |
| C-013 | A recovered full-model R4X-D32A `llama-bench -p W` sweep at ubatch 512 reached 699.677849 logical prefill rows/s at W512 in the tested range. | HISTORICAL | `repro/r4x/width-sweep/sanitized_receipt.json` and `results.csv`; historical diagnostic rows/s, not generation tokens/s, with the original executor and weights excluded. |
| C-014 | Public Qwen3.8-Flash-Next metadata describes a 125B model with approximately 6B activated parameters, 512 experts, 10 routed experts plus a shared expert, and one native MTP layer. | HISTORICAL | Official public model card and configuration; this is architecture metadata, not a local throughput result. |
| C-015 | The corrected MIX34 authority reconciles 49 expert blocks, 63,968,378,880 expert payload bytes, 4.15 bpw, and 192,675,840 scanned selector blocks. | HISTORICAL | Sanitized current research log; raw receipts and model payloads are outside the public tree. |
| C-016 | A bounded MIX34 warm-cache replay can report logical bytes/token and cache behavior without proving live bandwidth or resident generation speed. | EXPERIMENTAL | Partial-trace model with explicit `MODELED` and `UNMEASURED` boundaries; runtime integration is incomplete. |
| C-017 | Native Qwen3.8 MTP reconstruction identifies 31 source tensors and a transaction covering attention, recurrent, PLE, QSA, positional, and MTP-private state. | EXPERIMENTAL | Source inventory and bounded graph/transaction fixture; real-weight quality and production integration remain unmeasured. |
| C-018 | `MTP_NET_FIXTURE = 0.3275` is a synthetic random-weight fixture result, not a real trained-model multiplier. | EXPERIMENTAL | Fixture-only paired timing with 0/291 natural draft acceptances; `MTP_NET_REAL` remains unmeasured. |
| C-019 | Qwen3.8 Flash-Next generates coherent text end-to-end on the reference machine through the deployment lane; 12/12 deployment canaries produced correct outputs and two 128-token completions ran to budget. | EXPERIMENTAL | Sanitized receipt in `repro/flash-next/deployment-2026-09-18/`. Canary verdicts are evaluated against expected answers and the harness has no automated grader; short-context deployment profile. |
| C-020 | A 262,144-token context passes with the Q8 KV configuration on the reference machine. | EXPERIMENTAL | Same receipt. Capacity and coherence smoke only, with no long-context quality claim; a 384K profile was defined but never run. |
| C-021 | The hottest 42% of `(layer, expert)` units (10,321 of 24,576) carry 90.19590358841332% of routed access mass on a 1,542-step, 6-workload route trace. | EXPERIMENTAL | Route-trace section of the deployment receipt. Trace-derived on one captured trace set; not claimed as a prompt-independent invariant. |
| C-022 | Route-aware prefetch on identical weights raised decode from 1.28/0.81 to 3.86/4.07 tokens/s and cut major page faults from 3,931,682 to 13,714 in one matched control/prefetch pair. | EXPERIMENTAL | Sanitized receipt plus `research/flash-next/PREFETCH_RESULT.md`. Single pair run back-to-back; device counters are machine-wide and prefetch emission counters were not logged. |
| C-023 | On the measured equal-added-byte panel, a 2-bit residual codebook removed approximately 47.7% of activation error where a dense ternary T2 plane removed approximately 20.8% over the donor base, and an LS-scale ternary re-encode removed approximately 62.9% over the T1 base where dense ternary T2 removed approximately 46.0%. | EXPERIMENTAL | `repro/flash-next/representation-panel/`. Tensor- and activation-level fidelity on a 9-slice panel; not capability evidence, and the teacher-KL probe was not captured. |
| C-024 | A Q8_0 residual island on layer 0 / expert 283 moved weight relRMS from approximately 0.779 to approximately 0.0032 with 5,222,400 bytes of residual payload, at about 4.86× the equal-byte budget. | EXPERIMENTAL | Same lane. Mechanism evidence only: the island is byte-inefficient next to a stronger base codec and is explicitly not the final representation. |
| C-025 | The V2 hot region for the hottest 42% of expert units is rebuilt directly from the BF16 authority as `q4_K` gate/up and `q4_0` down, verified by byte-identical re-encode; the cold 58% and the non-expert core remain on the temporary Q2 scaffold. | EXPERIMENTAL | Deployment receipt. This is a migration configuration, not final ancestry and not a quality result. |
| C-026 | Streaming the BF16-derived V2 hot bank measured 0.73–1.29 tokens/s decode at 0.229–1.667 GB/token of device-normalized reads, slower than the scaffold control; streaming that region is rejected as the intended architecture. | EXPERIMENTAL | Deployment receipt. The streaming measurements are `MEASURED`; the architectural rejection is an interpretation recorded alongside them. |

Claims C-001 through C-006 and C-009 are research statements, not a promise
of production coverage. C-011 and C-012 remain valid only while the release
commands in `DEVELOPMENT.md` have passed on the final tree; regenerate
evidence when that tree changes. C-012 describes one reference phenotype and
does not generalize to other hardware.
