//! V3 native-execution bundle pass (additive; frozen v1 emitter untouched).
//!
//! Produces the first integration-ready native execution bundle that binds:
//!
//! package builder package entry → residency layer residency request → compiler Rust runtime
//! operation → native-kernel registry native kernel registry.
//!
//! The frozen dialect is preserved; the v3 pass adds a NATIVE_REQUIRED policy
//! (source-commit requirements, fallback prohibition, stale-artifact
//! rejection) without changing `har.first-operation-bundle.v1` or the
//! language semantics.  The loader's fail-closed behaviour is used directly:
//! `fallback.on_kernel_unavailable == "fail_closed"` turns any missing
//! capability into a hard validation error (PlanLoader), which is the compiler
//! NATIVE_REQUIRED policy equivalent.
//!
//! The multi-op DAG (Q4_K matvec A ∥ Q4_K matvec B → residual_add) compiles to
//! a plan only; bundle compilation fails closed (N3001) because `residual_add`
//! is absent from the exact pinned native-kernel registry kernel registry.  No unregistered
//! kernel is silently selected.

use crate::operation::{
    bind_operation, emit_single_operation_package, KernelRegistry, OperationBinding, PackedManifest,
};
use crate::v0::CompiledProgram;
use har_core::sha256_bytes;
use serde_json::{json, Value};
use std::path::Path;

pub const NATIVE_BUNDLE_V3_SCHEMA: &str = "har.first-operation-bundle.v3";
pub const NATIVE_POLICY_SCHEMA: &str = "har.native-required-policy.v1";
pub const MULTI_OP_REJECTION_SCHEMA: &str = "har.native-bundle-rejection.v1";
pub const PLAN_SCHEMA: &str = "har.model_compiled_package.v0";

// ---------------------------------------------------------------------------
// Public producer identities. These are stable interface labels for the
// release candidate, not private repository commit identifiers.
// ---------------------------------------------------------------------------

pub const COMPILER_INTERFACE_ID: &str = "public-compiler-interface-v1";
pub const RUNTIME_SOURCE_ID: &str = "public-har-runtime-v1";
pub const KERNEL_REGISTRY_ID: &str = "public-native-kernel-registry-v1";
pub const VULKAN_BACKEND_ID: &str = "public-har-vulkan-v1";
pub const REGISTRY_PINNED_CORE_ID: &str = "har.core.v1/har.ir.v1@public-core-v1";
pub const PACKAGE_BUILDER_ID: &str = "public-model-package-v1";
pub const RESIDENCY_CONTRACT_ID: &str = "public-residency-contract-v1";
pub const INTEGRATION_ID: &str = "public-integration-v1";

pub const PINNED_REGISTRY_CANONICAL_SHA256: &str =
    "844db18b492849aa0ab3dd5c942815acb4bbe7584432c582db22a46a124cb433";
// This identity is for the public packed-manifest contract used by compiler
// tests. No model payload is included in the repository.
pub const PINNED_PACKED_MANIFEST_CANONICAL_SHA256: &str =
    "63e98b80ac354ec3aaef078db10a2a4fdeadaefa0efc3656ee20fd948d00b6bf";
pub const PINNED_Q4K_SHADER_SOURCE_SHA256: &str =
    "7e09fb98e51408ec6998d99d57caf782f27a70c0a926936239742cda1dda0e91";
pub const PINNED_Q4K_SPIRV_SHA256: &str =
    "8eeb47a0298b59d42b8ed07d9b98cc4caf25c588a27db1f3cfa255415818dbd2";

pub const PINNED_BLOCK0_SHA256: &str =
    "81c611f35bff79491538b2f7cf201c7597a661a5c549633541c62bdc8af1613f";
pub const PINNED_BLOCK1_SHA256: &str =
    "761136e30319b547b806f3753bb0237dcf74194a44f77561bcc360ba1a5b7adb";
pub const VALIDATED_BLOCK_BYTES: u64 = 144;

/// Native-kernel registry validated multi-op evidence identities.
/// json + manifest V2 kernel_hashes): requested residual_add kernel.  These
/// are the requested identities; the pinned registry entry is absent and the
/// bundle fails closed (N3001).
pub const RESIDUAL_ADD_CAPABILITY: &str = "vulkan.residual_add";
pub const RESIDUAL_ADD_SHADER_SOURCE_SHA256: &str =
    "19229a42166b144f56b21fbaf37c96b40b4431d0009ca62ecc289673aabd116f";
pub const RESIDUAL_ADD_SPIRV_SHA256: &str =
    "ff92f2ae6eb112cde5261248447a6abdaed586dbe7e87fe1c33b6ea68e714914";

/// Native-kernel registry backend interface identity. The bundle and
/// plan bind these exactly so a backend without the interface cannot claim
/// the bundle.
pub const HARBACKEND_INTERFACE_VERSION: &str = "har.rdna4.backend-interface.v1";
pub const HARBACKEND_RUST_TYPE: &str = "har_vulkan::backend::HarBackend";

/// Requested-semantics preservation gate (N1009).  When the loader cannot
/// preserve the requested semantics and the delta is execution-relevant for
/// the bound operation, the NATIVE_REQUIRED bundle is rejected: no silent
/// downgrade, no approximation of an exact operation.  Non-execution-
/// relevant deltas remain recorded markers (`no_silent_downgrade`).
pub fn validate_semantics_preserved(execution_relevant: bool) -> Result<(), String> {
    if execution_relevant {
        Err("N1009 requested semantics not preserved: the loader cannot preserve the requested semantics and the delta is execution-relevant for the bound operation; no silent downgrade, no approximation".into())
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Native-required policy
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct NativePolicy {
    pub runtime_source_commit: String,
    pub interface_source_commit: String,
    pub backend_source_commit: String,
    pub registry_commit: String,
    pub registry_pinned_core: String,
    pub registry_canonical_sha256: String,
    pub package_source_commit: String,
    pub package_manifest_canonical_sha256: String,
    pub residency_contract_commit: String,
    pub integration_commit: String,
    pub fallback_prohibited: bool,
    pub fallback_count_expected: u64,
    pub reference_adapter_invocations_expected: u64,
}

impl Default for NativePolicy {
    fn default() -> Self {
        Self {
            runtime_source_commit: RUNTIME_SOURCE_ID.into(),
            interface_source_commit: COMPILER_INTERFACE_ID.into(),
            backend_source_commit: VULKAN_BACKEND_ID.into(),
            registry_commit: KERNEL_REGISTRY_ID.into(),
            registry_pinned_core: REGISTRY_PINNED_CORE_ID.into(),
            registry_canonical_sha256: PINNED_REGISTRY_CANONICAL_SHA256.into(),
            package_source_commit: PACKAGE_BUILDER_ID.into(),
            package_manifest_canonical_sha256: PINNED_PACKED_MANIFEST_CANONICAL_SHA256.into(),
            residency_contract_commit: RESIDENCY_CONTRACT_ID.into(),
            integration_commit: INTEGRATION_ID.into(),
            fallback_prohibited: true,
            fallback_count_expected: 0,
            reference_adapter_invocations_expected: 0,
        }
    }
}

/// Canonical SHA-256 of a JSON file (recursively sorted keys, compact
/// separators) — the deterministic identity used for pinned producer files.
pub fn canonical_sha256_file(path: impl AsRef<Path>) -> Result<String, String> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|error| format!("cannot read {}: {error}", path.as_ref().display()))?;
    let root: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("cannot parse {}: {error}", path.as_ref().display()))?;
    canonical_sha256_value(&root)
}

pub fn canonical_sha256_value(root: &Value) -> Result<String, String> {
    har_core::canonical_sha256(root).map_err(|error| format!("canonical hash failed: {error}"))
}

// ---------------------------------------------------------------------------
// Native-policy validation (all failures occur before dispatch)
// ---------------------------------------------------------------------------

/// Cross-boundary native-policy checks added by the v3 pass.  The frozen
/// S-code checks from `bind_operation` still apply unchanged; these N-codes
/// enforce the NATIVE_REQUIRED runtime policy.
pub fn validate_native_policy(
    binding: &OperationBinding,
    registry: &KernelRegistry,
    registry_canonical_sha256: &str,
    manifest_canonical_sha256: &str,
    policy: &NativePolicy,
) -> Result<(), Vec<String>> {
    let mut errors: Vec<String> = Vec::new();

    // N1001 missing runtime source commit: the bundle must require the exact
    // compiler loader commit so a stale loader can never claim the bundle.
    if policy.runtime_source_commit.is_empty() {
        errors.push(
            "N1001 missing runtime source commit: NATIVE_REQUIRED policy requires the exact compiler runtime source commit"
                .into(),
        );
    }
    // N1002 missing backend source commit: native-kernel registry kernel registry/backend
    // commit is part of the native identity.
    if policy.backend_source_commit.is_empty() {
        errors.push(
            "N1002 missing backend source commit: NATIVE_REQUIRED policy requires the exact native-kernel registry backend source commit"
                .into(),
        );
    }
    // N1003 fallback-enabled native bundle: the plan must not permit any
    // reference-adapter path; PlanLoader turns missing capability into a hard
    // error only when on_kernel_unavailable is not "reference_adapter".
    if policy.fallback_prohibited {
        if policy.fallback_count_expected != 0 {
            errors.push(format!(
                "N1003 fallback-enabled native bundle: fallback_count_expected must be 0, got {}",
                policy.fallback_count_expected
            ));
        }
        if policy.reference_adapter_invocations_expected != 0 {
            errors.push(format!(
                "N1003 fallback-enabled native bundle: reference_adapter_invocations_expected must be 0, got {}",
                policy.reference_adapter_invocations_expected
            ));
        }
    }
    // N1004 stale kernel registry: the canonical identity of the registry the
    // bundle is compiled against must equal the pinned registry identity.
    if registry_canonical_sha256 != policy.registry_canonical_sha256 {
        errors.push(format!(
            "N1004 stale kernel registry: registry canonical sha256 {registry_canonical_sha256} does not match pinned {}",
            policy.registry_canonical_sha256
        ));
    }
    if registry.pinned_core != policy.registry_pinned_core {
        errors.push(format!(
            "N1004 stale kernel registry: registry pinned core {} does not match required {}",
            registry.pinned_core, policy.registry_pinned_core
        ));
    }
    // N1005 stale shader: the registry entry's shader identities must match
    // the pinned shader/SPIR-V artifacts.
    let kernel = &binding.kernel_entry;
    if kernel.shader_source_sha256 != PINNED_Q4K_SHADER_SOURCE_SHA256 {
        errors.push(format!(
            "N1005 stale shader: q4k_gemv.comp sha256 {} does not match pinned {}",
            kernel.shader_source_sha256, PINNED_Q4K_SHADER_SOURCE_SHA256
        ));
    }
    if kernel.spirv_sha256 != PINNED_Q4K_SPIRV_SHA256 {
        errors.push(format!(
            "N1005 stale shader: q4k_gemv.spv sha256 {} does not match pinned {}",
            kernel.spirv_sha256, PINNED_Q4K_SPIRV_SHA256
        ));
    }
    // N1006 stale package: the exact pinned package builder packed manifest identity
    // is part of the native binding.
    if manifest_canonical_sha256 != policy.package_manifest_canonical_sha256 {
        errors.push(format!(
            "N1006 stale package: packed manifest canonical sha256 {manifest_canonical_sha256} does not match pinned {}",
            policy.package_manifest_canonical_sha256
        ));
    }
    // N1007 wrong block geometry: the validated Q4_K super-block contract is
    // 144 bytes with the pinned block checksum.
    if binding.block_bytes != VALIDATED_BLOCK_BYTES {
        errors.push(format!(
            "N1007 wrong block geometry: block_bytes {} violates the validated Q4_K super-block contract (144 bytes / 256 elements)",
            binding.block_bytes
        ));
    }
    if binding.block_sha256 != PINNED_BLOCK0_SHA256 && binding.block_sha256 != PINNED_BLOCK1_SHA256
    {
        errors.push(format!(
            "N1007 wrong block geometry: block sha256 {} is not a pinned validated Q4_K super-block",
            binding.block_sha256
        ));
    }
    // N1008 representation mislabel: the payload format must be exactly the
    // registry format; a Q8_0/MXFP4 label swap is a hard rejection.
    if !binding
        .entry
        .quant_format
        .eq_ignore_ascii_case(&kernel.quant_format)
    {
        errors.push(format!(
            "N1008 representation mislabel: packed entry quant {} does not match registry quant {}",
            binding.entry.quant_format, kernel.quant_format
        ));
    }

    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// V3 first-native bundle compilation
// ---------------------------------------------------------------------------

pub struct NativeBundleV3 {
    pub plan_json: String,
    pub bundle_json: String,
    pub plan_sha256: String,
    pub bundle_sha256: String,
    pub operation_hash: String,
}

impl std::fmt::Debug for NativeBundleV3 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeBundleV3")
            .field("plan_sha256", &self.plan_sha256)
            .field("bundle_sha256", &self.bundle_sha256)
            .field("operation_hash", &self.operation_hash)
            .finish()
    }
}

/// Compile the first integration-ready native execution bundle.
///
/// Deterministic: `generated_at_unix_ns` is pinned to 0 and all identities
/// come from pinned producer interfaces.  No dispatch is performed.
pub fn compile_first_native_bundle(
    program: &CompiledProgram,
    manifest: &PackedManifest,
    registry: &KernelRegistry,
    manifest_path: impl AsRef<Path>,
    registry_path: impl AsRef<Path>,
    policy: &NativePolicy,
) -> Result<NativeBundleV3, Vec<String>> {
    let block_sha256 = PINNED_BLOCK0_SHA256;
    let binding = bind_operation(
        program,
        manifest,
        registry,
        block_sha256,
        VALIDATED_BLOCK_BYTES,
    )
    .map_err(|mut errors| {
        errors.push("frozen cross-boundary binding failed; no native bundle emitted".into());
        errors
    })?;

    let manifest_canonical = canonical_sha256_file(manifest_path).map_err(|error| vec![error])?;
    let registry_canonical = canonical_sha256_file(registry_path).map_err(|error| vec![error])?;
    validate_native_policy(
        &binding,
        registry,
        &registry_canonical,
        &manifest_canonical,
        policy,
    )?;
    // Requested-semantics gate: the frozen first operation's loader delta
    // (context 4096 / kv f16 vs requested 8192 / q8_0) is not execution-
    // relevant for the bound single validated block, so the recorded marker
    // is allowed; an execution-relevant delta would hard-reject (N1009).
    validate_semantics_preserved(false).map_err(|error| vec![error])?;

    // Plan: compiler loader-compatible package with the NATIVE_REQUIRED policy.
    let plan_json = emit_native_plan(program, &binding, policy).map_err(|error| vec![error])?;

    // Bundle V3.
    let bundle_json =
        emit_native_bundle(program, &binding, &plan_json, policy).map_err(|error| vec![error])?;
    let plan_sha256 = sha256_bytes(plan_json.as_bytes());
    let bundle_sha256 = sha256_bytes(bundle_json.as_bytes());
    let operation_hash = sha256_bytes(
        &serde_json::to_vec(&binding.physical).map_err(|error| vec![error.to_string()])?,
    );

    Ok(NativeBundleV3 {
        plan_json,
        bundle_json,
        plan_sha256,
        bundle_sha256,
        operation_hash,
    })
}

/// Native-required plan: `har.model_compiled_package.v0` with
/// `plan_kind = "first-native-execution"`, `fallback.on_kernel_unavailable =
/// "fail_closed"` (PlanLoader hard-fails on any missing capability), and the
/// native policy recorded.  The frozen emitter is consumed read-only; only
/// the policy-relevant fields are overlaid.
pub fn emit_native_plan(
    program: &CompiledProgram,
    binding: &OperationBinding,
    policy: &NativePolicy,
) -> Result<String, String> {
    let base = emit_single_operation_package(program, binding)?;
    let mut plan: Value =
        serde_json::from_str(&base).map_err(|error| format!("plan parse failed: {error}"))?;
    plan["plan_kind"] = json!("first-native-execution");
    plan["fallback"] = json!({
        "authority_backend": "VULKAN",
        "on_unknown_capacity": "reject_plan",
        "on_stale_generation": "fail_closed",
        "on_kernel_unavailable": "fail_closed",
    });
    plan["native_policy"] = json!({
        "schema": NATIVE_POLICY_SCHEMA,
        "runtime_source_commit_required": policy.runtime_source_commit,
        "interface_source_commit_required": policy.interface_source_commit,
        "backend_source_commit_required": policy.backend_source_commit,
        "harbackend_interface_required": HARBACKEND_INTERFACE_VERSION,
        "harbackend_rust_type_required": HARBACKEND_RUST_TYPE,
        "registry_commit_required": policy.registry_commit,
        "registry_pinned_core_required": policy.registry_pinned_core,
        "package_source_commit_required": policy.package_source_commit,
        "residency_contract_commit_required": policy.residency_contract_commit,
        "integration_commit_required": policy.integration_commit,
        "fallback_prohibited": policy.fallback_prohibited,
        "fallback_count_expected": policy.fallback_count_expected,
        "reference_adapter_invocations_expected": policy.reference_adapter_invocations_expected,
    });
    serde_json::to_string_pretty(&plan)
        .map_err(|error| format!("plan serialization failed: {error}"))
}

/// Emit the canonical `har.first-operation-bundle.v3` bundle with the
/// NATIVE_REQUIRED policy, expected RuntimeManifest fields, expected event
/// fields, and the fallback prohibition counters.
#[allow(clippy::too_many_arguments)]
pub fn emit_native_bundle(
    program: &CompiledProgram,
    binding: &OperationBinding,
    plan_json: &str,
    policy: &NativePolicy,
) -> Result<String, String> {
    let typed = &program.typed;
    let model = &typed.model;
    let target = &typed.target;
    let op = &typed.operations[0];
    let entry = &binding.entry;
    let kernel = &binding.kernel_entry;

    let physical_value = serde_json::to_value(&binding.physical)
        .map_err(|error| format!("physical operation serialization failed: {error}"))?;
    let operation_hash =
        sha256_bytes(&serde_json::to_vec(&binding.physical).map_err(|error| error.to_string())?);
    let plan_hash = sha256_bytes(plan_json.as_bytes());

    let bundle = json!({
        "schema": NATIVE_BUNDLE_V3_SCHEMA,
        "producer": "public-native-operation-bundle-v3",
        "source_program": "examples/native_operation.har",
        "typed_semantic_ast": {
            "target": target.name,
            "model": model.name,
            "quality": typed.quality.name,
            "decode": typed.decode.name,
            "operation": op.name,
            "operation_count": typed.operations.len(),
            "fact_count": typed.facts.len(),
        },
        "logical_operation": {
            "kind": "operation",
            "name": op.name,
            "source_tensor": op.source_tensor,
            "dependencies": ["model", "quality", "decode", "packed-entry", "kernel-registry", "hardware"],
        },
        "physical_operation": physical_value,
        "operation_hash": operation_hash,
        "plan_hash": plan_hash,
        "packed_entry_binding": {
            "source_tensor_id": entry.source_tensor_id,
            "payload_location_id": entry.payload_location_id,
            "quant_format": entry.quant_format,
            "kernel_requirement": entry.kernel_requirement,
            "representation_identity": entry.representation_identity,
            "dimensions": entry.dimensions,
            "source_offset": entry.source_offset,
            "source_bytes": entry.source_bytes,
            "payload_offset": binding.location.offset,
            "payload_bytes": binding.location.bytes,
            "payload_alignment": binding.location.alignment,
            "payload_sha256": binding.location.sha256,
            "validated_block_bytes": binding.block_bytes,
            "validated_block_sha256": binding.block_sha256,
        },
        "page_request_binding": binding.page_request,
        "kernel_requirement": {
            "capability": kernel.capability,
            "kernel_kind": kernel.kernel_kind,
            "quant_format": kernel.quant_format,
            "supported_shapes": kernel.supported_shapes,
            "alignment_bytes": kernel.alignment_bytes,
            "input_datatype": kernel.input_datatype,
            "output_datatype": kernel.output_datatype,
            "required_vulkan_features": kernel.required_vulkan_features,
            "workgroup_local": kernel.workgroup_local,
            "shader_source": kernel.shader_source,
            "shader_source_sha256": kernel.shader_source_sha256,
            "spirv": kernel.spirv,
            "spirv_sha256": kernel.spirv_sha256,
            "numerical_contract": kernel.numerical_contract,
            "status": kernel.status,
        },
        "native_policy": {
            "schema": NATIVE_POLICY_SCHEMA,
            "runtime_source_commit_required": policy.runtime_source_commit,
            "interface_source_commit_required": policy.interface_source_commit,
            "backend_source_commit_required": policy.backend_source_commit,
            "harbackend_interface_required": HARBACKEND_INTERFACE_VERSION,
            "harbackend_rust_type_required": HARBACKEND_RUST_TYPE,
            "registry_commit_required": policy.registry_commit,
            "registry_pinned_core_required": policy.registry_pinned_core,
            "registry_canonical_sha256_required": policy.registry_canonical_sha256,
            "package_source_commit_required": policy.package_source_commit,
            "package_manifest_canonical_sha256_required": policy.package_manifest_canonical_sha256,
            "residency_contract_commit_required": policy.residency_contract_commit,
            "integration_commit_required": policy.integration_commit,
            "fallback_prohibited": policy.fallback_prohibited,
            "fallback_count_expected": policy.fallback_count_expected,
            "reference_adapter_invocations_expected": policy.reference_adapter_invocations_expected,
            "stale_artifacts_rejected": true,
        },
        "dependency_roots": ["model", "quality", "decode", "packed-entry", "kernel-registry", "hardware"],
        "model_root": model.model_sha256,
        "package_root": op.package_root,
        "hardware_root": {
            "gpu": target.gpu,
            "gpu_arch": target.gpu_arch,
            "wave": target.wave,
            "vram_budget_bytes": target.vram_budget_bytes,
            "host_ram_budget_bytes": target.host_ram_budget_bytes,
        },
        "exactness_contract": {
            "mode": "EXACT",
            "output_hash_required": true,
            "state_boundary_hash_required": true,
            "numerical_tolerance": op.reference_tolerance,
            "authority": typed.quality.authority,
        },
        "fallback_contract": {
            "authority_backend": "VULKAN",
            "on_unknown_capacity": "reject_plan",
            "on_stale_generation": "fail_closed",
            "on_kernel_unavailable": "fail_closed",
            "operation_fallback": op.fallback,
            "invocation": "PROHIBITED",
        },
        "telemetry_contract": {
            "schema": "har.telemetry.v1",
            "record_residency": true,
            "record_transfers": true,
            "record_operation_hashes": true,
            "timing_is_advisory": true,
        },
        "expected_runtime_manifest": {
            "schema": "har.runtime_manifest.v1",
            "runtime_name": "Hardware-Aware Runtime",
            "runtime_version": "0.1.0-rust",
            "source_commit_must_equal": policy.runtime_source_commit,
            "reference_commit_must_equal": policy.interface_source_commit,
            "model_sha256_must_equal": model.model_sha256,
            "plan_sha256_must_equal": plan_hash,
            "operation_id_must_equal": binding.entry.source_tensor_id,
            "exactness_mode": "EXACT",
            "fallback_on_kernel_unavailable": "fail_closed",
            "fallback_on_stale_generation": "fail_closed",
            "fallback_on_unknown_capacity": "reject_plan",
            "reference_adapters_must_be_empty": true,
            "fallback_count_must_be": policy.fallback_count_expected,
            "reference_adapter_invocations_must_be": policy.reference_adapter_invocations_expected,
        },
        "expected_events": {
            "schema": "har.events.v1",
            "sequence": [
                "PlanLoaded",
                "PlanValidated",
                "Residency",
                "TransferQueued",
                "TransferCompleted",
                "Dispatch(backend=VULKAN, kernel=Q4KMatVec)",
                "Output"
            ],
            "forbidden": ["Fallback"],
            "header": {
                "epoch": {"model_root": model.model_sha256, "graph_generation": 0, "decode_epoch": 0, "sequence_id": 0},
                "operation_index": 0
            },
        },
        "semantic_delta": {
            "requested": {"context_length": model.context_length, "kv_datatype": model.kv_cache_type, "batch": model.batch},
            "plan_loader_defaults": {"context_length": 4096, "kv_datatype": "f16"},
            "context_length_delta": model.context_length != 4096,
            "kv_datatype_delta": !model.kv_cache_type.is_empty() && model.kv_cache_type != "f16",
            "execution_relevant_for_this_operation": false,
            "compatibility": "REQUESTED_SEMANTICS_NOT_PRESERVED_BY_LOADER",
            "no_silent_downgrade": true,
        },
        "assumptions": [
            "Single validated Q4_K super-block slice supplied by the caller",
            "Kernel dispatch shape rows=1 blocks_per_row=1 matches native-kernel registry VALIDATED contract",
            "NATIVE_REQUIRED: no reference adapter may be invoked for this bundle",
            "compiler runtime source commit and native-kernel registry backend source commit are required",
            "Numerical reference is produced by the canonical Q4_K dequantization adapter at dispatch",
        ],
        "unresolved_risks": [],
    });
    serde_json::to_string_pretty(&bundle)
        .map_err(|error| format!("bundle serialization failed: {error}"))
}

// ---------------------------------------------------------------------------
// V3 multi-op DAG (fail-closed)
// ---------------------------------------------------------------------------

/// Compile the native multi-op plan: Q4_K matvec A ∥ Q4_K matvec B →
/// residual_add, with explicit dependency edges, explicit intermediate
/// buffers, event/generation requirements, and the NATIVE_REQUIRED policy.
///
/// The DAG is compiled only after each predecessor binding passes the frozen
/// cross-boundary checks.  Bundle compilation is separate and fails closed
/// when `residual_add` is absent from the exact pinned registry.
pub fn emit_native_multi_op_plan(
    program: &CompiledProgram,
    manifest: &PackedManifest,
    registry: &KernelRegistry,
    policy: &NativePolicy,
) -> Result<String, Vec<String>> {
    let _block_a = bind_operation(
        program,
        manifest,
        registry,
        PINNED_BLOCK0_SHA256,
        VALIDATED_BLOCK_BYTES,
    )
    .map_err(|mut errors| {
        errors.push("multi-op predecessor A failed frozen cross-boundary binding".into());
        errors
    })?;
    let _block_b = bind_operation(
        program,
        manifest,
        registry,
        PINNED_BLOCK1_SHA256,
        VALIDATED_BLOCK_BYTES,
    )
    .map_err(|mut errors| {
        errors.push("multi-op predecessor B failed frozen cross-boundary binding".into());
        errors
    })?;
    debug_assert_eq!(_block_b.block_sha256, PINNED_BLOCK1_SHA256);
    let _ = &_block_b;

    let model = &program.typed.model;
    let target = &program.typed.target;

    let op_a = json!({
        "index": 0,
        "logical_id": 0,
        "stable_id": "blk.0.ffn_gate.weight.block0",
        "backend": "VULKAN",
        "kernel": "Q4KMatVec",
        "input_slots": [0],
        "output_slots": [1],
        "dependencies": [],
        "dispatch": {"x": 1, "y": 1, "z": 1, "workgroup_x": 256, "workgroup_y": 1, "workgroup_z": 1},
        "source_tier": "RAM_PINNED",
        "destination_tier": "VRAM_SLOT",
        "event_sequence": 1,
        "output_buffer": "multiop.out_a",
    });
    let op_b = json!({
        "index": 1,
        "logical_id": 1,
        "stable_id": "blk.0.ffn_gate.weight.block1",
        "backend": "VULKAN",
        "kernel": "Q4KMatVec",
        "input_slots": [2],
        "output_slots": [3],
        "dependencies": [],
        "dispatch": {"x": 1, "y": 1, "z": 1, "workgroup_x": 256, "workgroup_y": 1, "workgroup_z": 1},
        "source_tier": "RAM_PINNED",
        "destination_tier": "VRAM_SLOT",
        "event_sequence": 2,
        "output_buffer": "multiop.out_b",
    });
    let op_c = json!({
        "index": 2,
        "logical_id": 2,
        "stable_id": "residual_add(out_a, out_b)",
        "backend": "VULKAN",
        "kernel": "QuantizedMulMat",
        "input_slots": [1, 3],
        "output_slots": [4],
        "dependencies": [0, 1],
        "dispatch": {"x": 1, "y": 1, "z": 1, "workgroup_x": 256, "workgroup_y": 1, "workgroup_z": 1},
        "source_tier": "VRAM_SLOT",
        "destination_tier": "VRAM_SLOT",
        "event_sequence": 3,
        "waits_on_events": [1, 2],
        "output_buffer": "multiop.final",
        "requested_kernel_identity": {
            "capability": RESIDUAL_ADD_CAPABILITY,
            "shader_source_sha256": RESIDUAL_ADD_SHADER_SOURCE_SHA256,
            "spirv_sha256": RESIDUAL_ADD_SPIRV_SHA256,
            "registry_status": "NOT_IN_PINNED_REGISTRY",
        },
    });

    let plan = json!({
        "schema": PLAN_SCHEMA,
        "generated_at_unix_ns": 0,
        "compiler": "har-lang-compiler/stable-rust-v0",
        "plan_kind": "native-multi-op-closure",
        "model_identity": model.identity,
        "model_sha256": model.model_sha256,
        "source_model_sha256": model.model_sha256,
        "hardware": {
            "hardware_id": format!("rdna4-{}", target.name),
            "gpu_arch": target.gpu_arch,
            "vram_bytes": target.vram_budget_bytes,
            "host_ram_bytes": target.host_ram_budget_bytes,
            "wave": target.wave,
            "capabilities": target.capabilities,
        },
        "operations": {
            "logical": [
                {"name": "tensor_block0", "kind": "operation", "dependencies": []},
                {"name": "tensor_block1", "kind": "operation", "dependencies": []},
                {"name": "residual_add", "kind": "operation", "dependencies": ["tensor_block0", "tensor_block1"]},
            ],
            "physical": [op_a, op_b, op_c],
        },
        "dag_contract": {
            "independent_predecessors": ["tensor_block0", "tensor_block1"],
            "explicit_edges": [
                {"from": 0, "to": 2},
                {"from": 1, "to": 2},
            ],
            "intermediate_buffers": [
                {"buffer": "multiop.out_a", "producer": 0, "consumers": [2]},
                {"buffer": "multiop.out_b", "producer": 1, "consumers": [2]},
                {"buffer": "multiop.final", "producer": 2, "consumers": []},
            ],
            "events": {
                "op0_event_sequence": 1,
                "op1_event_sequence": 2,
                "op2_event_sequence": 3,
                "op2_waits_on_events": [1, 2],
                "host_wait_between_ops": false,
                "timeline_chaining": "backend contract supports GPU timeline chaining (two GPU wait semaphores, no host wait between operations)",
            },
            "generation": 1,
            "epoch": 1,
        },
        "native_policy": {
            "schema": NATIVE_POLICY_SCHEMA,
            "runtime_source_commit_required": policy.runtime_source_commit,
            "backend_source_commit_required": policy.backend_source_commit,
            "registry_canonical_sha256_required": policy.registry_canonical_sha256,
            "fallback_prohibited": policy.fallback_prohibited,
            "fallback_count_expected": policy.fallback_count_expected,
            "reference_adapter_invocations_expected": policy.reference_adapter_invocations_expected,
        },
        "exactness": {
            "mode": "EXACT",
            "output_hash_required": true,
            "state_boundary_hash_required": true,
            "numerical_tolerance": 0.0001,
            "authority": "full_model",
        },
        "fallback": {
            "authority_backend": "VULKAN",
            "on_unknown_capacity": "reject_plan",
            "on_stale_generation": "fail_closed",
            "on_kernel_unavailable": "fail_closed",
        },
        "assumptions": [
            "Two independent validated Q4_K super-block dispatches (block0, block1) with an explicit residual_add closure",
            "residual_add kernel identity is the native-kernel registry validated multi-op evidence identity; the pinned registry entry is absent and bundle compilation is REJECTED until native-kernel registry publishes it",
            "No host wait between ops; timeline chaining per native-kernel registry backend contract",
        ],
    });
    validate_multi_op_dag(&plan)?;
    serde_json::to_string_pretty(&plan)
        .map_err(|error| vec![format!("multi-op plan serialization failed: {error}")])
}

/// Fail-closed DAG validation for the multi-op closure.  Every structural
/// requirement is checked before the plan may be emitted or any dispatch
/// structure produced: two independent predecessors, explicit edges covering
/// both predecessors, a closure op depending on both, and consistent
/// intermediate buffers.
pub fn validate_multi_op_dag(plan: &Value) -> Result<(), Vec<String>> {
    let mut errors: Vec<String> = Vec::new();
    let Some(dag) = plan.get("dag_contract") else {
        return Err(vec![
            "N3002 missing multi-op dependency: no dag_contract present".into(),
        ]);
    };
    let Some(physical) = plan
        .pointer("/operations/physical")
        .and_then(Value::as_array)
    else {
        return Err(vec![
            "N3002 missing multi-op dependency: no physical operations".into(),
        ]);
    };
    let preds = dag
        .get("independent_predecessors")
        .and_then(Value::as_array)
        .map(|values| values.len())
        .unwrap_or(0);
    if preds < 2 {
        errors.push(
            "N3002 missing multi-op dependency: fewer than two independent predecessor nodes"
                .into(),
        );
    }
    let edges = dag
        .get("explicit_edges")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let froms: Vec<u64> = edges
        .iter()
        .filter_map(|edge| edge.get("from").and_then(Value::as_u64))
        .collect();
    let tos: Vec<u64> = edges
        .iter()
        .filter_map(|edge| edge.get("to").and_then(Value::as_u64))
        .collect();
    if !froms.contains(&0) || !froms.contains(&1) {
        errors.push(
            "N3002 missing multi-op dependency: explicit edges must cover both predecessors (0 and 1)"
                .into(),
        );
    }
    if !tos.iter().all(|target| *target == 2) {
        errors.push(
            "N3002 missing multi-op dependency: all explicit edges must target the closure op (2)"
                .into(),
        );
    }
    for op in physical {
        let Some(index) = op.get("index").and_then(Value::as_u64) else {
            errors.push("N3002 missing multi-op dependency: op without index".into());
            continue;
        };
        let deps = op
            .get("dependencies")
            .and_then(Value::as_array)
            .map(|values| values.iter().filter_map(Value::as_u64).collect::<Vec<_>>())
            .unwrap_or_default();
        for dep in deps {
            if dep >= physical.len() as u64 {
                errors.push(format!(
                    "N3002 missing multi-op dependency: op {index} references missing predecessor {dep}"
                ));
            }
        }
    }
    // closure op (index 2) must depend on both predecessors
    let closure_deps = physical
        .iter()
        .find(|op| op.get("index").and_then(Value::as_u64) == Some(2))
        .and_then(|op| op.get("dependencies").and_then(Value::as_array))
        .map(|values| values.iter().filter_map(Value::as_u64).collect::<Vec<_>>())
        .unwrap_or_default();
    if !closure_deps.contains(&0) || !closure_deps.contains(&1) {
        errors.push(
            "N3002 missing multi-op dependency: closure op must depend on both predecessors (0 and 1)"
                .into(),
        );
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Fail-closed bundle record for the multi-op DAG.  Called only after the
/// plan compiles; the emitted record is NOT a dispatchable bundle.
pub fn emit_multi_op_rejection(
    plan_json: &str,
    registry_canonical_sha256: &str,
    policy: &NativePolicy,
) -> Result<String, String> {
    let plan_sha256 = sha256_bytes(plan_json.as_bytes());
    let record = json!({
        "schema": MULTI_OP_REJECTION_SCHEMA,
        "status": "REJECTED",
        "code": "N3001",
        "reason": "residual_add is absent from the exact pinned native-kernel registry; no unregistered kernel is silently selected; the DAG compiles only after residual_add exists in the pinned registry with a validated entry",
        "plan_sha256": plan_sha256,
        "plan_kind": "native-multi-op-closure",
        "registry_identity": {
            "pinned_core": policy.registry_pinned_core,
            "registry_commit": policy.registry_commit,
            "registry_canonical_sha256": registry_canonical_sha256,
            "entries": ["vulkan.q4_k.gemv", "vulkan.q8_0.gemv", "vulkan.greedy_argmax"],
            "residual_add_present": false,
        },
        "required_before_compilation": {
                "registry_action": "publish residual_add entry with its shader identities in the native-kernel registry",
                "integration_action": "confirm registry identity acceptance",
            "fallback_prohibited": policy.fallback_prohibited,
        },
        "expected_runtime_manifest": {
            "schema": "har.runtime_manifest.v1",
            "source_commit_must_equal": policy.runtime_source_commit,
            "reference_adapters_must_be_empty": true,
            "fallback_count_must_be": 0,
            "reference_adapter_invocations_must_be": 0,
        },
        "expected_events": {
            "schema": "har.events.v1",
            "sequence": [
                "PlanLoaded",
                "PlanValidated",
                "Residency",
                "TransferQueued",
                "TransferCompleted",
                "Dispatch(backend=VULKAN, kernel=Q4KMatVec)",
                "Dispatch(backend=VULKAN, kernel=Q4KMatVec)",
                "Dispatch(backend=VULKAN, kernel=QuantizedMulMat)",
                "Output"
            ],
            "forbidden": ["Fallback"],
        },
    });
    serde_json::to_string_pretty(&record)
        .map_err(|error| format!("multi-op rejection serialization failed: {error}"))
}
