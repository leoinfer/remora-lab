//! First-executable-operation compiler pass.
//!
//! Consumes, read-only, the pinned producer interfaces:
//! - package builder packed package manifest (`har.packed_model_package.v0`);
//! - native-kernel registry RDNA4 kernel registry (`har.rdna4.kernel-registry.v1`);
//! - compiler `har_ir::PhysicalOperation` / `har_core` contracts;
//! - residency layer unified page-store request schema (`har.residency.unified`).
//!
//! Emits a canonical first-operation bundle whose physical operation is
//! directly executable through the runtime chain without parsing HAR source.

use crate::v0::CompiledProgram;
use har_core::{sha256_bytes, BackendKind, KernelKind, MemoryTier};
use har_ir::{DispatchShape, PhysicalOperation};
use har_lang_semantics::v0::{KernelRequirement, QuantFormat};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

pub const FIRST_OPERATION_BUNDLE_SCHEMA: &str = "har.first-operation-bundle.v1";
pub const PAGE_REQUEST_SCHEMA: &str = "har.page-request.v1";
pub const PACKED_MANIFEST_SCHEMA: &str = "har.packed_model_package.v0";
pub const KERNEL_REGISTRY_SCHEMA: &str = "har.rdna4.kernel-registry.v1";

// ---------------------------------------------------------------------------
// Pinned producer data (read-only)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackedEntry {
    pub source_tensor_id: String,
    pub quant_format: String,
    pub payload_location_id: String,
    pub kernel_requirement: String,
    pub representation_identity: String,
    pub dimensions: Vec<u64>,
    pub source_offset: u64,
    pub source_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PayloadLocation {
    pub id: String,
    pub offset: u64,
    pub bytes: u64,
    pub alignment: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackedManifest {
    pub schema: String,
    pub source_root_sha256: String,
    pub required_kernels: Vec<String>,
    pub quality_claim: String,
    pub entries: Vec<PackedEntry>,
    pub locations: Vec<PayloadLocation>,
}

impl PackedManifest {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let bytes =
            fs::read(path).map_err(|error| format!("cannot read packed manifest: {error}"))?;
        let root: Value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("packed manifest is not JSON: {error}"))?;
        if root.get("schema").and_then(Value::as_str) != Some(PACKED_MANIFEST_SCHEMA) {
            return Err("packed manifest schema is not har.packed_model_package.v0".into());
        }
        let source_root = root
            .pointer("/source_model/root_sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| "packed manifest has no source_model.root_sha256".to_string())?;
        let mut entries = Vec::new();
        for entry in root
            .get("packed_entries")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
        {
            let get = |key: &str| {
                entry
                    .get(key)
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string()
            };
            let dims = entry
                .get("dimensions")
                .and_then(Value::as_array)
                .map(|values| values.iter().filter_map(Value::as_u64).collect::<Vec<_>>())
                .unwrap_or_default();
            entries.push(PackedEntry {
                source_tensor_id: get("source_tensor_id"),
                quant_format: get("quant_format"),
                payload_location_id: get("payload_location_id"),
                kernel_requirement: get("kernel_requirement"),
                representation_identity: get("representation_identity"),
                dimensions: dims,
                source_offset: entry
                    .get("source_offset")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                source_bytes: entry
                    .get("source_bytes")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
            });
        }
        let mut locations = Vec::new();
        for location in root
            .get("payload_locations")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
        {
            locations.push(PayloadLocation {
                id: location
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                offset: location.get("offset").and_then(Value::as_u64).unwrap_or(0),
                bytes: location.get("bytes").and_then(Value::as_u64).unwrap_or(0),
                alignment: location
                    .get("alignment")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                sha256: location
                    .get("sha256")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            });
        }
        let quality_claim = root
            .pointer("/claims/quality_claim")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let required_kernels = root
            .get("required_kernels")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self {
            schema: PACKED_MANIFEST_SCHEMA.into(),
            source_root_sha256: source_root.to_string(),
            required_kernels,
            quality_claim,
            entries,
            locations,
        })
    }

    pub fn entry(&self, id: &str) -> Option<&PackedEntry> {
        self.entries
            .iter()
            .find(|entry| entry.source_tensor_id == id)
    }
    pub fn location(&self, id: &str) -> Option<&PayloadLocation> {
        self.locations.iter().find(|location| location.id == id)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelRegistryEntry {
    pub capability: String,
    pub kernel_kind: String,
    pub quant_format: String,
    pub supported_shapes: Vec<String>,
    pub alignment_bytes: u64,
    pub input_datatype: String,
    pub output_datatype: String,
    pub required_vulkan_features: Vec<String>,
    pub workgroup_local: Vec<u32>,
    pub shader_source: String,
    pub shader_source_sha256: String,
    pub spirv: String,
    pub spirv_sha256: String,
    pub numerical_contract: String,
    pub status: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KernelRegistry {
    pub schema: String,
    pub pinned_core: String,
    pub kernels: Vec<KernelRegistryEntry>,
}

impl KernelRegistry {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let bytes =
            fs::read(path).map_err(|error| format!("cannot read kernel registry: {error}"))?;
        let root: Value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("kernel registry is not JSON: {error}"))?;
        if root.get("schema").and_then(Value::as_str) != Some(KERNEL_REGISTRY_SCHEMA) {
            return Err("kernel registry schema is not har.rdna4.kernel-registry.v1".into());
        }
        let pinned_core = root
            .get("pinned_core")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let mut kernels = Vec::new();
        for kernel in root
            .get("kernels")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
        {
            let get = |key: &str| {
                kernel
                    .get(key)
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string()
            };
            let shapes = kernel
                .get("supported_shapes")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default();
            let features = kernel
                .get("required_vulkan_features")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default();
            let local = kernel
                .pointer("/workgroup_geometry/local")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_u64)
                        .map(|v| v as u32)
                        .collect()
                })
                .unwrap_or_default();
            kernels.push(KernelRegistryEntry {
                capability: get("capability"),
                kernel_kind: get("kernel_kind"),
                quant_format: get("quant_format"),
                supported_shapes: shapes,
                alignment_bytes: kernel
                    .get("alignment_bytes")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                input_datatype: get("input_datatype"),
                output_datatype: get("output_datatype"),
                required_vulkan_features: features,
                workgroup_local: local,
                shader_source: get("shader_source"),
                shader_source_sha256: get("shader_source_sha256"),
                spirv: get("spirv"),
                spirv_sha256: get("spirv_sha256"),
                numerical_contract: get("numerical_contract"),
                status: get("status"),
            });
        }
        Ok(Self {
            schema: KERNEL_REGISTRY_SCHEMA.into(),
            pinned_core,
            kernels,
        })
    }

    pub fn kernel(&self, capability: &str) -> Option<&KernelRegistryEntry> {
        self.kernels
            .iter()
            .find(|kernel| kernel.capability == capability)
    }
}

/// Canonical kernel capability for a language kernel requirement (native-kernel registry
/// registry names).
pub fn registry_capability(kernel: KernelRequirement) -> &'static str {
    match kernel {
        KernelRequirement::Q4KMatVec => "vulkan.q4_k.gemv",
        KernelRequirement::QuantizedMulMat => "vulkan.q8_0.gemv",
        KernelRequirement::Sampling => "vulkan.greedy_argmax",
        _ => "unsupported",
    }
}

fn kernel_kind_name(kernel: KernelRequirement) -> KernelKind {
    match kernel {
        KernelRequirement::Q4KMatVec => KernelKind::Q4KMatVec,
        KernelRequirement::QuantizedMulMat => KernelKind::QuantizedMulMat,
        KernelRequirement::Sampling => KernelKind::Sampling,
        KernelRequirement::Attention => KernelKind::Attention,
        KernelRequirement::MtpVerify => KernelKind::MtpVerify,
        KernelRequirement::EmbeddingLookup => KernelKind::EmbeddingLookup,
        KernelRequirement::Cpu => KernelKind::DenseMulMat,
        KernelRequirement::Vulkan => KernelKind::QuantizedMulMat,
    }
}

fn quant_name(format: QuantFormat) -> &'static str {
    match format {
        QuantFormat::Q4KS => "Q4_K",
        QuantFormat::Q4KM => "Q4_K_M",
        QuantFormat::Q8_0 => "Q8_0",
        QuantFormat::F16 => "F16",
        QuantFormat::F32 => "F32",
    }
}

// ---------------------------------------------------------------------------
// Resolved binding
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct OperationBinding {
    pub operation_index: usize,
    pub entry: PackedEntry,
    pub location: PayloadLocation,
    pub kernel_entry: KernelRegistryEntry,
    pub physical: PhysicalOperation,
    pub page_request: Value,
    pub block_sha256: String,
    pub block_bytes: u64,
}

/// Resolve and cross-boundary-validate one `operation` declaration.
pub fn bind_operation(
    program: &CompiledProgram,
    manifest: &PackedManifest,
    registry: &KernelRegistry,
    block_sha256: &str,
    block_bytes: u64,
) -> Result<OperationBinding, Vec<String>> {
    let mut errors: Vec<String> = Vec::new();
    let op = &program.typed.operations[0];
    let model = &program.typed.model;
    let target = &program.typed.target;

    // 7. model-root mismatch
    if op.model_root != model.model_sha256 {
        errors.push(format!(
            "S1057 model-root mismatch: operation roots to {} but model identity is {}",
            op.model_root, model.model_sha256
        ));
    }
    // 1. packed package root mismatch
    if op.package_root != manifest.source_root_sha256 {
        errors.push(format!(
            "S1064 packed package root mismatch: operation declares {} but packed package root is {}",
            op.package_root, manifest.source_root_sha256
        ));
    }
    if op.package_root != model.model_sha256 {
        errors.push(format!(
            "S1064 packed package root mismatch: package root {} differs from model root {}",
            op.package_root, model.model_sha256
        ));
    }
    // 2. packed entry not present
    let Some(entry) = manifest.entry(&op.source_tensor) else {
        errors.push(format!(
            "S1065 packed entry not present: {} is not in the packed manifest (payload_entry {})",
            op.source_tensor, op.payload_entry
        ));
        return Err(errors);
    };
    // 3. source tensor mismatch
    if op.payload_entry != entry.payload_location_id {
        errors.push(format!(
            "S1066 source tensor mismatch: payload_entry {} does not match packed entry location {}",
            op.payload_entry, entry.payload_location_id
        ));
    }
    if op.source_tensor != entry.source_tensor_id {
        errors.push(format!(
            "S1066 source tensor mismatch: source_tensor {} does not match packed entry {}",
            op.source_tensor, entry.source_tensor_id
        ));
    }
    // 4. quant/kernel incompatibility
    let requested_quant = quant_name(op.quant_format);
    if !entry.quant_format.eq_ignore_ascii_case(requested_quant) {
        errors.push(format!(
            "S1067 quant/kernel incompatibility: operation quant {} does not match packed entry quant {}",
            requested_quant, entry.quant_format
        ));
    }
    let capability = registry_capability(op.kernel);
    if capability == "unsupported" {
        errors.push("S1067 quant/kernel incompatibility: kernel has no registry capability".into());
    }
    if entry.kernel_requirement != "har.vulkan.gemm.q4_k"
        && op.kernel == KernelRequirement::Q4KMatVec
    {
        errors.push(format!(
            "S1067 quant/kernel incompatibility: packed entry requires {} but operation selects q4_k_matvec",
            entry.kernel_requirement
        ));
    }
    // 13. kernel not supported by hardware phenotype
    let required_capability = kernel_kind_name(op.kernel).capability();
    if !target
        .capabilities
        .iter()
        .any(|capability| capability == required_capability)
    {
        errors.push(format!(
            "S1073 kernel not supported by hardware phenotype: target lacks capability {}",
            required_capability
        ));
    }
    if !target
        .capabilities
        .iter()
        .any(|capability| capability == "vulkan")
    {
        errors.push(
            "S1073 kernel not supported by hardware phenotype: target lacks vulkan capability"
                .into(),
        );
    }
    // 5. unsupported shape (validated registry contract: rows=1, blocks_per_row=1)
    if op.rows != 1 || op.columns != 256 {
        errors.push(format!(
            "S1063 unsupported shape: rows={} columns={} is outside the validated kernel contract (rows=1, blocks_per_row=1 => 256 columns)",
            op.rows, op.columns
        ));
    }
    // kernel registry entry must exist and be VALIDATED
    let Some(kernel_entry) = registry.kernel(capability) else {
        errors.push(format!(
            "S1073 kernel not supported by hardware phenotype: registry has no validated entry for {}",
            capability
        ));
        return Err(errors);
    };
    if kernel_entry.status != "VALIDATED" {
        errors.push(format!(
            "S1073 kernel not supported by hardware phenotype: registry entry {} status is {}",
            capability, kernel_entry.status
        ));
    }
    // 6. alignment mismatch
    if op.alignment < kernel_entry.alignment_bytes {
        errors.push(format!(
            "S1068 alignment mismatch: operation alignment {} is below registry alignment {}",
            op.alignment, kernel_entry.alignment_bytes
        ));
    }
    let Some(location) = manifest.location(&entry.payload_location_id) else {
        errors.push(format!(
            "S1065 packed entry not present: no payload location for {}",
            entry.payload_location_id
        ));
        return Err(errors);
    };
    if op.alignment < location.alignment {
        errors.push(format!(
            "S1068 alignment mismatch: operation alignment {} is below packed payload alignment {}",
            op.alignment, location.alignment
        ));
    }
    // 8. wrong representation identity
    if entry.representation_identity != "byte-identical-source" {
        errors.push(format!(
            "S1069 wrong representation identity: packed entry is {} but exact payload contract requires byte-identical-source",
            entry.representation_identity
        ));
    }
    // 12. approximate package entry under exact payload contract
    if manifest.quality_claim != "blocked" && manifest.quality_claim != "exact" {
        errors.push(format!(
            "S1072 approximate package entry used under exact payload contract: package quality claim is {}",
            manifest.quality_claim
        ));
    }
    // 9. missing checksum contract
    if !op.require_exact_checksum {
        errors.push(
            "S1070 missing checksum contract: operation must require exact_payload_checksum".into(),
        );
    }
    // 11. missing fallback
    if op.fallback.is_empty() {
        errors.push("S1071 missing fallback: operation must declare a fallback policy".into());
    }
    if !matches!(op.fallback.as_str(), "depth_zero" | "exact_target") {
        errors.push(format!(
            "S1071 missing fallback: unsupported fallback policy `{}`",
            op.fallback
        ));
    }
    // 10. stale generation
    if op.generation.0 != 0 && op.generation.0 != model.epoch.0 {
        errors.push(format!(
            "S1055 stale generation: operation generation {} but model epoch is {}",
            op.generation.0, model.epoch.0
        ));
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    let physical = PhysicalOperation {
        index: 0,
        logical_id: 0,
        stable_id: entry.source_tensor_id.clone(),
        backend: BackendKind::Vulkan,
        kernel: kernel_kind_name(op.kernel),
        input_slots: vec![0],
        output_slots: vec![1],
        dependencies: vec![],
        dispatch: DispatchShape {
            x: 1,
            y: 1,
            z: 1,
            workgroup_x: kernel_entry.workgroup_local.first().copied().unwrap_or(256),
            workgroup_y: kernel_entry.workgroup_local.get(1).copied().unwrap_or(1),
            workgroup_z: kernel_entry.workgroup_local.get(2).copied().unwrap_or(1),
        },
        source_tier: MemoryTier::RamPinned,
        destination_tier: MemoryTier::VramSlot,
    };

    // residency layer unified page-store request (pinned schema; JSON binding).
    let page_request = json!({
        "schema": PAGE_REQUEST_SCHEMA,
        "page_id": {
            "logical_object": entry.payload_location_id,
            "representation": {
                "page_id": {
                    "model_root": model.model_sha256,
                    "kind": "WEIGHTS",
                    "ordinal": 0,
                },
                "format": requested_quant,
                "bytes": block_bytes,
            },
            "model_root": model.model_sha256,
            "payload_class": "WEIGHT_TILE",
        },
        "preferred_location": "RAM_PINNED",
        "mandatory": true,
        "speculative": false,
        "expect_checksum": block_sha256,
        "source_location": "NVME_COLD",
        "destination_location": "VRAM_PAGE(0)",
        "storage_slice": {
            "tensor": entry.source_tensor_id,
            "payload_location_id": entry.payload_location_id,
            "offset": location.offset,
            "block_offset": 0,
            "payload_bytes": block_bytes,
            "entry_bytes": location.bytes,
            "entry_sha256": location.sha256,
        },
        "epoch": model.epoch.0,
        "generation": op.generation.0,
    });

    Ok(OperationBinding {
        operation_index: 0,
        entry: entry.clone(),
        location: location.clone(),
        kernel_entry: kernel_entry.clone(),
        physical,
        page_request,
        block_sha256: block_sha256.to_string(),
        block_bytes,
    })
}

/// Emit the compiler loadable single-operation package
/// (`har.model_compiled_package.v0`).
pub fn emit_single_operation_package(
    program: &CompiledProgram,
    binding: &OperationBinding,
) -> Result<String, String> {
    let typed = &program.typed;
    let target = &typed.target;
    let model = &typed.model;
    let quality = &typed.quality;
    let decode = &typed.decode;

    let package = json!({
        "schema": "har.model_compiled_package.v0",
        "generated_at_unix_ns": 0,
        "compiler": "har-lang-compiler/stable-rust-v0",
        "plan_kind": "first-executable-operation",
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
        "tensors": [{
            "tensor_id": binding.entry.source_tensor_id,
            "name": binding.entry.source_tensor_id,
            "bytes": binding.location.bytes,
            "planned_memory_tier": "vram_slot",
            "storage_location": "nvme_cold",
            "quant_format": binding.entry.quant_format,
            "supported_kernels": binding.entry.quant_format,
            "alignment": binding.location.alignment,
            "role": "FEED_FORWARD",
            "is_mtp": false,
            "generation": typed.operations[0].generation.0,
            "model_root": model.model_sha256,
        }],
        "storage_plan": {
            "tiers": {
                "vram": target.vram_budget_bytes,
                "nvme": binding.location.bytes,
                "ram": target.host_ram_budget_bytes,
            }
        },
        "required_kernels": [kernel_kind_name(typed.operations[0].kernel).capability()],
        "required_capabilities": target.capabilities,
        "exactness": {
            "mode": "EXACT",
            "output_hash_required": true,
            "state_boundary_hash_required": true,
            "numerical_tolerance": typed.operations[0].reference_tolerance,
            "authority": quality.authority,
        },
        "fallback": {
            "authority_backend": "VULKAN",
            "on_unknown_capacity": "reject_plan",
            "on_stale_generation": "fail_closed",
            "on_kernel_unavailable": "reject_plan",
        },
        "telemetry": {
            "schema": "har.telemetry.v1",
            "record_residency": true,
            "record_transfers": true,
            "record_operation_hashes": true,
            "timing_is_advisory": true,
        },
        "decode_policy": {
            "name": decode.name,
            "min_horizon": decode.min_horizon,
            "max_horizon": decode.max_horizon,
            "mode": if decode.min_horizon == decode.max_horizon { "fixed" } else { "elastic" },
            "objective": decode.objective,
            "require_exact_acceptance": decode.require_exact_acceptance,
            "sampling_location": decode.sampling_location,
            "sampling_policy": decode.sampling_policy,
            "placement": decode.placement,
            "fallback": decode.fallback,
        },
        "runtime_config": {
            "context_length": model.context_length,
            "batch": model.batch,
            "kv_datatype": if model.kv_cache_type.is_empty() {
                "f16".to_string()
            } else {
                model.kv_cache_type.clone()
            },
            "kv_cache_type": model.kv_cache_type,
            "mtp_enabled": model.nextn_predict_layers > 0,
            "nextn_predict_layers": model.nextn_predict_layers,
            "size_gib": model.size_gib,
        },
        "model_roots": {
            "identity": model.identity,
            "model_sha256": model.model_sha256,
            "build_hash": model.build_hash,
            "config_hash": model.config_hash,
            "epoch": model.epoch.0,
        },
        "hardware_root": {
            "gpu": target.gpu,
            "gpu_arch": target.gpu_arch,
            "wave": target.wave,
            "vram_budget_bytes": target.vram_budget_bytes,
            "host_ram_budget_bytes": target.host_ram_budget_bytes,
        },
        "operation": {
            "name": typed.operations[0].name,
            "source_tensor": typed.operations[0].source_tensor,
            "payload_entry": typed.operations[0].payload_entry,
            "rows": typed.operations[0].rows,
            "columns": typed.operations[0].columns,
            "alignment": typed.operations[0].alignment,
            "reference_tolerance": typed.operations[0].reference_tolerance,
            "fallback": typed.operations[0].fallback,
        },
        "assumptions": [
            "Single validated Q4_K super-block operation; no full-model claim",
            "Exact identities are bound at compile time from pinned producer interfaces",
        ],
        "unresolved_risks": [],
    });
    serde_json::to_string_pretty(&package)
        .map_err(|error| format!("plan serialization failed: {error}"))
}

/// Emit the canonical first-operation bundle.
#[allow(clippy::too_many_arguments)]
pub fn emit_first_operation_bundle(
    program: &CompiledProgram,
    binding: &OperationBinding,
    plan_json: &str,
    idea_traceability: &Value,
    source_sha256: &str,
    block_sha256: &str,
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
        sha256_bytes(&serde_json::to_vec(&physical_value).map_err(|error| error.to_string())?);
    let plan_hash = sha256_bytes(plan_json.as_bytes());

    let bundle = json!({
        "schema": FIRST_OPERATION_BUNDLE_SCHEMA,
        "producer": "public-first-executable-operation",
        "source_program": "examples/native_operation.har",
        "source_sha256": source_sha256,
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
            "validated_block_sha256": block_sha256,
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
            "on_kernel_unavailable": "reject_plan",
            "operation_fallback": op.fallback,
        },
        "telemetry_contract": {
            "schema": "har.telemetry.v1",
            "record_residency": true,
            "record_transfers": true,
            "record_operation_hashes": true,
            "timing_is_advisory": true,
        },
        "reference_fixtures": {
            "model_slice": "external model fixture (not distributed)",
            "q4k_kernel": "public synthetic Q4_K fixture",
        },
        "semantic_delta": {
            "requested": {"context_length": model.context_length, "kv_datatype": model.kv_cache_type, "batch": model.batch},
            "plan_loader_defaults": {"context_length": 4096, "kv_datatype": "f16"},
            "context_length_delta": model.context_length != 4096,
            "kv_datatype_delta": !model.kv_cache_type.is_empty() && model.kv_cache_type != "f16",
            "execution_relevant_for_this_operation": false,
            "compatibility": "REQUESTED_SEMANTICS_NOT_PRESERVED_BY_LOADER",
        },
        "idea_traceability": idea_traceability,
        "assumptions": [
            "Single validated Q4_K super-block slice supplied by the caller",
            "Kernel dispatch shape rows=1 blocks_per_row=1 matches native-kernel registry VALIDATED contract",
            "Numerical reference is produced by the canonical Q4_K dequantization adapter at dispatch",
        ],
        "unresolved_risks": [],
    });
    serde_json::to_string_pretty(&bundle)
        .map_err(|error| format!("bundle serialization failed: {error}"))
}

/// Load the canonical idea registry (idea registry) and build the traceability
/// section for the operation lane.
pub fn operation_idea_traceability(
    registry: Option<&BTreeMap<String, crate::idea_registry::IdeaRef>>,
) -> Result<Value, String> {
    let mappings: Vec<(&str, &str, &[&str])> = vec![
        (
            "operation.kernel.q4k_gemv",
            "native RDNA4 Q4_K GEMV execution",
            &["research-hardware-phenotype", "research-native-policy"],
        ),
        (
            "operation.payload_entry",
            "packed model package entry binding",
            &["research-model-package", "research-residency"],
        ),
        (
            "operation.page_request",
            "physical page request",
            &["research-page-request", "research-residency"],
        ),
        (
            "operation.checksum_contract",
            "exact payload checksum contract",
            &["research-exactness", "research-checksum"],
        ),
        (
            "operation.shape_validated",
            "validated kernel shape contract",
            &["research-shape-contract", "research-native-policy"],
        ),
        (
            "operation.fallback",
            "exactness fallback",
            &["research-exactness", "research-native-policy"],
        ),
    ];
    let mut traceability = Vec::new();
    for (construct, purpose, ids) in mappings {
        let mut entries = Vec::new();
        for id in ids {
            match registry {
                Some(registry) => {
                    let idea = registry.get(*id).ok_or_else(|| {
                        format!("canonical idea id {id} is missing from the registry")
                    })?;
                    entries.push(json!({
                        "canonical_id": id,
                        "canonical_title": idea.title,
                        "evidence_state": idea.evidence,
                    }));
                }
                None => entries.push(json!({ "canonical_id": id, "registry_absent": true })),
            }
        }
        traceability.push(json!({
            "language_construct": construct,
            "purpose": purpose,
            "ideas": entries,
        }));
    }
    Ok(Value::Array(traceability))
}

/// Emit the optional two-operation package (block0 -> block1 sequential
/// closure).  Both operations are validated `vulkan.q4_k.gemv` dispatches;
/// residual_add is intentionally not selected because it is absent from the
/// pinned kernel registry.
pub fn emit_two_operation_package(
    program: &CompiledProgram,
    binding0: &OperationBinding,
    binding1: &OperationBinding,
    block1_sha256: &str,
) -> Result<String, String> {
    let typed = &program.typed;
    let target = &typed.target;
    let model = &typed.model;
    let quality = &typed.quality;
    let decode = &typed.decode;

    let package = json!({
        "schema": "har.model_compiled_package.v0",
        "generated_at_unix_ns": 0,
        "compiler": "har-lang-compiler/stable-rust-v0",
        "plan_kind": "first-two-operation-closure",
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
        "tensors": [
            {
                "tensor_id": "blk.0.ffn_gate.weight.block0",
                "name": "blk.0.ffn_gate.weight.block0",
                "bytes": binding0.block_bytes,
                "planned_memory_tier": "vram_slot",
                "storage_location": "nvme_cold",
                "quant_format": "Q4_K",
                "supported_kernels": "Q4_K",
                "alignment": binding0.location.alignment,
                "role": "FEED_FORWARD",
                "is_mtp": false,
                "generation": typed.operations[0].generation.0,
                "model_root": model.model_sha256,
            },
            {
                "tensor_id": "blk.0.ffn_gate.weight.block1",
                "name": "blk.0.ffn_gate.weight.block1",
                "bytes": binding1.block_bytes,
                "planned_memory_tier": "vram_slot",
                "storage_location": "nvme_cold",
                "quant_format": "Q4_K",
                "supported_kernels": "Q4_K",
                "alignment": binding1.location.alignment,
                "role": "FEED_FORWARD",
                "is_mtp": false,
                "generation": typed.operations[1].generation.0,
                "model_root": model.model_sha256,
            },
        ],
        "storage_plan": {
            "tiers": {
                "vram": target.vram_budget_bytes,
                "nvme": binding0.location.bytes,
                "ram": target.host_ram_budget_bytes,
            }
        },
        "required_kernels": ["q4_k_matvec"],
        "required_capabilities": target.capabilities,
        "exactness": {
            "mode": "EXACT",
            "output_hash_required": true,
            "state_boundary_hash_required": true,
            "numerical_tolerance": typed.operations[0].reference_tolerance,
            "authority": quality.authority,
        },
        "fallback": {
            "authority_backend": "VULKAN",
            "on_unknown_capacity": "reject_plan",
            "on_stale_generation": "fail_closed",
            "on_kernel_unavailable": "reject_plan",
        },
        "telemetry": {
            "schema": "har.telemetry.v1",
            "record_residency": true,
            "record_transfers": true,
            "record_operation_hashes": true,
            "timing_is_advisory": true,
        },
        "decode_policy": {
            "name": decode.name,
            "min_horizon": decode.min_horizon,
            "max_horizon": decode.max_horizon,
            "objective": decode.objective,
            "require_exact_acceptance": decode.require_exact_acceptance,
            "sampling_location": decode.sampling_location,
            "sampling_policy": decode.sampling_policy,
            "placement": decode.placement,
            "fallback": decode.fallback,
        },
        "runtime_config": {
            "context_length": model.context_length,
            "batch": model.batch,
            "kv_datatype": if model.kv_cache_type.is_empty() {
                "f16".to_string()
            } else {
                model.kv_cache_type.clone()
            },
            "kv_cache_type": model.kv_cache_type,
            "mtp_enabled": model.nextn_predict_layers > 0,
            "size_gib": model.size_gib,
        },
        "model_roots": {
            "identity": model.identity,
            "model_sha256": model.model_sha256,
            "build_hash": model.build_hash,
            "config_hash": model.config_hash,
            "epoch": model.epoch.0,
        },
        "hardware_root": {
            "gpu": target.gpu,
            "gpu_arch": target.gpu_arch,
            "wave": target.wave,
        },
        "operations": [
            {
                "name": typed.operations[0].name,
                "source_tensor": typed.operations[0].source_tensor,
                "payload_entry": typed.operations[0].payload_entry,
                "block_offset": 0,
                "rows": typed.operations[0].rows,
                "columns": typed.operations[0].columns,
                "block_sha256": binding0.block_sha256,
            },
            {
                "name": typed.operations[1].name,
                "source_tensor": typed.operations[1].source_tensor,
                "payload_entry": typed.operations[1].payload_entry,
                "block_offset": binding1.block_bytes,
                "rows": typed.operations[1].rows,
                "columns": typed.operations[1].columns,
                "block_sha256": block1_sha256,
            },
        ],
        "assumptions": [
            "Two sequential validated Q4_K super-block dispatches; residual_add not selected (absent from pinned kernel registry)",
        ],
        "unresolved_risks": [],
    });
    serde_json::to_string_pretty(&package)
        .map_err(|error| format!("two-op plan serialization failed: {error}"))
}

/// Emit the optional two-operation bundle.
pub fn emit_two_operation_bundle(
    program: &CompiledProgram,
    binding0: &OperationBinding,
    binding1: &OperationBinding,
    plan_json: &str,
    block1_sha256: &str,
) -> Result<String, String> {
    let typed = &program.typed;
    let physical0 = serde_json::to_value(&binding0.physical).map_err(|error| error.to_string())?;
    let mut operation1 = binding1.physical.clone();
    operation1.index = 1;
    operation1.logical_id = 1;
    operation1.stable_id = "blk.0.ffn_gate.weight.block1".into();
    operation1.dependencies = vec![0];
    let physical1_chained = serde_json::to_value(&operation1).map_err(|error| error.to_string())?;
    let bundle = json!({
        "schema": "har.two-operation-bundle.v1",
        "producer": "public-first-executable-operation",
        "source_program": "examples/native_two_operation.har",
        "typed_semantic_ast": {
            "model": typed.model.name,
            "operation_count": typed.operations.len(),
        },
        "logical_operations": [
            {"kind": "operation", "name": typed.operations[0].name, "dependencies": []},
            {"kind": "operation", "name": typed.operations[1].name, "dependencies": [0]},
        ],
        "physical_operations": [physical0, physical1_chained],
        "packed_entry_binding": {
            "source_tensor_id": binding0.entry.source_tensor_id,
            "payload_location_id": binding0.entry.payload_location_id,
            "payload_offset": binding0.location.offset,
            "payload_sha256": binding0.location.sha256,
            "block0_sha256": binding0.block_sha256,
            "block1_sha256": block1_sha256,
        },
        "page_request_bindings": [binding0.page_request, binding1.page_request],
        "model_root": typed.model.model_sha256,
        "package_root": typed.operations[0].package_root,
        "plan_hash": har_core::sha256_bytes(plan_json.as_bytes()),
        "residual_add_status": "NOT_SELECTED: capability is absent from the pinned native-kernel registry; no unregistered kernel is selected",
        "assumptions": [
            "Optional two-operation closure; both dispatches match the validated rows=1 blocks_per_row=1 contract",
        ],
    });
    serde_json::to_string_pretty(&bundle)
        .map_err(|error| format!("two-op bundle serialization failed: {error}"))
}
