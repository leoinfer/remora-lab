//! Immutable execution plans and compatibility loading for plan importer's
//! `har.model_compiled_package.v0` JSON.  The loader copies only descriptors;
//! it never executes source-language text or performs runtime reflection.

use har_core::{
    canonical_json, canonical_sha256, BackendKind, ExactnessContract, ExactnessMode,
    FallbackContract, HarError, HardwarePhenotype, KernelKind, MemoryBudget, MemoryTier,
    QuantFormat, ReasonCode, ResourceBudget, Result, TelemetryContract, TransferPlan,
};
use har_ir::{DispatchShape, OperationTable, PhysicalOperation};
use har_model::ModelPhenotype;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

pub const PLAN_SCHEMA: &str = "har.execution_plan.v1";
pub const MODEL_PACKAGE_SCHEMA: &str = "har.model_compiled_package.v0";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TensorPlacement {
    pub tensor_id: String,
    pub tensor_name: String,
    pub bytes: u64,
    pub source_tier: MemoryTier,
    pub preferred_tier: MemoryTier,
    pub backend: BackendKind,
    pub kernel: KernelKind,
    pub alignment_bytes: u64,
    pub reasons: Vec<ReasonCode>,
    pub dependency_closure: Vec<String>,
    pub explanation: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExecutionPlan {
    pub schema: String,
    pub plan_id: String,
    pub plan_kind: String,
    pub generated_at_unix_ns: u64,
    pub model_identity: String,
    pub model_sha256: String,
    pub hardware: HardwarePhenotype,
    pub target_context: u32,
    pub kv_datatype: String,
    pub mtp_enabled: bool,
    pub quality_policy: String,
    pub budget: ResourceBudget,
    pub tensor_placements: Vec<TensorPlacement>,
    pub transfers: Vec<TransferPlan>,
    pub operations: OperationTable,
    pub required_kernels: Vec<String>,
    pub exactness: ExactnessContract,
    pub fallback: FallbackContract,
    pub telemetry: TelemetryContract,
    pub assumptions: Vec<String>,
    pub unresolved_risks: Vec<String>,
    pub source_model_package_schema: Option<String>,
    pub source_model_package_sha256: Option<String>,
}
impl ExecutionPlan {
    pub fn validate(&self) -> ValidationReport {
        let mut report = ValidationReport::default();
        if self.schema != PLAN_SCHEMA {
            report
                .errors
                .push(format!("unsupported plan schema {}", self.schema));
        }
        if self.plan_id.is_empty() {
            report.errors.push("plan_id is empty".into());
        }
        if self.model_identity.is_empty() {
            report.errors.push("model identity is empty".into());
        }
        if self.model_sha256.is_empty() {
            report
                .warnings
                .push("model_sha256 is empty; identity is not hash-bound".into());
        }
        let mut ids = BTreeSet::new();
        for placement in &self.tensor_placements {
            if !ids.insert(placement.tensor_id.clone()) {
                report.errors.push(format!(
                    "duplicate tensor placement {}",
                    placement.tensor_id
                ));
            }
            if placement.bytes == 0 {
                report
                    .warnings
                    .push(format!("zero-byte placement {}", placement.tensor_id));
            }
        }
        if let Err(error) = self.operations.validate() {
            report.errors.push(error);
        }
        // Unsupported kernels and capabilities fail before execution.  A
        // production plan never substitutes a foreign backend for a missing
        // native Rust kernel.
        if self.operations.physical.is_empty() {
            report
                .warnings
                .push("plan has no physical operations; shape-only plan".into());
        }
        for operation in &self.operations.physical {
            if operation.kernel == KernelKind::Unknown {
                report.errors.push(format!(
                    "operation {} declares UNKNOWN kernel",
                    operation.stable_id
                ));
                continue;
            }
            let capability = operation.kernel.capability();
            let supported = self.hardware.capabilities.supports(capability);
            if !supported {
                report.errors.push(format!(
                    "{} requires capability {capability} that the hardware phenotype does not declare",
                    operation.stable_id
                ));
            }
            if operation.backend == BackendKind::Vulkan
                && !self.hardware.capabilities.supports("vulkan")
            {
                report.errors.push(format!(
                    "{} requires Vulkan backend but hardware has no vulkan capability",
                    operation.stable_id
                ));
            }
        }
        // Resource overcommit fails closed.
        for budget in &self.budget.tiers {
            let assigned = budget.assigned_bytes.saturating_add(budget.reserved_bytes);
            if assigned > budget.capacity_bytes {
                report.errors.push(format!(
                    "memory tier {} overcommitted: assigned+reserved {} > capacity {}",
                    budget.tier, assigned, budget.capacity_bytes
                ));
            }
        }
        if self
            .budget
            .kv_bytes
            .saturating_add(self.budget.staging_bytes)
            .saturating_add(self.budget.scratch_bytes)
            .saturating_add(self.budget.model_bytes)
            > 0
            && self.budget.tiers.is_empty()
        {
            report
                .warnings
                .push("resource budget declares bytes but no tier capacities".into());
        }
        // Approximate outputs cannot silently satisfy exact contracts.
        if self.exactness.mode == ExactnessMode::Approximate && self.exactness.output_hash_required
        {
            report
                .errors
                .push("exactness contract is Approximate but output hash is required".into());
        }
        // Fallback contract must name an authority backend.
        if self.fallback.authority_backend == BackendKind::None {
            report
                .errors
                .push("fallback contract has no authority backend".into());
        }
        for transfer in &self.transfers {
            if transfer.bytes == 0 {
                report
                    .warnings
                    .push(format!("zero-byte transfer {}", transfer.id));
            }
        }
        report
            .warnings
            .extend(self.unresolved_risks.iter().cloned());
        report
    }
    pub fn canonical_bytes(&self) -> Result<Vec<u8>> {
        canonical_json(self)
    }
    pub fn identity_hash(&self) -> Result<String> {
        let mut identity = self.clone();
        identity.generated_at_unix_ns = 0;
        canonical_sha256(&identity)
    }
    pub fn operation_for_tensor(&self, tensor_name: &str) -> Option<&PhysicalOperation> {
        self.operations
            .physical
            .iter()
            .find(|op| op.stable_id == tensor_name || op.stable_id.ends_with(tensor_name))
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidationReport {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}
impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LoadedPlan {
    pub source_path: String,
    pub source_sha256: String,
    pub source_schema: String,
    pub plan: ExecutionPlan,
    pub plan_identity_hash: String,
    pub validation: ValidationReport,
    pub source_root_keys: Vec<String>,
}
impl LoadedPlan {
    pub fn is_executable(&self) -> bool {
        self.validation.is_valid()
    }
}

pub struct PlanLoader;
impl PlanLoader {
    pub fn load(path: impl AsRef<Path>) -> Result<LoadedPlan> {
        let path = path.as_ref();
        let bytes = fs::read(path)?;
        let source_sha256 = har_core::sha256_bytes(&bytes);
        let value: Value = serde_json::from_slice(&bytes)?;
        let object = value.as_object().ok_or_else(|| HarError::Invalid {
            kind: "plan importer plan",
            message: "root must be an object".into(),
        })?;
        let source_schema = string(object, "schema").unwrap_or_else(|| "unknown".into());
        let plan = parse_model_package(object, &source_schema, &source_sha256)?;
        let plan_identity_hash = plan.identity_hash()?;
        let validation = plan.validate();
        let source_root_keys = object.keys().cloned().collect();
        Ok(LoadedPlan {
            source_path: path.display().to_string(),
            source_sha256,
            source_schema,
            plan,
            plan_identity_hash,
            validation,
            source_root_keys,
        })
    }
    pub fn from_model(model: &ModelPhenotype, hardware: HardwarePhenotype) -> ExecutionPlan {
        compile_model_plan(model, hardware)
    }
}

fn parse_model_package(
    root: &serde_json::Map<String, Value>,
    source_schema: &str,
    source_sha256: &str,
) -> Result<ExecutionPlan> {
    let model_identity = string(root, "model_identity")
        .or_else(|| string(root, "model_id"))
        .unwrap_or_else(|| "unknown-model".into());
    let model_sha256 = string(root, "source_model_sha256")
        .or_else(|| string(root, "model_sha256"))
        .unwrap_or_default();
    let hardware = parse_hardware(root.get("hardware"));
    let mut tensor_placements = Vec::new();
    let mut transfers = Vec::new();
    let mut operations = OperationTable::new();
    let tensors = root
        .get("tensors")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let hardware_backend = if hardware.gpu.name.is_empty() {
        BackendKind::Cpu
    } else {
        BackendKind::Vulkan
    };
    for (index, tensor) in tensors.iter().enumerate() {
        let map = tensor.as_object().ok_or_else(|| HarError::Invalid {
            kind: "plan importer tensor",
            message: format!("tensor {index} is not an object"),
        })?;
        let id = string(map, "tensor_id")
            .or_else(|| string(map, "name"))
            .unwrap_or_else(|| format!("tensor.{index}"));
        let name = string(map, "name").unwrap_or_else(|| id.clone());
        let bytes = u64_value(map, "bytes")
            .or_else(|| u64_value(map, "payload_bytes"))
            .unwrap_or(0);
        let planned_tier = string(map, "planned_memory_tier")
            .or_else(|| string(map, "memory_tier"))
            .unwrap_or_else(|| "ram".into());
        let preferred = parse_tier(&planned_tier);
        let source = parse_tier(
            string(map, "storage_location")
                .as_deref()
                .unwrap_or("model"),
        );
        let quant = string(map, "quant_format")
            .or_else(|| string(map, "quantization"))
            .unwrap_or_else(|| "unknown".into());
        let _role = string(map, "role").unwrap_or_default();
        let kernel = kernel_from_format(
            string(map, "supported_kernels")
                .unwrap_or_else(|| quant.clone())
                .as_str(),
        );
        let alignment = u64_value(map, "alignment")
            .or_else(|| u64_value(map, "alignment_bytes"))
            .unwrap_or(256);
        let mut reasons = vec![ReasonCode::ModelPackageImported];
        if preferred == MemoryTier::VramResident || preferred == MemoryTier::VramSlot {
            reasons.push(ReasonCode::FitsVram);
        }
        tensor_placements.push(TensorPlacement {
            tensor_id: id.clone(),
            tensor_name: name.clone(),
            bytes,
            source_tier: source.clone(),
            preferred_tier: preferred.clone(),
            backend: hardware_backend.clone(),
            kernel: kernel.clone(),
            alignment_bytes: alignment,
            reasons,
            dependency_closure: vec![id.clone()],
            explanation:
                "Imported from plan importer compiled package; runtime does not infer missing policy"
                    .into(),
        });
        if source != preferred {
            transfers.push(TransferPlan {
                id: format!("transfer.{index}"),
                resource_id: id.clone(),
                source: source.clone(),
                destination: preferred.clone(),
                bytes,
                alignment_bytes: alignment,
                staging_required: preferred == MemoryTier::VramResident
                    || preferred == MemoryTier::VramSlot,
                queue: if preferred == MemoryTier::VramResident || preferred == MemoryTier::VramSlot
                {
                    "transfer".into()
                } else {
                    "host".into()
                },
                dependency: None,
            });
        }
        let physical = PhysicalOperation {
            index: operations.physical.len() as u32,
            logical_id: operations.physical.len() as u32,
            stable_id: name.clone(),
            backend: hardware_backend.clone(),
            kernel,
            input_slots: vec![],
            output_slots: vec![],
            dependencies: if operations.physical.is_empty() {
                vec![]
            } else {
                vec![(operations.physical.len() - 1) as u32]
            },
            dispatch: DispatchShape {
                x: 1,
                y: 1,
                z: 1,
                workgroup_x: 64,
                workgroup_y: 1,
                workgroup_z: 1,
            },
            source_tier: source,
            destination_tier: preferred,
        };
        operations.physical.push(physical);
    }
    let budget = parse_budget(root.get("storage_plan"), &hardware);
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
    let unresolved_risks = root
        .get("unresolved_risks")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let quality_policy = string(root, "compiler").unwrap_or_else(|| "model-package-import".into());
    // format-compatibility-v1 (compiler decision ACCEPTED_WITH_CHANGES, review note): the plan's
    // runtime_config is honored when present; loader defaults are applied only
    // when absent and are then recorded as explicit deltas.  Silent default
    // mutation is rejected by contract.  kv_datatype is canonicalized to
    // lowercase (both "Q8_0" and "q8_0" load); unknown values are hard-rejected;
    // zero context_length is rejected.
    let runtime_config = root.get("runtime_config").and_then(Value::as_object);
    let target_context = runtime_config
        .and_then(|rc| rc.get("context_length").and_then(Value::as_u64))
        .map(|v| v as u32)
        .unwrap_or(4096);
    if target_context == 0 {
        return Err(HarError::Invalid {
            kind: "runtime_config.context_length",
            message: "context_length=0 is invalid; the loader never applies a zero default".into(),
        });
    }
    let kv_datatype = runtime_config
        .and_then(|rc| rc.get("kv_datatype").and_then(Value::as_str))
        .unwrap_or("f16")
        .to_string()
        .to_ascii_lowercase();
    if !matches!(
        kv_datatype.as_str(),
        "f16" | "f32" | "bf16" | "q8_0" | "q4_0" | "q4_k" | "q8_k"
    ) {
        return Err(HarError::Invalid { kind: "runtime_config.kv_datatype", message: format!("unsupported kv_datatype {kv_datatype:?}; must be one of f16/f32/bf16/q8_0/q4_0/q4_k/q8_k") });
    }
    let mtp_enabled = runtime_config
        .and_then(|rc| rc.get("mtp_enabled").and_then(Value::as_bool))
        .unwrap_or(true);
    let context_delta = match runtime_config { Some(rc) if rc.contains_key("context_length") => format!("context_length={target_context} honored from plan runtime_config"), _ => "runtime_config absent; loader default target_context=4096 applied explicitly (no silent mutation)".into() };
    let kv_delta = match runtime_config { Some(rc) if rc.contains_key("kv_datatype") => format!("kv_datatype={kv_datatype} honored from plan runtime_config"), _ => "runtime_config absent; loader default kv_datatype=f16 applied explicitly (no silent mutation)".into() };
    let plan = ExecutionPlan {
        schema: PLAN_SCHEMA.into(),
        plan_id: format!(
            "model-package.{}",
            &source_sha256[..16.min(source_sha256.len())]
        ),
        plan_kind: "model-package-import".into(),
        generated_at_unix_ns: har_core::unix_timestamp_nanos(),
        model_identity,
        model_sha256,
        hardware,
        target_context,
        kv_datatype,
        mtp_enabled,
        quality_policy,
        budget,
        tensor_placements,
        transfers,
        operations,
        required_kernels,
        exactness: ExactnessContract::default(),
        fallback: FallbackContract::default(),
        telemetry: TelemetryContract::default(),
        assumptions: vec![
            "plan importer package was imported as data; no source code is executed".into(),
            context_delta,
            kv_delta,
        ],
        unresolved_risks,
        source_model_package_schema: Some(source_schema.into()),
        source_model_package_sha256: Some(source_sha256.into()),
    };
    Ok(plan)
}

fn compile_model_plan(model: &ModelPhenotype, mut hardware: HardwarePhenotype) -> ExecutionPlan {
    let mut placements = Vec::with_capacity(model.tensors.len());
    let mut transfers = Vec::new();
    let mut operations = OperationTable::new();
    for (index, tensor) in model.tensors.iter().enumerate() {
        let preferred = if tensor.is_mtp
            || tensor.role == har_core::TensorRole::Router
            || tensor.role == har_core::TensorRole::OutputProjection
        {
            MemoryTier::VramResident
        } else {
            MemoryTier::RamMapped
        };
        let backend = if preferred == MemoryTier::VramResident {
            BackendKind::Vulkan
        } else {
            BackendKind::Cpu
        };
        let kernel = if tensor.ggml_type == 12 {
            KernelKind::Q4KMatVec
        } else if tensor.is_quantized() {
            KernelKind::QuantizedMulMat
        } else {
            KernelKind::DenseMulMat
        };
        placements.push(TensorPlacement {
            tensor_id: tensor.name.clone(),
            tensor_name: tensor.name.clone(),
            bytes: tensor.payload_bytes,
            source_tier: MemoryTier::RamMapped,
            preferred_tier: preferred.clone(),
            backend: backend.clone(),
            kernel: kernel.clone(),
            alignment_bytes: tensor.alignment_bytes,
            reasons: vec![ReasonCode::ModelPackageImported],
            dependency_closure: vec![tensor.name.clone()],
            explanation: "Native Rust planner from model phenotype".into(),
        });
        if preferred != MemoryTier::RamMapped {
            transfers.push(TransferPlan {
                id: format!("transfer.{index}"),
                resource_id: tensor.name.clone(),
                source: MemoryTier::RamMapped,
                destination: preferred.clone(),
                bytes: tensor.payload_bytes,
                alignment_bytes: tensor.alignment_bytes,
                staging_required: true,
                queue: "transfer".into(),
                dependency: None,
            });
        }
        operations.physical.push(PhysicalOperation {
            index: index as u32,
            logical_id: index as u32,
            stable_id: tensor.name.clone(),
            backend,
            kernel,
            input_slots: vec![],
            output_slots: vec![],
            dependencies: if index == 0 {
                vec![]
            } else {
                vec![(index - 1) as u32]
            },
            dispatch: DispatchShape::default(),
            source_tier: MemoryTier::RamMapped,
            destination_tier: preferred,
        });
    }
    hardware.storage.probe_path = model.path.clone();
    let budget = ResourceBudget {
        model_bytes: model.file_bytes,
        tiers: vec![MemoryBudget {
            tier: MemoryTier::VramResident,
            capacity_bytes: hardware.gpu.safe_allocatable_vram_bytes,
            reserved_bytes: 0,
            assigned_bytes: placements
                .iter()
                .filter(|x| x.preferred_tier == MemoryTier::VramResident)
                .map(|x| x.bytes)
                .sum(),
            reservation_basis: "hardware phenotype".into(),
        }],
        ..Default::default()
    };
    ExecutionPlan {
        schema: PLAN_SCHEMA.into(),
        plan_id: format!(
            "native.{}",
            model.sha256.clone().unwrap_or_else(|| "unhashed".into())
        ),
        plan_kind: "native-rust-model-plan".into(),
        generated_at_unix_ns: har_core::unix_timestamp_nanos(),
        model_identity: model.model_name.clone(),
        model_sha256: model.sha256.clone().unwrap_or_default(),
        hardware,
        target_context: 4096,
        kv_datatype: "f16".into(),
        mtp_enabled: model.nextn_predict_layers > 0,
        quality_policy: "exact-native-rust".into(),
        budget,
        tensor_placements: placements,
        transfers,
        operations,
        required_kernels: vec!["q4_k_matvec".into()],
        exactness: ExactnessContract::default(),
        fallback: FallbackContract::default(),
        telemetry: TelemetryContract::default(),
        assumptions: vec!["Plan is immutable after validation".into()],
        unresolved_risks: Vec::new(),
        source_model_package_schema: None,
        source_model_package_sha256: None,
    }
}

fn parse_hardware(value: Option<&Value>) -> HardwarePhenotype {
    let mut hardware = HardwarePhenotype::synthetic_rdna4();
    if let Some(map) = value.and_then(Value::as_object) {
        if let Some(id) = string(map, "hardware_id") {
            hardware.schema = format!("har.hardware_phenotype.v1:{id}");
        }
        if let Some(arch) = string(map, "gpu_arch") {
            hardware.gpu.rdna_generation = arch;
        }
        if let Some(vram) = u64_value(map, "vram_bytes") {
            hardware.gpu.vram_total_bytes = vram;
        }
        if let Some(vram) = u64_value(map, "vram_bytes") {
            hardware.gpu.safe_allocatable_vram_bytes = vram;
        }
    }
    hardware
}
fn parse_budget(value: Option<&Value>, hardware: &HardwarePhenotype) -> ResourceBudget {
    let mut budget = ResourceBudget::default();
    budget.tiers.push(MemoryBudget {
        tier: MemoryTier::VramResident,
        capacity_bytes: hardware.gpu.safe_allocatable_vram_bytes,
        reserved_bytes: 0,
        assigned_bytes: 0,
        reservation_basis: "imported hardware profile".into(),
    });
    if let Some(map) = value
        .and_then(Value::as_object)
        .and_then(|x| x.get("tiers"))
        .and_then(Value::as_object)
    {
        for (tier, bytes) in map {
            if let Some(value) = bytes.as_u64() {
                budget.tiers.push(MemoryBudget {
                    tier: parse_tier(tier),
                    capacity_bytes: value,
                    reserved_bytes: 0,
                    assigned_bytes: 0,
                    reservation_basis: "plan importer storage_plan".into(),
                });
            }
        }
    }
    budget
}
fn parse_tier(value: &str) -> MemoryTier {
    match value.to_ascii_lowercase().as_str() {
        "nvme" | "nvme_cold" | "expert_sidecar" => MemoryTier::NvmeCold,
        "ram" | "ram_mapped" | "model" | "metadata" => MemoryTier::RamMapped,
        "ram_pinned" => MemoryTier::RamPinned,
        "vram" | "vram_resident" => MemoryTier::VramResident,
        "vram_slot" => MemoryTier::VramSlot,
        "scratch" => MemoryTier::ReconstructionScratch,
        _ => MemoryTier::RamMapped,
    }
}
fn kernel_from_format(value: &str) -> KernelKind {
    let format = QuantFormat::from_gguf_name(value);
    let hint = format.kernel_hint();
    if hint != KernelKind::Unknown {
        return hint;
    }
    let lower = value.to_ascii_lowercase();
    if lower.contains("router") {
        KernelKind::Sampling
    } else if lower.contains("mtp") {
        KernelKind::MtpVerify
    } else if lower.contains("q") {
        KernelKind::QuantizedMulMat
    } else {
        KernelKind::DenseMulMat
    }
}
fn string(map: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
    map.get(key).and_then(|x| x.as_str()).map(str::to_owned)
}
fn u64_value(map: &serde_json::Map<String, Value>, key: &str) -> Option<u64> {
    map.get(key)
        .and_then(Value::as_u64)
        .or_else(|| map.get(key).and_then(Value::as_f64).map(|x| x as u64))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn imports_model_package_shape() {
        let value: Value = serde_json::json!({"schema": MODEL_PACKAGE_SCHEMA, "model_identity":"fixture", "source_model_sha256":"abc", "hardware":{"hardware_id":"rdna4-vulkan"}, "tensors":[{"tensor_id":"x","name":"x","bytes":4,"quant_format":"Q4_K_S","planned_memory_tier":"vram"}], "storage_plan":{"tiers":{"vram":100}}, "unresolved_risks":[]});
        let map = value.as_object().unwrap();
        let plan = parse_model_package(map, MODEL_PACKAGE_SCHEMA, "0123456789abcdef").unwrap();
        assert_eq!(plan.tensor_placements.len(), 1);
        assert!(plan.validate().is_valid());
    }
    #[test]
    fn identity_hash_excludes_generation_timestamp() {
        let value: Value = serde_json::json!({"schema": MODEL_PACKAGE_SCHEMA, "model_identity":"fixture", "source_model_sha256":"abc", "tensors":[], "unresolved_risks":[]});
        let map = value.as_object().unwrap();
        let mut a = parse_model_package(map, MODEL_PACKAGE_SCHEMA, "0123456789abcdef").unwrap();
        let mut b = a.clone();
        a.generated_at_unix_ns = 1;
        b.generated_at_unix_ns = 2;
        assert_eq!(a.identity_hash().unwrap(), b.identity_hash().unwrap());
    }

    fn fixture(overrides: serde_json::Value) -> ExecutionPlan {
        let mut value = serde_json::json!({"schema": MODEL_PACKAGE_SCHEMA, "model_identity":"fixture", "source_model_sha256":"abc", "tensors":[{"tensor_id":"x","name":"x","bytes":4,"quant_format":"Q4_K","planned_memory_tier":"vram"}], "storage_plan":{"tiers":{"vram":10000}}, "unresolved_risks":[]});
        if let Some(object) = overrides.as_object() {
            if let Some(map) = value.as_object_mut() {
                for (key, item) in object {
                    map.insert(key.clone(), item.clone());
                }
            }
        }
        parse_model_package(
            value.as_object().unwrap(),
            MODEL_PACKAGE_SCHEMA,
            "0123456789abcdef",
        )
        .unwrap()
    }

    #[test]
    fn loader_rejects_unknown_kernels() {
        let mut plan = fixture(serde_json::json!({}));
        plan.operations.physical[0].kernel = KernelKind::Unknown;
        let report = plan.validate();
        assert!(report.errors.iter().any(|e| e.contains("UNKNOWN kernel")));
    }

    #[test]
    fn loader_rejects_illegal_cycles() {
        let mut plan = fixture(serde_json::json!({}));
        plan.operations.physical.push(PhysicalOperation {
            index: 1,
            logical_id: 1,
            stable_id: "y".into(),
            backend: BackendKind::Cpu,
            kernel: KernelKind::Q4KMatVec,
            input_slots: vec![],
            output_slots: vec![],
            dependencies: vec![0],
            dispatch: DispatchShape::default(),
            source_tier: MemoryTier::RamMapped,
            destination_tier: MemoryTier::RamMapped,
        });
        plan.operations.physical[0].dependencies = vec![1];
        assert!(plan.validate().errors.iter().any(|e| e.contains("cycle")));
    }

    #[test]
    fn loader_rejects_missing_capability_when_fallback_rejects() {
        let mut plan = fixture(serde_json::json!({}));
        plan.fallback.on_kernel_unavailable = "reject_plan".into();
        plan.hardware.capabilities.insert("q4_k_matvec", false);
        let report = plan.validate();
        assert!(report
            .errors
            .iter()
            .any(|e| e.contains("requires capability q4_k_matvec")));
    }

    #[test]
    fn loader_rejects_missing_capability_without_foreign_fallback() {
        let mut plan = fixture(serde_json::json!({}));
        plan.hardware.capabilities.insert("q4_k_matvec", false);
        let report = plan.validate();
        assert!(!report.is_valid());
        assert!(report
            .errors
            .iter()
            .any(|error| error.contains("requires capability q4_k_matvec")));
    }

    #[test]
    fn loader_rejects_resource_overcommit() {
        let mut plan = fixture(serde_json::json!({}));
        for budget in &mut plan.budget.tiers {
            budget.assigned_bytes = budget.capacity_bytes + 1;
        }
        assert!(plan
            .validate()
            .errors
            .iter()
            .any(|e| e.contains("overcommitted")));
    }

    #[test]
    fn loader_rejects_approximate_with_hash_requirement() {
        let mut plan = fixture(serde_json::json!({}));
        plan.exactness.mode = ExactnessMode::Approximate;
        assert!(plan
            .validate()
            .errors
            .iter()
            .any(|e| e.contains("Approximate")));
    }

    #[test]
    fn loader_rejects_missing_fallback_authority() {
        let mut plan = fixture(serde_json::json!({}));
        plan.fallback.authority_backend = BackendKind::None;
        assert!(plan
            .validate()
            .errors
            .iter()
            .any(|e| e.contains("no authority backend")));
    }

    #[test]
    fn plan_identity_hash_is_canonical_and_stable() {
        let mut a = fixture(serde_json::json!({}));
        let mut b = fixture(serde_json::json!({}));
        a.generated_at_unix_ns = 0;
        b.generated_at_unix_ns = 0;
        assert_eq!(a.identity_hash().unwrap(), b.identity_hash().unwrap());
        assert_eq!(a.canonical_bytes().unwrap(), b.canonical_bytes().unwrap());
        // Serialization order is deterministic: re-serializing yields identical bytes.
        let first = a.canonical_bytes().unwrap();
        let second = a.canonical_bytes().unwrap();
        assert_eq!(first, second);
    }
}
