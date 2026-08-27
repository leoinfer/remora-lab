//! compiler loadable package emitter.
//!
//! The compiler emits the `har.model_compiled_package.v0` interchange object
//! that `har_plan::PlanLoader::load` consumes, plus the language-typed
//! sections (semantic AST summary, logical IR, decode policy, runtime config,
//! model/hardware roots, contracts, evidence, and idea traceability).  compiler
//! never reparses HAR source; this file is the serialization boundary.

use crate::v0::CompiledProgram;
use har_lang_semantics::v0::{KernelRequirement, MemoryTier, QuantFormat};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

pub const MODEL_PACKAGE_SCHEMA: &str = "har.model_compiled_package.v0";

/// One canonical idea-registry entry (idea registry authority).
#[derive(Clone, Debug)]
pub struct IdeaRef {
    pub title: String,
    pub evidence: String,
}

/// Load idea registry's canonical `spec/research/idea-registry.json` if present.
/// The registry is consumed read-only; missing ideas are reported, never
/// recreated.
pub fn load_idea_registry(path: impl AsRef<Path>) -> Result<BTreeMap<String, IdeaRef>, String> {
    let bytes = fs::read(path).map_err(|error| format!("cannot read idea registry: {error}"))?;
    let root: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("idea registry is not JSON: {error}"))?;
    let ideas = root
        .get("ideas")
        .and_then(Value::as_array)
        .ok_or_else(|| "idea registry has no ideas array".to_string())?;
    let mut map = BTreeMap::new();
    for idea in ideas {
        let id = idea
            .get("canonical_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if id.is_empty() {
            continue;
        }
        let title = idea
            .get("canonical_title")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let evidence = idea
            .get("evidence_state")
            .and_then(|state| state.get("label"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        map.insert(id, IdeaRef { title, evidence });
    }
    Ok(map)
}

/// Fixed mapping from language constructs and compiler passes to canonical
/// idea IDs.  IDs are validated against the registry when one is supplied.
pub fn language_idea_traceability() -> Vec<(&'static str, &'static str, &'static [&'static str])> {
    vec![
        (
            "decode.horizon.elastic",
            "elastic MTP horizon",
            &["research-decode-horizon", "research-speculation"],
        ),
        (
            "decode.objective.accepted_tokens_per_complete_cost",
            "accepted-token roofline",
            &["research-roofline", "research-acceptance"],
        ),
        (
            "decode.cost.ram_to_vram",
            "resource-complementarity scheduling",
            &["research-resource-complementarity"],
        ),
        (
            "model.epoch_and_generation",
            "epoch namespace",
            &["research-generation-identity"],
        ),
        (
            "phase.dependency_graph",
            "causal closure",
            &["research-causal-closure"],
        ),
        (
            "quality.exact_and_fallback",
            "exactness and fallback",
            &["research-exactness", "research-native-policy"],
        ),
        (
            "target.hardware_phenotype",
            "hardware phenotype compilation",
            &["research-hardware-phenotype"],
        ),
    ]
}

fn tier_name(tier: MemoryTier) -> &'static str {
    match tier {
        MemoryTier::DirectNvme => "nvme",
        MemoryTier::RamMapped => "ram_mapped",
        MemoryTier::PinnedRam => "ram_pinned",
        MemoryTier::Vram => "vram_resident",
        MemoryTier::VramSlot => "vram_slot",
        MemoryTier::Scratch => "scratch",
    }
}

fn kernel_name(kernel: KernelRequirement) -> &'static str {
    match kernel {
        KernelRequirement::Cpu => "cpu",
        KernelRequirement::Vulkan => "vulkan",
        KernelRequirement::MtpVerify => "mtp_verify",
        KernelRequirement::Sampling => "sampling",
        KernelRequirement::Q4KMatVec => "q4_k_matvec",
        KernelRequirement::QuantizedMulMat => "q8_k_matvec",
        KernelRequirement::Attention => "attention",
        KernelRequirement::EmbeddingLookup => "embedding_lookup",
    }
}

fn format_name(format: QuantFormat) -> &'static str {
    match format {
        QuantFormat::Q4KS => "Q4_K_S",
        QuantFormat::Q4KM => "Q4_K_M",
        QuantFormat::Q8_0 => "Q8_0",
        QuantFormat::F16 => "F16",
        QuantFormat::F32 => "F32",
    }
}

fn role_name(kernel: KernelRequirement) -> &'static str {
    match kernel {
        KernelRequirement::MtpVerify => "MTP",
        KernelRequirement::Sampling => "SAMPLING",
        KernelRequirement::Q4KMatVec | KernelRequirement::QuantizedMulMat => "FEED_FORWARD",
        KernelRequirement::Attention => "ATTENTION",
        KernelRequirement::EmbeddingLookup => "TOKEN_EMBEDDING",
        _ => "OTHER",
    }
}

/// Emit the compiler loadable package.  `registry` is idea registry's canonical idea
/// registry; when absent, traceability entries are emitted with IDs only and
/// marked `registry_absent`.
pub fn emit_model_package(
    program: &CompiledProgram,
    registry: Option<&BTreeMap<String, IdeaRef>>,
) -> Result<String, String> {
    let typed = &program.typed;
    let target = &typed.target;
    let model = &typed.model;
    let decode = &typed.decode;
    let quality = &typed.quality;

    let tensors: Vec<Value> = typed
        .tensors
        .iter()
        .map(|tensor| {
            json!({
                "tensor_id": tensor.name,
                "name": tensor.name,
                "bytes": tensor.required_vram_bytes,
                "planned_memory_tier": tier_name(tensor.tier),
                "storage_location": "nvme",
                "quant_format": format_name(tensor.format),
                "supported_kernels": kernel_name(tensor.kernel),
                "alignment": 256,
                "role": role_name(tensor.kernel),
                "is_mtp": tensor.kernel == KernelRequirement::MtpVerify,
                "shape": tensor.shape.0,
                "authority": if tensor.authority == har_lang_semantics::v0::ValueAuthority::Exact { "exact" } else { "approximate" },
                "generation": tensor.generation.0,
                "model_root": tensor.model_root,
            })
        })
        .collect();

    let mut required_kernels: Vec<String> = Vec::new();
    for tensor in &typed.tensors {
        let name = kernel_name(tensor.kernel).to_string();
        if !required_kernels.contains(&name) {
            required_kernels.push(name);
        }
    }

    let mut idea_traceability: Vec<Value> = Vec::new();
    for (construct, purpose, ids) in language_idea_traceability() {
        let mut entries: Vec<Value> = Vec::new();
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
        idea_traceability.push(json!({
            "language_construct": construct,
            "purpose": purpose,
            "ideas": entries,
        }));
    }

    let logical_nodes: Vec<Value> = program
        .logical_ir
        .nodes
        .iter()
        .map(|node| {
            json!({
                "kind": node.kind,
                "name": node.name,
                "dependencies": node.dependencies,
            })
        })
        .collect();

    let evidence: Vec<Value> = typed
        .facts
        .iter()
        .map(|fact| {
            json!({
                "fact": fact.name,
                "metric": fact.metric,
                "value": fact.value,
                "evidence": fact.evidence,
            })
        })
        .collect();

    let package = json!({
        "schema": MODEL_PACKAGE_SCHEMA,
        "generated_at_unix_ns": 0,
        "compiler": "har-lang-compiler/stable-rust-v0",
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
        "tensors": tensors,
        "storage_plan": {
            "tiers": {
                "vram": target.vram_budget_bytes,
                "nvme": (model.size_gib * 1073741824.0).round() as u64,
                "ram": target.host_ram_budget_bytes,
            }
        },
        "required_kernels": required_kernels,
        "required_capabilities": target.capabilities,
        "exactness": {
            "mode": "EXACT",
            "output_hash_required": true,
            "state_boundary_hash_required": true,
            "numerical_tolerance": 0.0,
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
            "gpu_layers": decode.gpu_layers,
            "gpu_layers_total": decode.gpu_layers_total,
            "topology_matched": decode.topology_matched,
            "fallback": decode.fallback,
            "requires_epoch": decode.required_epoch.as_ref().map(|epoch| epoch.0),
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
            "storage": tier_name(target.storage),
            "capabilities": target.capabilities,
        },
        "typed_semantic_ast": {
            "target": target.name,
            "model": model.name,
            "quality": quality.name,
            "decode": decode.name,
            "tensor_count": typed.tensors.len(),
            "tier_count": typed.tiers.len(),
            "phase_count": typed.phases.len(),
            "telemetry_count": typed.telemetry.len(),
            "fact_count": typed.facts.len(),
        },
        "validated_logical_ir": {
            "schema": program.logical_ir.schema,
            "nodes": logical_nodes,
        },
        "evidence": evidence,
        "idea_traceability": idea_traceability,
        "assumptions": [
            "Plan was compiled ahead of time; no HAR source text is parsed at runtime",
            "Exactness requires a state boundary; token equality alone is not a certificate",
            "MTP acceptance is a pure discrete decision and never depends on timing",
        ],
        "unresolved_risks": [],
    });
    serde_json::to_string_pretty(&package)
        .map_err(|error| format!("package serialization failed: {error}"))
}
