//! HAR language compiler: source -> syntax AST -> semantic checks -> typed
//! logical IR -> immutable physical/startup plan.  The returned plan contains
//! no parser tokens and is the only value accepted by the runtime boundary.

use har_core::{BackendKind, FallbackContract};
use har_decode_control::DecodeControl;
use har_ir::OperationTable;
use har_lang_ast::Value;
use har_lang_diagnostics::Diagnostic;
use har_lang_lexer::lex;
use har_lang_parser::parse;
use har_lang_semantics::{analyze, SemanticControl, SemanticProgram};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::Path;

pub const COMPILER_SCHEMA: &str = "har-compiled-program/v1";

/// Strict V0 language-plan compiler; compatibility lowering above remains intact.
pub mod v0;

/// compiler loadable package emitter and canonical idea-registry consumer.
pub mod idea_registry;

/// First-executable-operation compiler pass (bundle + cross-boundary checks).
pub mod operation;

/// V3 native-execution bundle pass (additive; frozen v1 emitter untouched).
pub mod native_v3;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompiledHarProgram {
    pub schema: String,
    pub source_name: String,
    pub source_sha256: String,
    pub language_version: String,
    pub controls: Vec<DecodeControl>,
    pub operation_table: OperationTable,
    pub warnings: Vec<String>,
    pub model_identity: JsonValue,
    pub logical_ir: JsonValue,
    pub physical_plan: JsonValue,
}

impl CompiledHarProgram {
    /// Deterministic, documented interchange JSON.  This is canonical data,
    /// not an interpreter bytecode format.
    pub fn to_json(&self) -> String {
        let value = json!({
            "schema": self.schema,
            "source_name": self.source_name,
            "source_digest": self.source_sha256,
            "model_identity": self.model_identity,
            "validated_logical_ir": self.logical_ir,
            "immutable_physical_plan": self.physical_plan,
            "controls": self.controls,
            "warnings": self.warnings,
            "compiler": "har-lang-compiler/stable-rust-v0"
        });
        serde_json::to_string_pretty(&value).expect("compiled HAR plan is serializable")
    }
}

pub fn compile(
    source_name: impl Into<String>,
    source: &str,
) -> Result<CompiledHarProgram, Vec<Diagnostic>> {
    compile_source(source_name, source)
}

pub fn compile_source(
    source_name: impl Into<String>,
    source: &str,
) -> Result<CompiledHarProgram, Vec<Diagnostic>> {
    let source_name = source_name.into();
    let tokens = lex(source)?;
    let program = parse(&tokens, source_name.clone())?;
    let semantic = analyze(&program)?;
    validate(&semantic)?;
    Ok(lower(&semantic, &source_name, source))
}

pub fn compile_file(path: impl AsRef<Path>) -> Result<CompiledHarProgram, Vec<Diagnostic>> {
    let path = path.as_ref();
    let source = match fs::read_to_string(path) {
        Ok(value) => value,
        Err(error) => return Err(vec![Diagnostic::error("C0001", error.to_string(), None)]),
    };
    compile_source(path.display().to_string(), &source)
}

fn validate(program: &SemanticProgram) -> Result<(), Vec<Diagnostic>> {
    let mut errors = Vec::new();
    let find = |kind: &str| program.controls.iter().find(|control| control.kind == kind);
    let model = find("model");
    let target = find("target");
    let quality = find("quality");
    let tensors: Vec<&SemanticControl> = program
        .controls
        .iter()
        .filter(|control| control.kind == "tensor")
        .collect();
    let decode = find("decode");
    if let (Some(target), true) = (target, !tensors.is_empty()) {
        let budget = target
            .fields
            .get("vram_budget")
            .and_then(quantity_bytes)
            .unwrap_or(0);
        let required: u64 = tensors
            .iter()
            .filter_map(|tensor| tensor.fields.get("vram_required").and_then(quantity_bytes))
            .sum();
        if budget > 0 && required > budget {
            errors.push(Diagnostic::error(
                "S1050",
                format!("declared VRAM requirement {required} exceeds target budget {budget}"),
                Some(target.span),
            ));
        }
    }
    if let (Some(model), Some(decode)) = (model, decode) {
        let model_epoch = model.fields.get("epoch").and_then(number_u64).unwrap_or(0);
        let required_epoch = decode
            .fields
            .get("requires_epoch")
            .and_then(number_u64)
            .unwrap_or(model_epoch);
        if required_epoch != model_epoch {
            errors.push(Diagnostic::error(
                "S1053",
                format!("decode requires epoch {required_epoch}, model is epoch {model_epoch}"),
                Some(decode.span),
            ));
        }
    }
    if let Some(quality) = quality {
        let exact = atom(&quality.fields, "authority").as_deref() == Some("full_model")
            || quality.name == "exact";
        let fallback = atom(&quality.fields, "fallback").unwrap_or_default();
        if exact && fallback.eq_ignore_ascii_case("forbidden") {
            errors.push(Diagnostic::error(
                "S1033",
                "exact quality requires an explicit fallback",
                Some(quality.span),
            ));
        }
        if exact
            && tensors
                .iter()
                .any(|tensor| atom(&tensor.fields, "authority").as_deref() == Some("approximate"))
            && fallback.eq_ignore_ascii_case("forbidden")
        {
            errors.push(Diagnostic::error(
                "S1052",
                "approximate tensor cannot satisfy exact authority when fallback is forbidden",
                Some(quality.span),
            ));
        }
    }
    for tensor in &tensors {
        if let Some(kernel) = atom(&tensor.fields, "kernel") {
            if !matches!(
                kernel.to_ascii_lowercase().as_str(),
                "cpu" | "vulkan" | "mtp_verify" | "sampling"
            ) {
                errors.push(Diagnostic::error(
                    "S1043",
                    format!("unsupported kernel `{kernel}`"),
                    Some(tensor.span),
                ));
            }
        }
    }
    if let Some(decode) = decode {
        if let Some(objective) = atom(&decode.fields, "optimize") {
            if objective != "accepted_tokens_per_complete_cost" {
                errors.push(Diagnostic::error(
                    "S1036",
                    format!("unsupported objective `{objective}`"),
                    Some(decode.span),
                ));
            }
        }
    }
    let phases: Vec<&SemanticControl> = program
        .controls
        .iter()
        .filter(|control| control.kind == "phase")
        .collect();
    let phase_names: BTreeSet<String> = phases.iter().map(|phase| phase.name.clone()).collect();
    for phase in &phases {
        if let Some(parent) = atom(&phase.fields, "depends_on") {
            if !phase_names.contains(&parent) {
                errors.push(Diagnostic::error(
                    "S1047",
                    format!("phase `{}` depends on missing `{parent}`", phase.name),
                    Some(phase.span),
                ));
            }
        }
    }
    if has_phase_cycle(&phases) {
        let span = phases.first().map(|x| x.span);
        errors.push(Diagnostic::error(
            "S1048",
            "phase dependency graph contains a cycle",
            span,
        ));
    }
    if !errors.is_empty() {
        Err(errors)
    } else {
        Ok(())
    }
}

fn lower(program: &SemanticProgram, source_name: &str, source: &str) -> CompiledHarProgram {
    let model = program
        .controls
        .iter()
        .find(|control| control.kind == "model");
    let target = program
        .controls
        .iter()
        .find(|control| control.kind == "target");
    let decode = program
        .controls
        .iter()
        .find(|control| control.kind == "decode");
    let quality = program
        .controls
        .iter()
        .find(|control| control.kind == "quality");
    let model_identity = json!({ "name": model.map(|x| x.name.clone()).unwrap_or_default(), "identity": model.and_then(|x| x.fields.get("identity")).and_then(value_json_string).unwrap_or_default(), "model_sha256": model.and_then(|x| atom(&x.fields, "model_sha256")).unwrap_or_default(), "build_hash": model.and_then(|x| atom(&x.fields, "build_hash")).unwrap_or_default(), "config_hash": model.and_then(|x| atom(&x.fields, "config_hash")).unwrap_or_default(), "epoch": model.and_then(|x| x.fields.get("epoch")).and_then(number_u64).unwrap_or(0) });
    let logical_nodes: Vec<JsonValue> = program.controls.iter().map(|control| { let dependencies = if control.kind == "phase" { atom(&control.fields, "depends_on").map(|value| vec![value]).unwrap_or_default() } else if control.kind == "tensor" { atom(&control.fields, "tier").map(|value| vec![format!("tier:{value}")]).unwrap_or_default() } else if control.kind == "decode" { vec!["quality".into(), "model".into()] } else { Vec::new() }; json!({ "kind": kind_name(&control.kind), "name": control.name, "dependencies": dependencies }) }).collect();
    let logical_ir = json!({ "schema": "har-logical-ir/v1", "nodes": logical_nodes });
    let vram_budget = target
        .and_then(|x| x.fields.get("vram_budget"))
        .and_then(quantity_bytes)
        .unwrap_or(0);
    let host_budget = target
        .and_then(|x| x.fields.get("host_ram_budget"))
        .and_then(quantity_bytes)
        .unwrap_or(0);
    let required_tiers = required_tiers(program);
    let decode_model = decode
        .and_then(|control| atom(&control.fields, "model"))
        .or_else(|| model_identity["identity"].as_str().map(str::to_owned))
        .unwrap_or_default();
    let decode_control = lower_decode(decode, &decode_model);
    let required_memory_tiers: Vec<JsonValue> = program.controls.iter().filter(|control| control.kind == "tensor").map(|tensor| json!({ "tensor": tensor.name, "format": atom(&tensor.fields, "format").unwrap_or_default(), "logical_tier": atom(&tensor.fields, "tier").unwrap_or_default(), "tier": resolve_tier(atom(&tensor.fields, "tier").unwrap_or_default().as_str()), "authority": atom(&tensor.fields, "authority").unwrap_or_else(|| "exact".into()), "kernel": atom(&tensor.fields, "kernel").unwrap_or_default() })).collect();
    let physical_plan = json!({ "schema": "har-physical-plan/v1", "identity": { "model_sha256": model_identity["model_sha256"], "build_hash": model_identity["build_hash"], "config_hash": model_identity["config_hash"], "epoch": model_identity["epoch"] }, "vram_budget_bytes": vram_budget, "host_ram_budget_bytes": host_budget, "required_tiers": required_tiers, "required_memory_tiers": required_memory_tiers, "decode_policy": { "name": decode_control.name, "min_horizon": decode_control.horizon_start, "max_horizon": decode_control.horizon_end, "mode": if decode_control.horizon_end > decode_control.horizon_start { "elastic" } else { "fixed" }, "objective": decode.and_then(|x| atom(&x.fields, "optimize")).unwrap_or_default(), "require_exact_acceptance": decode.map(|x| x.fields.contains_key("require")).unwrap_or(false), "sampling_location": decode.and_then(|x| atom(&x.fields, "sampling")).unwrap_or_else(|| "cpu".into()), "fallback": decode.and_then(|x| atom(&x.fields, "fallback")).unwrap_or_else(|| "depth_zero".into()) }, "exact_authority": quality.and_then(|x| atom(&x.fields, "authority")).unwrap_or_else(|| "full_model".into()), "telemetry_requirements": telemetry_fields(program) });
    let operation_table = decode_control.lower_operation_table();
    let mut warnings = Vec::new();
    for control in &program.controls {
        for field in control.fields.keys() {
            if field == "unknown" {
                warnings.push(format!("unknown field in {}", control.name));
            }
        }
    }
    CompiledHarProgram {
        schema: COMPILER_SCHEMA.into(),
        source_name: source_name.into(),
        source_sha256: har_core::sha256_bytes(source.as_bytes()),
        language_version: "har-decode-control/v1".into(),
        controls: vec![decode_control],
        operation_table,
        warnings,
        model_identity,
        logical_ir,
        physical_plan,
    }
}

fn lower_decode(control: Option<&SemanticControl>, model: &str) -> DecodeControl {
    let mut result = DecodeControl {
        model: model.into(),
        ..DecodeControl::default()
    };
    if let Some(control) = control {
        result.name = control.name.clone();
        if let Some(Value::Range { start, end }) = control.fields.get("horizon") {
            result.horizon_start = start.parse().unwrap_or(0);
            result.horizon_end = end.parse().unwrap_or(0);
        }
        if let Some(value) = atom(&control.fields, "sampling") {
            result.backend = match value.to_ascii_lowercase().as_str() {
                "vulkan" | "backend" => BackendKind::Vulkan,
                "cpu" => BackendKind::Cpu,
                _ => BackendKind::Cpu,
            };
        }
        if let Some(fallback) = atom(&control.fields, "fallback") {
            result.fallback = FallbackContract {
                on_unknown_capacity: fallback.clone(),
                on_stale_generation: fallback.clone(),
                on_kernel_unavailable: fallback.clone(),
                ..FallbackContract::default()
            };
        }
    }
    result
}
fn atom(fields: &BTreeMap<String, Value>, key: &str) -> Option<String> {
    fields.get(key).and_then(value_json_string)
}
fn value_json_string(value: &Value) -> Option<String> {
    match value {
        Value::Ident(value) | Value::String(value) | Value::Number(value) => Some(value.clone()),
        Value::Quantity { number, unit } => Some(format!("{number} {unit}")),
        Value::Range { .. } | Value::List(_) => None,
    }
}
fn number_u64(value: &Value) -> Option<u64> {
    atom_value(value).and_then(|value| value.parse().ok())
}
fn atom_value(value: &Value) -> Option<String> {
    match value {
        Value::Ident(value) | Value::String(value) | Value::Number(value) => Some(value.clone()),
        _ => None,
    }
}
fn quantity_bytes(value: &Value) -> Option<u64> {
    match value {
        Value::Quantity { number, unit } => {
            let number: f64 = number.parse().ok()?;
            let multiplier = match unit.to_ascii_lowercase().as_str() {
                "b" => 1.0,
                "kib" => 1024.0,
                "mib" => 1024.0 * 1024.0,
                "gib" => 1024.0 * 1024.0 * 1024.0,
                "tib" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
                _ => return None,
            };
            Some((number * multiplier).round() as u64)
        }
        Value::Number(number) => number.parse().ok(),
        _ => None,
    }
}
fn resolve_tier(value: &str) -> String {
    match value {
        "expert_weights" => "vram_slot".into(),
        "kv_cache" => "vram".into(),
        "vram_slot" | "vram" | "pinned_ram" | "ram_mapped" | "direct_nvme" | "scratch" => {
            value.into()
        }
        other => other.into(),
    }
}
fn required_tiers(program: &SemanticProgram) -> Vec<String> {
    let mut values = Vec::new();
    for control in &program.controls {
        if control.kind == "tier" {
            for key in ["hot", "warm", "cold"] {
                if let Some(value) = atom(&control.fields, key) {
                    let resolved = resolve_tier(&value);
                    if !values.contains(&resolved) {
                        values.push(resolved);
                    }
                }
            }
        }
        if control.kind == "tensor" {
            if let Some(value) = atom(&control.fields, "tier") {
                let resolved = resolve_tier(&value);
                if !values.contains(&resolved) {
                    values.push(resolved);
                }
            }
        }
    }
    values
}
fn telemetry_fields(program: &SemanticProgram) -> Vec<String> {
    program
        .controls
        .iter()
        .filter(|x| x.kind == "telemetry")
        .flat_map(|x| {
            x.ordered_fields
                .iter()
                .filter(|(key, _)| key == "require")
                .filter_map(|(_, value)| atom_value(value))
        })
        .collect()
}
fn kind_name(kind: &str) -> &str {
    match kind {
        "tier" => "memory_tier",
        "decode" => "decode_policy",
        "tensor" => "tensor",
        other => other,
    }
}
fn has_phase_cycle(phases: &[&SemanticControl]) -> bool {
    let graph: HashMap<&str, &str> = phases
        .iter()
        .filter_map(|phase| {
            atom(&phase.fields, "depends_on").map(|parent| {
                (
                    phase.name.as_str(),
                    Box::leak(parent.into_boxed_str()) as &str,
                )
            })
        })
        .collect();
    for phase in phases {
        let mut cursor = phase.name.as_str();
        let mut seen = BTreeSet::new();
        while let Some(parent) = graph.get(cursor) {
            if !seen.insert(cursor) {
                return true;
            }
            cursor = parent;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_lowers_to_indexed_ir() {
        let compiled = compile_source(
            "x.har",
            "decode q4 { model \"q4.gguf\"; backend cpu; horizon 0..4; strict true; }",
        )
        .unwrap();
        assert_eq!(compiled.controls[0].model, "q4.gguf");
        assert_eq!(compiled.controls[0].horizon_end, 4);
        assert_eq!(compiled.operation_table.physical.len(), 1);
    }
}
