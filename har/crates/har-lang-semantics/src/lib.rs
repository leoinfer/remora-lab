//! Semantic checks for HAR source.  This crate only builds a typed declaration;
//! it cannot execute arbitrary functions or callbacks.

use har_lang_ast::{Program, Value};
use har_lang_diagnostics::{Diagnostic, Span};
use std::collections::BTreeMap;

/// Strict V0 semantic implementation used by the language-plan compiler.
/// The legacy control lowering above remains available for compiler joins.
pub mod v0;

#[derive(Clone, Debug, PartialEq)]
pub struct SemanticControl {
    pub kind: String,
    pub name: String,
    pub fields: BTreeMap<String, Value>,
    pub ordered_fields: Vec<(String, Value)>,
    pub span: Span,
}
#[derive(Clone, Debug, PartialEq)]
pub struct SemanticProgram {
    pub source_name: String,
    pub controls: Vec<SemanticControl>,
}

pub fn analyze(program: &Program) -> Result<SemanticProgram, Vec<Diagnostic>> {
    let mut diagnostics = Vec::new();
    let mut controls = Vec::new();
    for block in &program.declarations {
        if !matches!(
            block.kind.as_str(),
            "target"
                | "model"
                | "tensor"
                | "tier"
                | "quality"
                | "decode"
                | "phase"
                | "telemetry"
                | "budget"
                | "dependency"
                | "runtime"
                | "plan"
        ) {
            diagnostics.push(Diagnostic::error(
                "S0001",
                format!("unsupported HAR declaration kind `{}`", block.kind),
                Some(block.span),
            ));
        }
        if block.name.is_empty() {
            diagnostics.push(Diagnostic::error(
                "S0002",
                "declaration name cannot be empty",
                Some(block.span),
            ));
        }
        let mut fields = BTreeMap::new();
        let mut ordered_fields = Vec::new();
        for field in &block.fields {
            if fields
                .insert(field.key.clone(), field.value.clone())
                .is_some()
                && field.key != "require"
            {
                diagnostics.push(Diagnostic::error(
                    "S0003",
                    format!("duplicate field `{}`", field.key),
                    Some(field.span),
                ));
            }
            ordered_fields.push((field.key.clone(), field.value.clone()));
            if !matches!(
                field.key.as_str(),
                "gpu"
                    | "wave"
                    | "vram_budget"
                    | "host_ram_budget"
                    | "storage"
                    | "identity"
                    | "model_sha256"
                    | "build_hash"
                    | "config_hash"
                    | "epoch"
                    | "hot"
                    | "warm"
                    | "cold"
                    | "format"
                    | "shape"
                    | "tier"
                    | "authority"
                    | "kernel"
                    | "vram_required"
                    | "generation"
                    | "fallback"
                    | "horizon"
                    | "optimize"
                    | "require"
                    | "sampling"
                    | "requires_epoch"
                    | "required_nvme"
                    | "ram_to_vram"
                    | "vram_reserve"
                    | "verification_compute"
                    | "queue_slots"
                    | "depends_on"
                    | "telemetry"
                    | "model"
                    | "backend"
                    | "target_context"
                    | "kv_datatype"
                    | "strict"
                    | "exactness"
            ) {
                diagnostics.push(Diagnostic::warning(
                    "S0004",
                    format!(
                        "unknown field `{}` is preserved but ignored by V1 lowering",
                        field.key
                    ),
                    Some(field.span),
                ));
            }
        }
        controls.push(SemanticControl {
            kind: block.kind.clone(),
            name: block.name.clone(),
            fields,
            ordered_fields,
            span: block.span,
        });
    }
    if diagnostics
        .iter()
        .any(|item| matches!(item.severity, har_lang_diagnostics::Severity::Error))
    {
        Err(diagnostics)
    } else {
        Ok(SemanticProgram {
            source_name: program.source_name.clone(),
            controls,
        })
    }
}
