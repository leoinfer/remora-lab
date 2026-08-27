//! Certificates are evidence records, not permission to bypass validation.

use har_core::{canonical_sha256, ExactnessMode, HardwarePhenotype, Result, RuntimeManifest};
use har_plan::{ExecutionPlan, LoadedPlan};
use har_telemetry::TelemetrySnapshot;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const CERTIFICATE_SCHEMA: &str = "har.certificates.v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct IdentityCertificate {
    pub schema: String,
    pub model_sha256: String,
    pub hardware_sha256: String,
    pub plan_sha256: String,
    pub exactness: ExactnessMode,
    pub source_paths: Vec<String>,
    pub valid: bool,
    pub reasons: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompositionCertificate {
    pub schema: String,
    pub component_ids: Vec<String>,
    pub authority_root: String,
    pub state_boundary_hashes: Vec<String>,
    pub scopes: Vec<String>,
    pub valid: bool,
    pub missing_fields: Vec<String>,
}

pub fn identity_certificate(
    plan: &ExecutionPlan,
    hardware: &HardwarePhenotype,
    source_paths: Vec<String>,
) -> Result<IdentityCertificate> {
    let plan_sha256 = plan.identity_hash()?;
    let hardware_sha256 = canonical_sha256(hardware)?;
    let valid = !plan.model_sha256.is_empty() && !plan.plan_id.is_empty();
    Ok(IdentityCertificate {
        schema: CERTIFICATE_SCHEMA.into(),
        model_sha256: plan.model_sha256.clone(),
        hardware_sha256,
        plan_sha256,
        exactness: plan.exactness.mode.clone(),
        source_paths,
        valid,
        reasons: if valid {
            Vec::new()
        } else {
            vec!["model hash or plan identity is missing".into()]
        },
    })
}

pub fn compose_identity(
    certificate: &IdentityCertificate,
    telemetry: &TelemetrySnapshot,
) -> CompositionCertificate {
    let mut missing = Vec::new();
    if certificate.model_sha256.is_empty() {
        missing.push("model_sha256".into());
    }
    if certificate.plan_sha256.is_empty() {
        missing.push("plan_sha256".into());
    }
    if telemetry.epoch.model_root.is_empty() {
        missing.push("state_boundary.model_root".into());
    }
    if telemetry
        .operations
        .iter()
        .any(|op| op.output_hash.is_empty())
    {
        missing.push("output_hash".into());
    }
    CompositionCertificate {
        schema: CERTIFICATE_SCHEMA.into(),
        component_ids: vec!["identity".into(), "telemetry".into()],
        authority_root: certificate.model_sha256.clone(),
        state_boundary_hashes: vec![telemetry.epoch.model_root.identity()],
        scopes: vec!["one-operation".into()],
        valid: missing.is_empty(),
        missing_fields: missing,
    }
}

pub fn runtime_manifest(
    plan: &LoadedPlan,
    hardware: &HardwarePhenotype,
    operation_id: impl Into<String>,
    source_commit: impl Into<String>,
) -> Result<RuntimeManifest> {
    let mut manifest = RuntimeManifest {
        model_sha256: plan.plan.model_sha256.clone(),
        hardware_sha256: canonical_sha256(hardware)?,
        plan_sha256: plan.plan.identity_hash()?,
        operation_id: operation_id.into(),
        source_commit: source_commit.into(),
        reference_commit: "none".into(),
        exactness: plan.plan.exactness.mode.clone(),
        plan_validation: if plan.validation.is_valid() {
            if plan.validation.warnings.is_empty() {
                "PASS"
            } else {
                "PASS_WITH_WARNINGS"
            }
        } else {
            "BLOCKED"
        }
        .into(),
        reference_adapters: Vec::new(),
        ..RuntimeManifest::default()
    };
    manifest.notes.push(
        "execution_backend=native-rust; reference adapters are not part of the production runtime"
            .into(),
    );
    manifest
        .notes
        .extend(plan.validation.warnings.iter().take(16).cloned());
    if !plan.validation.is_valid() {
        manifest
            .notes
            .extend(plan.validation.errors.iter().cloned());
    }
    Ok(manifest)
}

pub fn write_json<T: Serialize>(path: impl AsRef<Path>, value: &T) -> Result<()> {
    let bytes = har_core::canonical_json(value)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_identity_is_not_valid() {
        let plan = ExecutionPlan {
            schema: har_plan::PLAN_SCHEMA.into(),
            plan_id: "".into(),
            plan_kind: "".into(),
            generated_at_unix_ns: 0,
            model_identity: "".into(),
            model_sha256: "".into(),
            hardware: HardwarePhenotype::synthetic_rdna4(),
            target_context: 0,
            kv_datatype: "".into(),
            mtp_enabled: false,
            quality_policy: "".into(),
            budget: Default::default(),
            tensor_placements: vec![],
            transfers: vec![],
            operations: har_ir::OperationTable::new(),
            required_kernels: vec![],
            exactness: Default::default(),
            fallback: Default::default(),
            telemetry: Default::default(),
            assumptions: vec![],
            unresolved_risks: vec![],
            source_model_package_schema: None,
            source_model_package_sha256: None,
        };
        assert!(
            !identity_certificate(&plan, &plan.hardware, vec![])
                .unwrap()
                .valid
        );
    }
}
