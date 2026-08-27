//! Complete native RuntimeManifest (PHASE 4, compiler-owned).
//!
//! `NativeRuntimeManifest` (schema `har.runtime_manifest.native.v1`) carries
//! every mandatory identity required by
//! `HAR_INTEGRATED_RUNTIME_MANIFEST_REQUIREMENTS.json` plus the native
//! execution fields: runtime/integration/language/backend/package source
//! commits, interface versions, model/package/plan/hardware roots,
//! package-entry identity, operation identity, kernel registry hash, kernel
//! identity, shader/SPIR-V hashes, runtime policy, fallback policy,
//! fallback count, reference invocation count, buffers, generations, events,
//! timestamps, correctness classification and telemetry hashes.
//!
//! `validate()` fails closed: any empty mandatory identity is an error, so a
//! manifest can never be produced with a silently missing identity.

use har_core::{ExactnessMode, HarError, HardwarePhenotype, Result};
use serde::{Deserialize, Serialize};

pub const NATIVE_MANIFEST_SCHEMA: &str = "har.runtime_manifest.native.v1";

/// One backend buffer identity record (native-kernel registry `BackendBufferId` mapping).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BufferRecord {
    pub resource: String,
    pub epoch: String,
    pub generation: u64,
    pub bytes: u64,
    pub tier: String,
}

/// One explicit completion event record (native-kernel registry `BackendEventId` mapping).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EventRecord {
    pub resource: String,
    pub epoch: String,
    pub generation: u64,
    pub sequence: u64,
    pub kind: String,
    pub timestamp_ns: u64,
}

/// Correctness classification (mirrors the integration requirements enum).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CorrectnessClassification {
    WithinReferenceTolerance,
    ExactOperationMatch,
    NumericalMismatch,
    StructuralMismatch,
    HiddenFallback,
    InvalidComparison,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CorrectnessRecord {
    pub classification: CorrectnessClassification,
    pub max_abs_error: f64,
    pub tolerance_abs: f64,
    pub reference_value: f64,
    pub output_value: f64,
}

/// The complete native runtime manifest.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NativeRuntimeManifest {
    pub schema: String,
    pub runtime_name: String,
    pub runtime_version: String,
    // source identities (mandatory, non-empty)
    pub runtime_source_commit: String,
    pub integration_source_commit: String,
    pub language_bundle_commit: String,
    pub backend_source_commit: String,
    pub package_source_commit: String,
    pub residency_source_commit: String,
    // interface + roots
    pub interface_versions: Vec<String>,
    pub model_root: String,
    pub package_root: String,
    pub package_entry_identity: String,
    pub plan_root: String,
    pub hardware_root: String,
    pub operation_id: String,
    // native kernel identities
    pub kernel_registry_hash: String,
    pub kernel_identity: String,
    pub kernel_registry_path: String,
    pub shader_hashes: Vec<String>,
    // policy and counters
    pub runtime_policy: String,
    pub fallback_policy: String,
    pub fallback_count: u64,
    pub reference_invocation_count: u64,
    pub exactness: ExactnessMode,
    pub plan_validation: String,
    // execution records
    pub buffers: Vec<BufferRecord>,
    pub generations: Vec<u64>,
    pub events: Vec<EventRecord>,
    pub timestamps_ns: Vec<u64>,
    pub correctness: Option<CorrectnessRecord>,
    pub telemetry_hashes: Vec<String>,
    pub unsafe_ledger_identity: String,
    pub notes: Vec<String>,
}

impl NativeRuntimeManifest {
    pub fn empty() -> Self {
        Self {
            schema: NATIVE_MANIFEST_SCHEMA.into(),
            runtime_name: "Hardware-Aware Runtime".into(),
            runtime_version: "0.1.0-rust".into(),
            runtime_source_commit: String::new(),
            integration_source_commit: String::new(),
            language_bundle_commit: String::new(),
            backend_source_commit: String::new(),
            package_source_commit: String::new(),
            residency_source_commit: String::new(),
            interface_versions: Vec::new(),
            model_root: String::new(),
            package_root: String::new(),
            package_entry_identity: String::new(),
            plan_root: String::new(),
            hardware_root: String::new(),
            operation_id: String::new(),
            kernel_registry_hash: String::new(),
            kernel_identity: String::new(),
            kernel_registry_path: String::new(),
            shader_hashes: Vec::new(),
            runtime_policy: String::new(),
            fallback_policy: String::new(),
            fallback_count: 0,
            reference_invocation_count: 0,
            exactness: ExactnessMode::Exact,
            plan_validation: "NOT_RUN".into(),
            buffers: Vec::new(),
            generations: Vec::new(),
            events: Vec::new(),
            timestamps_ns: Vec::new(),
            correctness: None,
            telemetry_hashes: Vec::new(),
            unsafe_ledger_identity: String::new(),
            notes: Vec::new(),
        }
    }

    /// Fail-closed completeness check.  Returns every missing/empty
    /// mandatory identity as an error; an empty error list means complete.
    pub fn validate(&self) -> Vec<String> {
        let mut errors: Vec<String> = Vec::new();
        let mut require = |field: &str, value: &str| {
            if value.trim().is_empty() {
                errors.push(format!("missing mandatory identity: {field}"));
            }
        };
        require("runtime_source_commit", &self.runtime_source_commit);
        require("integration_source_commit", &self.integration_source_commit);
        require("language_bundle_commit", &self.language_bundle_commit);
        require("backend_source_commit", &self.backend_source_commit);
        require("package_source_commit", &self.package_source_commit);
        require("residency_source_commit", &self.residency_source_commit);
        require("model_root", &self.model_root);
        require("package_root", &self.package_root);
        require("package_entry_identity", &self.package_entry_identity);
        require("plan_root", &self.plan_root);
        require("hardware_root", &self.hardware_root);
        require("operation_id", &self.operation_id);
        require("kernel_registry_hash", &self.kernel_registry_hash);
        require("kernel_identity", &self.kernel_identity);
        require("kernel_registry_path", &self.kernel_registry_path);
        require("runtime_policy", &self.runtime_policy);
        require("fallback_policy", &self.fallback_policy);
        require("unsafe_ledger_identity", &self.unsafe_ledger_identity);
        if self.interface_versions.is_empty() {
            errors.push("missing mandatory identity: interface_versions".into());
        }
        if self.shader_hashes.is_empty() {
            errors.push("missing mandatory identity: shader_hashes".into());
        }
        if self.buffers.is_empty() {
            errors.push("missing mandatory identity: buffers".into());
        }
        if self.generations.is_empty() {
            errors.push("missing mandatory identity: generations".into());
        }
        if self.events.is_empty() {
            errors.push("missing mandatory identity: events".into());
        }
        if self.telemetry_hashes.is_empty() {
            errors.push("missing mandatory identity: telemetry_hashes".into());
        }
        if self.correctness.is_none() {
            errors.push("missing mandatory identity: correctness".into());
        }
        // NATIVE_REQUIRED invariants are structural: the manifest must never
        // record a fallback or a reference invocation while claiming a
        // native-required policy.
        if self.runtime_policy.contains("NATIVE_REQUIRED")
            && (self.fallback_count != 0 || self.reference_invocation_count != 0)
        {
            errors.push(format!(
                "NATIVE_REQUIRED violated in manifest: fallback_count={} reference_invocation_count={}",
                self.fallback_count, self.reference_invocation_count
            ));
        }
        errors
    }

    pub fn is_complete(&self) -> bool {
        self.validate().is_empty()
    }

    pub fn into_result(self) -> Result<Self> {
        let errors = self.validate();
        if errors.is_empty() {
            Ok(self)
        } else {
            Err(HarError::Invalid {
                kind: "native runtime manifest",
                message: errors.join("; "),
            })
        }
    }

    /// Convenience constructor used by the fixture/adapter lanes; every
    /// identity must be supplied (fail closed).
    #[allow(clippy::too_many_arguments)]
    pub fn from_components(
        runtime_source_commit: String,
        integration_source_commit: String,
        language_bundle_commit: String,
        backend_source_commit: String,
        package_source_commit: String,
        residency_source_commit: String,
        interface_versions: Vec<String>,
        model_root: String,
        package_root: String,
        package_entry_identity: String,
        plan_root: String,
        hardware: &HardwarePhenotype,
        operation_id: String,
        kernel_registry_hash: String,
        kernel_identity: String,
        kernel_registry_path: String,
        shader_hashes: Vec<String>,
        runtime_policy: String,
        fallback_policy: String,
        fallback_count: u64,
        reference_invocation_count: u64,
        unsafe_ledger_identity: String,
    ) -> Self {
        Self {
            schema: NATIVE_MANIFEST_SCHEMA.into(),
            runtime_name: "Hardware-Aware Runtime".into(),
            runtime_version: "0.1.0-rust".into(),
            runtime_source_commit,
            integration_source_commit,
            language_bundle_commit,
            backend_source_commit,
            package_source_commit,
            residency_source_commit,
            interface_versions,
            model_root,
            package_root,
            package_entry_identity,
            plan_root,
            hardware_root: hardware.identity(),
            operation_id,
            kernel_registry_hash,
            kernel_identity,
            kernel_registry_path,
            shader_hashes,
            runtime_policy,
            fallback_policy,
            fallback_count,
            reference_invocation_count,
            exactness: ExactnessMode::Exact,
            plan_validation: "PASS".into(),
            buffers: Vec::new(),
            generations: Vec::new(),
            events: Vec::new(),
            timestamps_ns: Vec::new(),
            correctness: None,
            telemetry_hashes: Vec::new(),
            unsafe_ledger_identity,
            notes: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use har_core::HardwarePhenotype;

    fn complete() -> NativeRuntimeManifest {
        let hardware = HardwarePhenotype::synthetic_rdna4();
        NativeRuntimeManifest::from_components(
            "runtime-commit".into(),
            "integration-commit".into(),
            "bundle-commit".into(),
            "backend-commit".into(),
            "package-commit".into(),
            "residency-commit".into(),
            vec!["har.core.v1".into(), "har.ir.v1".into()],
            "model-root".into(),
            "package-root".into(),
            "blk.0.ffn_gate.weight".into(),
            "plan-root".into(),
            &hardware,
            "Q4K_MATVEC:blk.0.ffn_gate.weight".into(),
            "registry-hash".into(),
            "vulkan.q4_k.gemv".into(),
            "native-kernel-registry.json".into(),
            vec!["q4k_gemv.comp public-shader-identity".into()],
            "NATIVE_REQUIRED".into(),
            "on_kernel_unavailable: reject_plan".into(),
            0,
            0,
            "HAR_UNSAFE_RUST_LEDGER.json".into(),
        )
    }

    #[test]
    fn complete_manifest_passes_validation() {
        let mut manifest = complete();
        manifest.buffers.push(BufferRecord {
            resource: "q4k.weights.block0".into(),
            epoch: "epoch-1".into(),
            generation: 1,
            bytes: 144,
            tier: "VRAM_SLOT".into(),
        });
        manifest.generations.push(1);
        manifest.events.push(EventRecord {
            resource: "q4k.block0".into(),
            epoch: "epoch-1".into(),
            generation: 1,
            sequence: 1,
            kind: "NATIVE_DISPATCH_COMPLETE".into(),
            timestamp_ns: 0,
        });
        manifest.timestamps_ns.push(42);
        manifest.correctness = Some(CorrectnessRecord {
            classification: CorrectnessClassification::WithinReferenceTolerance,
            max_abs_error: 8.74e-09,
            tolerance_abs: 1e-4,
            reference_value: -0.10303788525717598,
            output_value: -0.10303788525717598,
        });
        manifest.telemetry_hashes.push("telemetry-hash".into());
        assert!(manifest.is_complete(), "errors: {:?}", manifest.validate());
        assert!(manifest.clone().into_result().is_ok());
    }

    // One negative test per mandatory field: every empty identity must be
    // reported by validate().
    #[test]
    fn every_mandatory_field_has_a_negative_test() {
        #[allow(clippy::type_complexity)]
        let fields: Vec<(&str, fn(&mut NativeRuntimeManifest))> = vec![
            ("runtime_source_commit", |m| m.runtime_source_commit.clear()),
            ("integration_source_commit", |m| {
                m.integration_source_commit.clear()
            }),
            ("language_bundle_commit", |m| {
                m.language_bundle_commit.clear()
            }),
            ("backend_source_commit", |m| m.backend_source_commit.clear()),
            ("package_source_commit", |m| m.package_source_commit.clear()),
            ("residency_source_commit", |m| {
                m.residency_source_commit.clear()
            }),
            ("model_root", |m| m.model_root.clear()),
            ("package_root", |m| m.package_root.clear()),
            ("package_entry_identity", |m| {
                m.package_entry_identity.clear()
            }),
            ("plan_root", |m| m.plan_root.clear()),
            ("hardware_root", |m| m.hardware_root.clear()),
            ("operation_id", |m| m.operation_id.clear()),
            ("kernel_registry_hash", |m| m.kernel_registry_hash.clear()),
            ("kernel_identity", |m| m.kernel_identity.clear()),
            ("kernel_registry_path", |m| m.kernel_registry_path.clear()),
            ("runtime_policy", |m| m.runtime_policy.clear()),
            ("fallback_policy", |m| m.fallback_policy.clear()),
            ("unsafe_ledger_identity", |m| {
                m.unsafe_ledger_identity.clear()
            }),
            ("interface_versions", |m| m.interface_versions.clear()),
            ("shader_hashes", |m| m.shader_hashes.clear()),
            ("buffers", |m| m.buffers.clear()),
            ("generations", |m| m.generations.clear()),
            ("events", |m| m.events.clear()),
            ("telemetry_hashes", |m| m.telemetry_hashes.clear()),
            ("correctness", |m| m.correctness = None),
        ];
        for (name, mutate) in fields {
            let mut manifest = complete();
            manifest.buffers.push(BufferRecord {
                resource: "q4k.weights.block0".into(),
                epoch: "epoch-1".into(),
                generation: 1,
                bytes: 144,
                tier: "VRAM_SLOT".into(),
            });
            manifest.generations.push(1);
            manifest.events.push(EventRecord {
                resource: "q4k.block0".into(),
                epoch: "epoch-1".into(),
                generation: 1,
                sequence: 1,
                kind: "NATIVE_DISPATCH_COMPLETE".into(),
                timestamp_ns: 0,
            });
            manifest.timestamps_ns.push(42);
            manifest.correctness = Some(CorrectnessRecord {
                classification: CorrectnessClassification::ExactOperationMatch,
                max_abs_error: 0.0,
                tolerance_abs: 1e-4,
                reference_value: -0.10303788525717598,
                output_value: -0.10303788525717598,
            });
            manifest.telemetry_hashes.push("telemetry-hash".into());
            mutate(&mut manifest);
            let errors = manifest.validate();
            assert!(
                errors.iter().any(|error| error.contains(name)),
                "expected {name} rejection, got {errors:?}"
            );
            assert!(!manifest.is_complete(), "{name} must fail completeness");
            assert!(manifest.clone().into_result().is_err());
        }
    }

    #[test]
    fn native_required_manifest_rejects_recorded_fallback() {
        let mut manifest = complete();
        manifest.fallback_count = 1;
        let errors = manifest.validate();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("NATIVE_REQUIRED violated")),
            "{errors:?}"
        );
        manifest.fallback_count = 0;
        manifest.reference_invocation_count = 1;
        let errors = manifest.validate();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("NATIVE_REQUIRED violated")),
            "{errors:?}"
        );
    }

    #[test]
    fn empty_manifest_reports_all_mandatory_fields() {
        let errors = NativeRuntimeManifest::empty().validate();
        // every mandatory identity must be reported
        assert!(
            errors.len() >= 25,
            "got {} errors: {errors:?}",
            errors.len()
        );
        assert!(!NativeRuntimeManifest::empty().is_complete());
    }
}
