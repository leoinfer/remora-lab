//! Runtime owner.  It accepts an immutable compiled plan, validates identity
//! and delegates only mathematical dispatch to a public adapter trait.

use har_certificates::{identity_certificate, runtime_manifest};
use har_core::{HarError, HardwarePhenotype, Result, RuntimeManifest};
use har_execution::{ExecutionEngine, ExecutionResult, OperationAdapter};
use har_plan::{LoadedPlan, PlanLoader, ValidationReport};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const RUNTIME_INTERFACE: &str = "har.runtime.v1";

pub mod metabolism;
pub use metabolism::RuntimeMetabolism;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeIdentityInput {
    pub model_sha256: String,
    pub hardware_identity: String,
    pub source_path: String,
}

pub struct HarRuntime {
    pub loaded: LoadedPlan,
    pub hardware: HardwarePhenotype,
    pub identity: har_certificates::IdentityCertificate,
}
impl HarRuntime {
    pub fn load(plan_path: impl AsRef<Path>, hardware: HardwarePhenotype) -> Result<Self> {
        let loaded = PlanLoader::load(plan_path)?;
        Self::from_loaded(loaded, hardware)
    }
    pub fn from_loaded(loaded: LoadedPlan, hardware: HardwarePhenotype) -> Result<Self> {
        if !loaded.validation.is_valid() {
            return Err(HarError::Invalid {
                kind: "plan",
                message: loaded.validation.errors.join("; "),
            });
        }
        if loaded.plan.model_sha256.is_empty() {
            return Err(HarError::IdentityMismatch {
                field: "model_sha256".into(),
                expected: "non-empty".into(),
                actual: "empty".into(),
            });
        }
        if loaded.plan.hardware.gpu.name != hardware.gpu.name
            || loaded.plan.hardware.gpu.rdna_generation != hardware.gpu.rdna_generation
            || loaded.plan.hardware.gpu.subgroup_size != hardware.gpu.subgroup_size
            || loaded.plan.hardware.gpu.vram_total_bytes != hardware.gpu.vram_total_bytes
        {
            return Err(HarError::IdentityMismatch {
                field: "hardware phenotype".into(),
                expected: loaded.plan.hardware.identity(),
                actual: hardware.identity(),
            });
        }
        let identity =
            identity_certificate(&loaded.plan, &hardware, vec![loaded.source_path.clone()])?;
        Ok(Self {
            loaded,
            hardware,
            identity,
        })
    }
    pub fn metabolism(&self, sequence_id: u64, event_capacity: usize) -> RuntimeMetabolism {
        RuntimeMetabolism::from_loaded(&self.loaded, &self.hardware, sequence_id, event_capacity)
    }

    pub fn validation(&self) -> ValidationReport {
        self.loaded.validation.clone()
    }
    pub fn engine(&self) -> ExecutionEngine {
        ExecutionEngine::new(self.loaded.plan.clone())
    }
    pub fn execute<A: OperationAdapter>(
        &self,
        operation_index: u32,
        input: &[f32],
        adapter: &mut A,
    ) -> Result<ExecutionResult> {
        self.engine().execute(operation_index, input, adapter)
    }
    pub fn manifest(
        &self,
        operation_id: impl Into<String>,
        source_commit: impl Into<String>,
    ) -> Result<RuntimeManifest> {
        runtime_manifest(&self.loaded, &self.hardware, operation_id, source_commit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn runtime_interface_is_versioned() {
        assert_eq!(RUNTIME_INTERFACE, "har.runtime.v1");
    }
}

pub mod native_manifest;
pub mod policy;

pub use native_manifest::{
    BufferRecord, CorrectnessClassification, CorrectnessRecord, EventRecord, NativeRuntimeManifest,
};
pub use policy::RuntimePolicy;
