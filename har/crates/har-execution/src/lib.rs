//! HAR-owned operation sequencing.  A backend adapter supplies mathematics;
//! this crate owns plan order, resource identity, residency transitions,
//! output comparison, and telemetry.

use har_core::{sha256_f32, BackendKind, HarError, MemoryTier, ResidencyState, Result};
use har_events::{EventHeader, RuntimeEvent, RuntimeEventKind};
use har_ir::PhysicalOperation;
use har_memory::{ResidencyMachine, ResidencyRecord};
use har_plan::{ExecutionPlan, ValidationReport};
use har_telemetry::{elapsed_ns, OperationTelemetry, TelemetryCollector, TelemetrySnapshot};
use serde::{Deserialize, Serialize};
use std::time::Instant;

pub mod speculation;
pub use speculation::{
    BlockResolution, SpeculativeBlock, SpeculativeCandidate, SPECULATION_BOUNDARY,
};

pub const EXECUTION_INTERFACE: &str = "har.execution.v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DifferentialVerdict {
    ExactMatch,
    WithinReferenceTolerance,
    NumericalMismatch,
    StructuralMismatch,
    Unsupported,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DispatchOutput {
    pub output: Vec<f32>,
    pub reference_output: Vec<f32>,
    pub bytes_moved: u64,
    pub note: String,
}

pub trait OperationAdapter {
    fn backend(&self) -> BackendKind;
    fn dispatch(&mut self, operation: &PhysicalOperation, input: &[f32]) -> Result<DispatchOutput>;
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExecutionResult {
    pub schema: String,
    pub operation_id: String,
    pub tensor_name: String,
    pub verdict: DifferentialVerdict,
    pub input_elements: u64,
    pub output_elements: u64,
    pub bytes_moved: u64,
    pub elapsed_ns: u64,
    pub rms_error: f64,
    pub max_error: f64,
    pub normalized_error: f64,
    pub output_hash: String,
    pub reference_output_hash: String,
    pub output_values: Vec<f32>,
    pub reference_output_values: Vec<f32>,
    pub residency_events: Vec<har_memory::ResidencyEvent>,
    pub telemetry: TelemetrySnapshot,
    pub plan_validation: ValidationReport,
    pub note: String,
}

pub struct ExecutionEngine {
    pub plan: ExecutionPlan,
}
impl ExecutionEngine {
    pub fn new(plan: ExecutionPlan) -> Self {
        Self { plan }
    }
    pub fn validate(&self) -> ValidationReport {
        self.plan.validate()
    }
    pub fn execute<A: OperationAdapter>(
        &self,
        operation_index: u32,
        input: &[f32],
        adapter: &mut A,
    ) -> Result<ExecutionResult> {
        let validation = self.validate();
        if !validation.is_valid() {
            return Err(HarError::Invalid {
                kind: "execution plan",
                message: validation.errors.join("; "),
            });
        }
        let operation = self
            .plan
            .operations
            .operation(operation_index)
            .ok_or_else(|| HarError::Invalid {
                kind: "operation index",
                message: operation_index.to_string(),
            })?
            .clone();
        let plan_hash = self.plan.identity_hash()?;
        let epoch = har_core::EpochNamespace::new(
            har_core::ModelRoot::new(
                self.plan.model_identity.clone(),
                self.plan.model_sha256.clone(),
            ),
            0,
        );
        let mut telemetry = TelemetryCollector::new(epoch.clone(), plan_hash, 256);
        let mut machine = ResidencyMachine::new(
            operation.stable_id.clone(),
            self.plan.model_sha256.clone(),
            0,
        );
        let mut record = ResidencyRecord::new(
            har_memory::BufferId::new(operation_index, 0, operation.stable_id.clone()),
            operation.stable_id.clone(),
            1,
            self.plan.model_sha256.clone(),
        );
        machine.transition(ResidencyState::Indexed, "immutable plan operation indexed")?;
        record.machine = machine.clone();
        machine.transition(
            ResidencyState::ReadQueued,
            "HAR scheduler queued source span",
        )?;
        machine.transition(
            ResidencyState::Reading,
            "reference adapter copied bounded source span",
        )?;
        machine.transition(
            ResidencyState::ReadyHost,
            "source bytes and input are host-ready",
        )?;
        let needs_transfer = matches!(
            operation.destination_tier,
            MemoryTier::VramResident | MemoryTier::VramSlot
        );
        if needs_transfer {
            machine.transition(
                ResidencyState::TransferQueued,
                "plan requires explicit host-to-VRAM residency",
            )?;
            machine.transition(
                ResidencyState::CopyingToVram,
                "reference adapter models the copy boundary",
            )?;
            machine.transition(ResidencyState::ReadyVram, "copy boundary complete")?;
        }
        machine.transition(ResidencyState::Computing, "HAR indexed dispatch")?;
        let header = EventHeader {
            sequence: 0,
            timestamp_ns: har_core::unix_timestamp_nanos(),
            epoch: epoch.clone(),
            operation_index,
        };
        telemetry.record_event(RuntimeEvent {
            header: header.clone(),
            kind: RuntimeEventKind::Residency {
                resource_id: operation.stable_id.clone(),
                from: ResidencyState::ReadyHost,
                to: if needs_transfer {
                    ResidencyState::ReadyVram
                } else {
                    ResidencyState::Computing
                },
                reason: "plan-controlled residency".into(),
            },
        });
        telemetry.record_event(RuntimeEvent {
            header: EventHeader {
                sequence: 1,
                ..header.clone()
            },
            kind: RuntimeEventKind::Dispatch {
                backend: adapter.backend(),
                kernel: operation.kernel.clone(),
                operation_id: operation.stable_id.clone(),
            },
        });
        let start = Instant::now();
        let output = adapter.dispatch(&operation, input)?;
        let elapsed = elapsed_ns(start);
        if output.output.len() != output.reference_output.len() {
            return Ok(self.structural_result(
                operation, input, machine, telemetry, validation, elapsed, output,
            ));
        }
        let (rms, max, normalized) = errors(&output.output, &output.reference_output);
        let output_hash = sha256_f32(&output.output);
        let reference_hash = sha256_f32(&output.reference_output);
        let verdict = if output.output.is_empty() {
            DifferentialVerdict::StructuralMismatch
        } else if output_hash == reference_hash {
            DifferentialVerdict::ExactMatch
        } else if normalized <= 5e-5 && max <= 5e-3 {
            DifferentialVerdict::WithinReferenceTolerance
        } else {
            DifferentialVerdict::NumericalMismatch
        };
        let _ = machine.transition(
            ResidencyState::ReadyHost,
            "output materialized for comparison",
        );
        let events = machine.events.clone();
        let operation_telemetry = OperationTelemetry {
            operation_id: operation.stable_id.clone(),
            input_hash: sha256_f32(input),
            output_hash: output_hash.clone(),
            reference_output_hash: reference_hash.clone(),
            input_elements: input.len() as u64,
            output_elements: output.output.len() as u64,
            bytes_moved: output.bytes_moved,
            dispatch_count: 1,
            residency_event_count: events.len() as u64,
            elapsed_ns: elapsed,
            exact: matches!(verdict, DifferentialVerdict::ExactMatch),
            normalized_error_ppm: (normalized * 1_000_000.0).round() as u64,
        };
        telemetry.record_operation(operation_telemetry);
        telemetry.record_event(RuntimeEvent {
            header: EventHeader {
                sequence: 2,
                ..header
            },
            kind: RuntimeEventKind::Output {
                output_hash: output_hash.clone(),
                elements: output.output.len() as u64,
            },
        });
        Ok(ExecutionResult {
            schema: EXECUTION_INTERFACE.into(),
            operation_id: operation.stable_id.clone(),
            tensor_name: operation.stable_id,
            verdict,
            input_elements: input.len() as u64,
            output_elements: output.output.len() as u64,
            bytes_moved: output.bytes_moved,
            elapsed_ns: elapsed,
            rms_error: rms,
            max_error: max,
            normalized_error: normalized,
            output_hash,
            reference_output_hash: reference_hash,
            output_values: output.output.clone(),
            reference_output_values: output.reference_output.clone(),
            residency_events: events,
            telemetry: telemetry.snapshot(),
            plan_validation: validation,
            note: output.note,
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn structural_result(
        &self,
        operation: PhysicalOperation,
        input: &[f32],
        machine: ResidencyMachine,
        telemetry: TelemetryCollector,
        validation: ValidationReport,
        elapsed: u64,
        output: DispatchOutput,
    ) -> ExecutionResult {
        ExecutionResult {
            schema: EXECUTION_INTERFACE.into(),
            operation_id: operation.stable_id.clone(),
            tensor_name: operation.stable_id,
            verdict: DifferentialVerdict::StructuralMismatch,
            input_elements: input.len() as u64,
            output_elements: output.output.len() as u64,
            bytes_moved: output.bytes_moved,
            elapsed_ns: elapsed,
            rms_error: 0.0,
            max_error: 0.0,
            normalized_error: 0.0,
            output_hash: sha256_f32(&output.output),
            reference_output_hash: sha256_f32(&output.reference_output),
            output_values: output.output.clone(),
            reference_output_values: output.reference_output.clone(),
            residency_events: machine.events,
            telemetry: telemetry.snapshot(),
            plan_validation: validation,
            note: "reference and HAR output lengths differ".into(),
        }
    }
}

fn errors(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    if a.len() != b.len() || a.is_empty() {
        return (f64::INFINITY, f64::INFINITY, f64::INFINITY);
    }
    let mut sum = 0.0;
    let mut reference = 0.0;
    let mut maximum: f64 = 0.0;
    for (x, y) in a.iter().zip(b) {
        let difference = *x as f64 - *y as f64;
        sum += difference * difference;
        reference += *y as f64 * *y as f64;
        maximum = maximum.max(difference.abs());
    }
    let rms = (sum / a.len() as f64).sqrt();
    (
        rms,
        maximum,
        rms / (reference / a.len() as f64).sqrt().max(1e-12),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use har_core::{BackendKind, KernelKind};
    use har_ir::{DispatchShape, PhysicalOperation};
    struct Adapter;
    impl OperationAdapter for Adapter {
        fn backend(&self) -> BackendKind {
            BackendKind::Cpu
        }
        fn dispatch(&mut self, _: &PhysicalOperation, input: &[f32]) -> Result<DispatchOutput> {
            Ok(DispatchOutput {
                output: input.iter().map(|x| x * 2.0).collect(),
                reference_output: input.iter().map(|x| x * 2.0).collect(),
                bytes_moved: 16,
                note: "test adapter".into(),
            })
        }
    }
    #[test]
    fn indexed_executor_owns_order_and_compare() {
        let mut plan = ExecutionPlan {
            schema: har_plan::PLAN_SCHEMA.into(),
            plan_id: "test".into(),
            plan_kind: "test".into(),
            generated_at_unix_ns: 0,
            model_identity: "m".into(),
            model_sha256: "h".into(),
            hardware: har_core::HardwarePhenotype::synthetic_rdna4(),
            target_context: 1,
            kv_datatype: "f16".into(),
            mtp_enabled: false,
            quality_policy: "exact".into(),
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
        plan.operations.physical.push(PhysicalOperation {
            index: 0,
            logical_id: 0,
            stable_id: "x".into(),
            backend: BackendKind::Cpu,
            kernel: KernelKind::DenseMulMat,
            input_slots: vec![],
            output_slots: vec![],
            dependencies: vec![],
            dispatch: DispatchShape::default(),
            source_tier: MemoryTier::RamMapped,
            destination_tier: MemoryTier::CpuHeap,
        });
        let result = ExecutionEngine::new(plan)
            .execute(0, &[1.0, 2.0], &mut Adapter)
            .unwrap();
        assert!(matches!(result.verdict, DifferentialVerdict::ExactMatch));
    }
}
