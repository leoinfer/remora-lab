//! Typed decode-control structures produced by the HAR language compiler.
//! These values are immutable runtime inputs; source text never enters the
//! token loop.

use har_core::{BackendKind, ExactnessMode, FallbackContract, TelemetryContract};
use har_core::{KernelKind, MemoryTier};
use har_ir::{
    DispatchShape, LogicalOperation, LogicalOperationKind, OperationTable, PhysicalOperation,
    TensorHandle,
};
use serde::{Deserialize, Serialize};

pub const DECODE_CONTROL_SCHEMA: &str = "har.decode_control.v1";

/// Language-compiler-owned, parser-free startup plan surface. The existing
/// compiler decode-control descriptor remains intact; this additive module is
/// the versioned join for HAR language V0.
pub mod language;
pub mod metabolism;

pub use metabolism::{DecodeMetabolismConfig, DecodeMetabolismGate, DecodeObservation};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecodeControl {
    pub schema: String,
    pub name: String,
    pub model: String,
    pub backend: BackendKind,
    pub target_context: u32,
    pub kv_datatype: String,
    pub horizon_start: u32,
    pub horizon_end: u32,
    pub strict: bool,
    pub exactness: ExactnessMode,
    pub fallback: FallbackContract,
    pub telemetry: TelemetryContract,
    #[serde(default)]
    pub metabolism: DecodeMetabolismConfig,
}
impl Default for DecodeControl {
    fn default() -> Self {
        Self {
            schema: DECODE_CONTROL_SCHEMA.into(),
            name: "default".into(),
            model: String::new(),
            backend: BackendKind::Cpu,
            target_context: 4096,
            kv_datatype: "f16".into(),
            horizon_start: 0,
            horizon_end: 0,
            strict: true,
            exactness: ExactnessMode::Exact,
            fallback: FallbackContract::default(),
            telemetry: TelemetryContract::default(),
            metabolism: DecodeMetabolismConfig::default(),
        }
    }
}
impl DecodeControl {
    pub fn lower_operation_table(&self) -> OperationTable {
        let mut table = OperationTable::new();
        table.logical.push(LogicalOperation {
            id: 0,
            stable_id: format!("{}.decode", self.name),
            kind: LogicalOperationKind::MtpVerify,
            inputs: vec![TensorHandle::new(0, "input")],
            outputs: vec![TensorHandle::new(1, "output")],
            dependencies: vec![],
            exactness: self.exactness.clone(),
        });
        table.physical.push(PhysicalOperation {
            index: 0,
            logical_id: 0,
            stable_id: format!("{}.decode", self.name),
            backend: self.backend.clone(),
            kernel: KernelKind::MtpVerify,
            input_slots: vec![0],
            output_slots: vec![1],
            dependencies: vec![],
            dispatch: DispatchShape::default(),
            source_tier: MemoryTier::RamMapped,
            destination_tier: MemoryTier::CpuHeap,
        });
        table
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CapabilityRequest {
    pub backend: BackendKind,
    pub required_formats: Vec<String>,
    pub required_kernels: Vec<String>,
}
