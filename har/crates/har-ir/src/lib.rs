//! Canonical operation IR.  Source-language blocks are lowered to this data;
//! no source text or string dispatch is required by the executor.

use har_core::{BackendKind, KernelKind, MemoryTier};
use serde::{Deserialize, Serialize};

pub const IR_SCHEMA: &str = "har.ir.v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TensorHandle {
    pub index: u32,
    pub stable_id: String,
}
impl TensorHandle {
    pub fn new(index: u32, stable_id: impl Into<String>) -> Self {
        Self {
            index,
            stable_id: stable_id.into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct LogicalOperation {
    pub id: u32,
    pub stable_id: String,
    pub kind: LogicalOperationKind,
    pub inputs: Vec<TensorHandle>,
    pub outputs: Vec<TensorHandle>,
    pub dependencies: Vec<u32>,
    pub exactness: har_core::ExactnessMode,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LogicalOperationKind {
    Load,
    MulMat,
    Add,
    Attention,
    MtpVerify,
    Sample,
    Copy,
    Barrier,
    Other,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PhysicalOperation {
    pub index: u32,
    pub logical_id: u32,
    pub stable_id: String,
    pub backend: BackendKind,
    pub kernel: KernelKind,
    pub input_slots: Vec<u32>,
    pub output_slots: Vec<u32>,
    pub dependencies: Vec<u32>,
    pub dispatch: DispatchShape,
    pub source_tier: MemoryTier,
    pub destination_tier: MemoryTier,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct DispatchShape {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub workgroup_x: u32,
    pub workgroup_y: u32,
    pub workgroup_z: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DependencyEdge {
    pub from: u32,
    pub to: u32,
    pub reason: DependencyReason,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DependencyReason {
    Data,
    Residency,
    Fence,
    Epoch,
    Authority,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutionEpoch {
    pub namespace: har_core::EpochNamespace,
    pub generation: u64,
    pub operation_count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct OperationTable {
    pub schema: String,
    pub logical: Vec<LogicalOperation>,
    pub physical: Vec<PhysicalOperation>,
    pub edges: Vec<DependencyEdge>,
}
impl OperationTable {
    pub fn new() -> Self {
        Self {
            schema: IR_SCHEMA.into(),
            ..Self::default()
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        for (index, operation) in self.physical.iter().enumerate() {
            if operation.index as usize != index {
                return Err(format!(
                    "physical operation index {} is not dense",
                    operation.index
                ));
            }
            if operation
                .dependencies
                .iter()
                .any(|dependency| *dependency as usize >= self.physical.len())
            {
                return Err(format!(
                    "operation {} has out-of-range dependency",
                    operation.index
                ));
            }
        }
        // Illegal cycles fail closed: DFS over dependency edges.
        let mut color = vec![0u8; self.physical.len()]; // 0 = white, 1 = grey, 2 = black
        fn visit(
            table: &OperationTable,
            color: &mut [u8],
            node: usize,
            path: &mut Vec<u32>,
        ) -> Result<(), String> {
            match color[node] {
                1 => {
                    let mut cycle: Vec<String> = path.iter().map(|x| x.to_string()).collect();
                    cycle.push(node.to_string());
                    Err(format!("dependency cycle: {}", cycle.join(" -> ")))
                }
                2 => Ok(()),
                _ => {
                    color[node] = 1;
                    path.push(node as u32);
                    for dependency in &table.physical[node].dependencies {
                        visit(table, color, *dependency as usize, path)?;
                    }
                    path.pop();
                    color[node] = 2;
                    Ok(())
                }
            }
        }
        for start in 0..self.physical.len() {
            let mut path = Vec::new();
            visit(self, &mut color, start, &mut path)?;
        }
        Ok(())
    }
    pub fn operation(&self, index: u32) -> Option<&PhysicalOperation> {
        self.physical.get(index as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn table_has_indexed_dispatch() {
        let mut t = OperationTable::new();
        t.physical.push(PhysicalOperation {
            index: 0,
            logical_id: 0,
            stable_id: "op".into(),
            backend: BackendKind::Cpu,
            kernel: KernelKind::Q4KMatVec,
            input_slots: vec![],
            output_slots: vec![],
            dependencies: vec![],
            dispatch: DispatchShape::default(),
            source_tier: MemoryTier::RamMapped,
            destination_tier: MemoryTier::CpuHeap,
        });
        assert!(t.validate().is_ok());
    }
    #[test]
    fn dependency_cycles_fail_closed() {
        let mut t = OperationTable::new();
        for index in 0..3 {
            t.physical.push(PhysicalOperation {
                index,
                logical_id: index,
                stable_id: format!("op{index}"),
                backend: BackendKind::Cpu,
                kernel: KernelKind::DenseMulMat,
                input_slots: vec![],
                output_slots: vec![],
                dependencies: vec![(index + 1) % 3],
                dispatch: DispatchShape::default(),
                source_tier: MemoryTier::RamMapped,
                destination_tier: MemoryTier::CpuHeap,
            });
        }
        let error = t.validate().unwrap_err();
        assert!(error.contains("cycle"), "{error}");
    }
    #[test]
    fn dependency_cycles_across_diamond_are_detected() {
        // 0 -> 3 -> {1,2} -> 0 closes a cycle through the diamond.
        let mut t = OperationTable::new();
        let spec: [(u32, Vec<u32>); 4] =
            [(0, vec![3]), (1, vec![0]), (2, vec![0]), (3, vec![1, 2])];
        for (index, (logical_id, dependencies)) in spec.into_iter().enumerate() {
            t.physical.push(PhysicalOperation {
                index: index as u32,
                logical_id,
                stable_id: format!("op{index}"),
                backend: BackendKind::Cpu,
                kernel: KernelKind::DenseMulMat,
                input_slots: vec![],
                output_slots: vec![],
                dependencies,
                dispatch: DispatchShape::default(),
                source_tier: MemoryTier::RamMapped,
                destination_tier: MemoryTier::CpuHeap,
            });
        }
        assert!(t.validate().is_err());
    }
}
