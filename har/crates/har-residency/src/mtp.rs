//! Read-only resource contract for plan importer MTP horizon selection.

use crate::manager::ResidencyManager;
use crate::types::{PageId, PageLocation, ResidencyState, ResourceSnapshot, StorageSlice};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PageResidencyView {
    pub page_id: PageId,
    pub generation: crate::types::Generation,
    pub state: ResidencyState,
    pub locations: Vec<PageLocation>,
    pub bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidencySnapshot {
    pub resources: ResourceSnapshot,
    pub pages: Vec<PageResidencyView>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertUnionEstimate {
    pub unique_pages: u64,
    pub requested_bytes: u64,
    pub ready_projection_count: u64,
    pub resident_projection_count: u64,
    pub ready_expert_overlap: u64,
    pub incremental_nvme_bytes: u64,
    pub incremental_pcie_bytes: u64,
    pub projection_bytes: BTreeMap<u64, u64>,
    pub projection_location: BTreeMap<u64, String>,
    pub resource_snapshot: ResourceSnapshot,
}

impl ExpertUnionEstimate {
    pub fn estimated_additional_nvme_bytes(&self) -> u64 {
        self.incremental_nvme_bytes
    }
    pub fn estimated_pcie_bytes(&self) -> u64 {
        self.incremental_pcie_bytes
    }
}

#[derive(Clone, Debug)]
pub struct MtpResourceInterface<'a> {
    manager: &'a ResidencyManager,
}

impl<'a> MtpResourceInterface<'a> {
    pub fn new(manager: &'a ResidencyManager) -> Self {
        Self { manager }
    }

    pub fn current_residency_snapshot(&self) -> ResidencySnapshot {
        ResidencySnapshot {
            resources: self.manager.snapshot.clone(),
            pages: self
                .manager
                .records
                .values()
                .map(|record| PageResidencyView {
                    page_id: record.page_id.clone(),
                    generation: record.generation,
                    state: record.state.clone(),
                    locations: record.locations().into_iter().collect(),
                    bytes: record.representation.bytes,
                })
                .collect(),
        }
    }

    pub fn snapshot(&self) -> ResidencySnapshot {
        self.current_residency_snapshot()
    }

    pub fn queue_backlog(&self) -> (u32, u32, u32) {
        let snapshot = &self.manager.snapshot;
        (
            snapshot.nvme_queue_depth,
            snapshot.transfer_queue_depth,
            snapshot.compute_queue_depth,
        )
    }

    pub fn resource_slack(&self) -> (u64, u64, u32, u32, u32) {
        let snapshot = &self.manager.snapshot;
        (
            snapshot.ram_pinned_slack(),
            snapshot.vram_slot_slack(),
            snapshot.nvme_queue_slack(),
            snapshot.transfer_queue_slack(),
            snapshot.compute_queue_slack(),
        )
    }

    pub fn expert_union_incremental_cost(&self, slices: &[StorageSlice]) -> ExpertUnionEstimate {
        let mut unique = BTreeMap::new();
        for slice in slices {
            unique.entry(slice.page_id.ordinal).or_insert(slice);
        }
        let mut ready = 0;
        let mut resident = 0;
        let mut ready_experts = std::collections::BTreeSet::new();
        let mut nvme = 0;
        let mut pcie = 0;
        let mut locations = BTreeMap::new();
        let mut bytes = BTreeMap::new();
        for (ordinal, slice) in unique {
            bytes.insert(ordinal, slice.payload_bytes);
            let record = self.manager.records.get(&slice.page_id);
            let location = if let Some(record) = record {
                if record.replicas.keys().any(PageLocation::is_vram)
                    && matches!(
                        record.state,
                        ResidencyState::Uploaded
                            | ResidencyState::ComputeReady
                            | ResidencyState::InUse
                    )
                {
                    ready += 1;
                    if let Some(expert) = slice.expert {
                        ready_experts.insert((slice.layer.unwrap_or(0), expert));
                    }
                    resident += 1;
                    "VRAM_SLOT"
                } else if record.replicas.keys().any(PageLocation::is_ram) {
                    resident += 1;
                    pcie += slice.payload_bytes;
                    "RAM_PINNED"
                } else {
                    nvme += slice.payload_bytes;
                    pcie += slice.payload_bytes;
                    "NVME_COLD"
                }
            } else {
                nvme += slice.payload_bytes;
                pcie += slice.payload_bytes;
                "NVME_COLD"
            };
            locations.insert(ordinal, location.into());
        }
        ExpertUnionEstimate {
            unique_pages: bytes.len() as u64,
            requested_bytes: bytes.values().sum(),
            ready_projection_count: ready,
            resident_projection_count: resident,
            ready_expert_overlap: ready_experts.len() as u64,
            incremental_nvme_bytes: nvme,
            incremental_pcie_bytes: pcie,
            projection_bytes: bytes,
            projection_location: locations,
            resource_snapshot: self.manager.snapshot.clone(),
        }
    }

    pub fn resource_contract(
        &self,
        slices: &[StorageSlice],
    ) -> (ResidencySnapshot, ExpertUnionEstimate) {
        (
            self.current_residency_snapshot(),
            self.expert_union_incremental_cost(slices),
        )
    }
}
