//! Typed events emitted by the runtime.  Human-readable serialization is a
//! telemetry concern; hot-path code records enum tags and indexed IDs.

use har_core::{BackendKind, EpochNamespace, KernelKind, MemoryTier, ResidencyState};
use serde::{Deserialize, Serialize};

pub const EVENTS_SCHEMA: &str = "har.events.v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EventHeader {
    pub sequence: u64,
    pub timestamp_ns: u64,
    pub epoch: EpochNamespace,
    pub operation_index: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RuntimeEventKind {
    PlanLoaded {
        plan_hash: String,
    },
    PlanValidated {
        warnings: u32,
    },
    Residency {
        resource_id: String,
        from: ResidencyState,
        to: ResidencyState,
        reason: String,
    },
    TransferQueued {
        resource_id: String,
        source: MemoryTier,
        destination: MemoryTier,
        bytes: u64,
    },
    TransferCompleted {
        resource_id: String,
        bytes: u64,
        useful: bool,
    },
    Dispatch {
        backend: BackendKind,
        kernel: KernelKind,
        operation_id: String,
    },
    Output {
        output_hash: String,
        elements: u64,
    },
    Fallback {
        reason: String,
    },
    SpeculationResolved {
        block_index: u64,
        candidate_horizon: u64,
        accepted_tokens: u64,
        rejected_tokens: u64,
    },
    PortionDecision {
        action_id: String,
        decision: String,
        reason: String,
        expected_value: i64,
    },
    ReserveChange {
        dimension: String,
        capacity: u64,
        committed: u64,
        available: u64,
        debt: u64,
    },
    ArtifactSalvaged {
        artifact_id: String,
        class: String,
        tiers: Vec<String>,
    },
    ArtifactInvalidated {
        artifact_id: String,
        reason: String,
        generation_present: u64,
        generation_observed: u64,
    },
    Reclaimed {
        artifact_id: String,
        class: String,
    },
    /// Additive REMORA snapshot. Energy is serialized as an explicit label;
    /// this event never turns UNKNOWN into a numeric measurement.
    MetabolismSnapshot {
        exact_tokens: u64,
        maintenance_vram_mib: u64,
        reserve_vram_mib: u64,
        reserve_ram_mib: u64,
        safe_surplus_mib: u64,
        optional_budget_mib: u64,
        reclaimed: u64,
        salvaged: u64,
        waste_spec_compute_ms: u64,
        waste_prefetch_unused_mib: u64,
        reuse_credit_ms: u64,
        overlap_credit_ms: u64,
        reserve_debt_mib: u64,
        fast_epoch: u64,
        slow_epoch: u64,
        energy: String,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeEvent {
    pub header: EventHeader,
    pub kind: RuntimeEventKind,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EventBuffer {
    pub schema: String,
    pub capacity: usize,
    pub events: Vec<RuntimeEvent>,
    pub dropped: u64,
}
impl EventBuffer {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            schema: EVENTS_SCHEMA.into(),
            capacity,
            events: Vec::with_capacity(capacity),
            dropped: 0,
        }
    }
    pub fn push(&mut self, event: RuntimeEvent) {
        if self.events.len() < self.capacity {
            self.events.push(event);
        } else {
            self.dropped += 1;
        }
    }
    pub fn len(&self) -> usize {
        self.events.len()
    }
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}
