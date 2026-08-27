//! Explicit memory/residency ownership.  No backend handle is exposed here.

use har_core::{EpochNamespace, HarError, MemoryTier, ModelRoot, ResidencyState, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const MEMORY_INTERFACE: &str = "har.memory.v1";

pub mod elastic_budget;
pub mod pinned_pool;

pub use elastic_budget::{
    AdjustmentKind, BudgetAdjustment, ElasticBudgetPolicy, ElasticVramBudget, EnginePhase,
};
pub use pinned_pool::{PinnedHandle, PinnedHostPool, PinnedPoolConfig, PinnedPoolStats};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct BufferId {
    pub index: u32,
    pub generation: u64,
    pub stable_id: String,
}
impl BufferId {
    pub fn new(index: u32, generation: u64, stable_id: impl Into<String>) -> Self {
        Self {
            index,
            generation,
            stable_id: stable_id.into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidencyEvent {
    pub sequence: u64,
    pub resource_id: String,
    pub from: ResidencyState,
    pub to: ResidencyState,
    pub reason: String,
    pub namespace: EpochNamespace,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidencyMachine {
    pub resource_id: String,
    pub generation: u64,
    pub state: ResidencyState,
    pub namespace: EpochNamespace,
    pub events: Vec<ResidencyEvent>,
}
impl ResidencyMachine {
    pub fn new(
        resource_id: impl Into<String>,
        model_root: impl Into<ModelRoot>,
        sequence_id: u64,
    ) -> Self {
        Self {
            resource_id: resource_id.into(),
            generation: 0,
            state: ResidencyState::Unavailable,
            namespace: EpochNamespace::new(model_root, sequence_id),
            events: Vec::new(),
        }
    }
    pub fn transition(&mut self, next: ResidencyState, reason: impl Into<String>) -> Result<()> {
        if !self.state.can_transition_to(&next) {
            return Err(HarError::Invalid {
                kind: "residency transition",
                message: format!("{} -> {}", self.state, next),
            });
        }
        let event = ResidencyEvent {
            sequence: self.events.len() as u64,
            resource_id: self.resource_id.clone(),
            from: self.state.clone(),
            to: next.clone(),
            reason: reason.into(),
            namespace: self.namespace.clone(),
        };
        self.state = next;
        self.events.push(event);
        Ok(())
    }
    pub fn fail(&mut self, reason: impl Into<String>) {
        let _ = self.transition(ResidencyState::Error, reason);
    }
    pub fn bump_generation(&mut self) {
        self.generation = self.generation.saturating_add(1);
        self.namespace.graph_generation = self.generation;
    }
    pub fn require(&self, expected: &ResidencyState, generation: u64) -> Result<()> {
        if &self.state != expected {
            return Err(HarError::Invalid {
                kind: "residency state",
                message: format!("expected {expected}, got {}", self.state),
            });
        }
        if generation != self.generation {
            return Err(HarError::IdentityMismatch {
                field: "generation".into(),
                expected: self.generation.to_string(),
                actual: generation.to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResidencyRecord {
    pub buffer: BufferId,
    pub resource_id: String,
    pub authoritative: MemoryTier,
    pub replicas: Vec<MemoryTier>,
    pub machine: ResidencyMachine,
    pub bytes: u64,
}
impl ResidencyRecord {
    pub fn new(
        buffer: BufferId,
        resource_id: impl Into<String>,
        bytes: u64,
        model_root: impl Into<ModelRoot>,
    ) -> Self {
        let id = resource_id.into();
        Self {
            buffer,
            resource_id: id.clone(),
            authoritative: MemoryTier::NvmeCold,
            replicas: Vec::new(),
            machine: ResidencyMachine::new(id, model_root, 0),
            bytes,
        }
    }
    pub fn add_replica(&mut self, tier: MemoryTier) {
        if !self.replicas.contains(&tier) {
            self.replicas.push(tier);
        }
    }
    pub fn has_replica(&self, tier: &MemoryTier) -> bool {
        &self.authoritative == tier || self.replicas.contains(tier)
    }
    pub fn validate(&self) -> Result<()> {
        if self.bytes == 0 {
            return Err(HarError::Invalid {
                kind: "residency record",
                message: "zero-byte resource".into(),
            });
        }
        if self.machine.state == ResidencyState::ReadyVram
            && !self.has_replica(&MemoryTier::VramResident)
            && !self.has_replica(&MemoryTier::VramSlot)
        {
            return Err(HarError::Invalid {
                kind: "residency record",
                message: "READY_VRAM without a VRAM replica".into(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TransferTicket {
    pub id: u64,
    pub resource_id: String,
    pub generation: u64,
    pub source: MemoryTier,
    pub destination: MemoryTier,
    pub bytes: u64,
    pub queue: String,
    pub mandatory: bool,
    pub completed: bool,
    pub useful: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct MemoryRegistry {
    pub schema: String,
    pub records: BTreeMap<String, ResidencyRecord>,
    pub transfers: Vec<TransferTicket>,
    pub next_transfer_id: u64,
}
impl MemoryRegistry {
    pub fn new() -> Self {
        Self {
            schema: MEMORY_INTERFACE.into(),
            ..Self::default()
        }
    }
    pub fn register(&mut self, record: ResidencyRecord) -> Result<()> {
        record.validate()?;
        if self.records.contains_key(&record.resource_id) {
            return Err(HarError::Invalid {
                kind: "memory registry",
                message: format!("duplicate resource {}", record.resource_id),
            });
        }
        self.records.insert(record.resource_id.clone(), record);
        Ok(())
    }
    pub fn get(&self, id: &str) -> Option<&ResidencyRecord> {
        self.records.get(id)
    }
    pub fn get_mut(&mut self, id: &str) -> Option<&mut ResidencyRecord> {
        self.records.get_mut(id)
    }
    pub fn queue_transfer(
        &mut self,
        resource_id: &str,
        source: MemoryTier,
        destination: MemoryTier,
        bytes: u64,
        queue: impl Into<String>,
        mandatory: bool,
    ) -> Result<u64> {
        let record = self
            .records
            .get(resource_id)
            .ok_or_else(|| HarError::Invalid {
                kind: "memory registry",
                message: format!("unknown resource {resource_id}"),
            })?;
        if record.bytes < bytes {
            return Err(HarError::Invalid {
                kind: "transfer",
                message: "requested bytes exceed resource".into(),
            });
        }
        let id = self.next_transfer_id;
        self.next_transfer_id += 1;
        self.transfers.push(TransferTicket {
            id,
            resource_id: resource_id.into(),
            generation: record.machine.generation,
            source,
            destination,
            bytes,
            queue: queue.into(),
            mandatory,
            completed: false,
            useful: false,
        });
        Ok(id)
    }
    pub fn complete_transfer(&mut self, id: u64, useful: bool) -> Result<()> {
        let ticket = self
            .transfers
            .iter_mut()
            .find(|x| x.id == id)
            .ok_or_else(|| HarError::Invalid {
                kind: "transfer",
                message: format!("unknown ticket {id}"),
            })?;
        ticket.completed = true;
        ticket.useful = useful;
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResourceBudgetSnapshot {
    pub epoch: u64,
    pub ram_bytes: u64,
    pub vram_bytes: u64,
    pub staging_bytes: u64,
    pub scratch_bytes: u64,
    pub kv_bytes: u64,
    pub in_flight_transfers: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn state_machine_requires_order_and_generation() {
        let mut machine = ResidencyMachine::new("w", "m", 1);
        assert!(machine.transition(ResidencyState::Indexed, "index").is_ok());
        assert!(machine
            .transition(ResidencyState::ReadQueued, "queue")
            .is_ok());
        assert!(machine
            .transition(ResidencyState::ReadyVram, "bad skip")
            .is_err());
        machine.bump_generation();
        assert!(machine.require(&ResidencyState::ReadQueued, 0).is_err());
    }
    #[test]
    fn registry_tracks_explicit_transfer() {
        let mut registry = MemoryRegistry::new();
        let record = ResidencyRecord::new(BufferId::new(0, 0, "w"), "w", 1024, "m");
        registry.register(record).unwrap();
        let ticket = registry
            .queue_transfer(
                "w",
                MemoryTier::NvmeCold,
                MemoryTier::RamMapped,
                1024,
                "nvme",
                true,
            )
            .unwrap();
        registry.complete_transfer(ticket, true).unwrap();
        assert!(registry.transfers[0].useful);
    }
}
