//! Generic physical page identities and ownership records.
//!
//! Weight pages and KV pages share the allocator vocabulary but not a semantic
//! identity: `PageKind` is part of `PageId`, so a KV page can never satisfy a
//! weight lookup by accident.

use std::collections::BTreeMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

pub type Result<T> = std::result::Result<T, ResidencyError>;

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0)
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ModelRoot(pub String);

impl ModelRoot {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum PageKind {
    Weights,
    Kv,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Generation(pub u64);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Epoch(pub u64);

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct PageId {
    pub model_root: ModelRoot,
    pub kind: PageKind,
    pub ordinal: u64,
}

impl PageId {
    pub fn weights(model_root: impl Into<String>, ordinal: u64) -> Self {
        Self {
            model_root: ModelRoot::new(model_root),
            kind: PageKind::Weights,
            ordinal,
        }
    }

    pub fn kv(model_root: impl Into<String>, ordinal: u64) -> Self {
        Self {
            model_root: ModelRoot::new(model_root),
            kind: PageKind::Kv,
            ordinal,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct RepresentationId {
    pub page_id: PageId,
    pub format: String,
    pub bytes: u64,
}

impl RepresentationId {
    pub fn new(page_id: PageId, format: impl Into<String>, bytes: u64) -> Self {
        Self {
            page_id,
            format: format.into(),
            bytes,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ExpertProjection {
    Gate,
    Up,
    Down,
    Other(String),
}

impl ExpertProjection {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
            Self::Other(value) => value.as_str(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StorageSlice {
    pub page_id: PageId,
    pub model_id: String,
    pub source_path: String,
    pub shard: String,
    pub tensor: String,
    pub offset: u64,
    pub payload_bytes: u64,
    pub alignment: u64,
    pub quant_type: String,
    pub layer: Option<u32>,
    pub expert: Option<u32>,
    pub projection: ExpertProjection,
    pub checksum_sha256: Option<String>,
    pub parent_offset: Option<u64>,
    pub parent_payload_bytes: Option<u64>,
}

impl StorageSlice {
    pub fn validate(&self) -> Result<()> {
        if self.model_id.is_empty() || self.source_path.is_empty() || self.tensor.is_empty() {
            return Err(ResidencyError::Invalid(
                "storage identity is incomplete".into(),
            ));
        }
        if self.payload_bytes == 0 || self.alignment == 0 {
            return Err(ResidencyError::Invalid(
                "storage payload and alignment must be non-zero".into(),
            ));
        }
        if let Some(checksum) = &self.checksum_sha256 {
            if checksum.len() != 64 || !checksum.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(ResidencyError::Invalid(
                    "checksum is not a SHA-256 hex digest".into(),
                ));
            }
        }
        // Weight and KV pages use the same physical slice vocabulary, but
        // their PageId.kind remains distinct so the allocator cannot merge
        // their semantics.
        Ok(())
    }

    pub fn representation(&self) -> RepresentationId {
        RepresentationId::new(
            self.page_id.clone(),
            self.quant_type.clone(),
            self.payload_bytes,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum PageLocation {
    NvmeCold,
    RamPageable,
    RamPinned,
    VramPage(u32),
    VramSlot(u32),
    Scratch(u32),
}

impl PageLocation {
    pub fn is_vram(&self) -> bool {
        matches!(self, Self::VramPage(_) | Self::VramSlot(_))
    }

    pub fn is_ram(&self) -> bool {
        matches!(self, Self::RamPageable | Self::RamPinned)
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::NvmeCold => "NVME_COLD",
            Self::RamPageable => "RAM_PAGEABLE",
            Self::RamPinned => "RAM_PINNED",
            Self::VramPage(_) => "VRAM_PAGE",
            Self::VramSlot(_) => "VRAM_SLOT",
            Self::Scratch(_) => "SCRATCH",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PageLease {
    pub lease_id: u64,
    pub page_id: PageId,
    pub generation: Generation,
    pub epoch: Epoch,
    pub location: PageLocation,
    pub purpose: String,
}

impl PageLease {
    pub fn validate_against(&self, record: &ResidencyRecord) -> Result<()> {
        if self.page_id != record.page_id || self.generation != record.generation {
            return Err(ResidencyError::StaleGeneration {
                page_id: self.page_id.clone(),
                expected: record.generation,
                got: self.generation,
            });
        }
        if !record.locations().contains(&self.location) {
            return Err(ResidencyError::NotResident(self.location.clone()));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransferKind {
    NvmeRead,
    RamToVram,
    VramEviction,
    RamEviction,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransferStatus {
    Queued,
    Running,
    Completed,
    Cancelled,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransferTicket {
    pub ticket_id: u64,
    pub page_id: PageId,
    pub generation: Generation,
    pub source: PageLocation,
    pub destination: PageLocation,
    pub kind: TransferKind,
    pub requested_bytes: u64,
    pub unique_bytes: u64,
    pub useful_bytes: u64,
    pub wasted_bytes: u64,
    pub mandatory: bool,
    pub speculative: bool,
    pub status: TransferStatus,
    pub created_ns: u64,
    pub queued_ns: u64,
    pub started_ns: Option<u64>,
    pub first_byte_ns: Option<u64>,
    pub completed_ns: Option<u64>,
    pub error: Option<String>,
}

impl TransferTicket {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ticket_id: u64,
        page_id: PageId,
        generation: Generation,
        source: PageLocation,
        destination: PageLocation,
        kind: TransferKind,
        requested_bytes: u64,
        mandatory: bool,
        speculative: bool,
    ) -> Self {
        let timestamp = now_ns();
        Self {
            ticket_id,
            page_id,
            generation,
            source,
            destination,
            kind,
            requested_bytes,
            unique_bytes: requested_bytes,
            useful_bytes: 0,
            wasted_bytes: 0,
            mandatory,
            speculative,
            status: TransferStatus::Queued,
            created_ns: timestamp,
            queued_ns: timestamp,
            started_ns: None,
            first_byte_ns: None,
            completed_ns: None,
            error: None,
        }
    }

    pub fn start(&mut self) -> Result<()> {
        if self.status != TransferStatus::Queued {
            return Err(ResidencyError::TicketState("ticket is not queued".into()));
        }
        self.status = TransferStatus::Running;
        self.started_ns = Some(now_ns());
        Ok(())
    }

    pub fn first_byte(&mut self) -> Result<()> {
        if self.status != TransferStatus::Running {
            return Err(ResidencyError::TicketState(
                "first byte requires a running ticket".into(),
            ));
        }
        self.first_byte_ns.get_or_insert_with(now_ns);
        Ok(())
    }

    pub fn complete(&mut self) -> Result<()> {
        if !matches!(
            self.status,
            TransferStatus::Queued | TransferStatus::Running
        ) {
            return Err(ResidencyError::TicketState(
                "ticket is not completable".into(),
            ));
        }
        self.status = TransferStatus::Completed;
        self.first_byte_ns.get_or_insert_with(now_ns);
        self.completed_ns = Some(now_ns());
        Ok(())
    }

    pub fn cancel(&mut self, reason: impl Into<String>) -> Result<()> {
        if self.status == TransferStatus::Completed {
            return Err(ResidencyError::TicketState(
                "completed ticket cannot be cancelled".into(),
            ));
        }
        self.status = TransferStatus::Cancelled;
        self.completed_ns = Some(now_ns());
        self.error = Some(reason.into());
        Ok(())
    }

    pub fn fail(&mut self, reason: impl Into<String>) -> Result<()> {
        if self.status == TransferStatus::Completed {
            return Err(ResidencyError::TicketState(
                "completed ticket cannot fail".into(),
            ));
        }
        self.status = TransferStatus::Failed;
        self.completed_ns = Some(now_ns());
        self.error = Some(reason.into());
        Ok(())
    }

    pub fn latency_ns(&self) -> Option<u64> {
        self.completed_ns
            .map(|end| end.saturating_sub(self.queued_ns))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResidencyState {
    Indexed,
    NvmeReadQueued,
    NvmeReading,
    RamReady,
    VramReservationQueued,
    VramReserved,
    UploadQueued,
    VramUploading,
    Uploaded,
    ComputeReady,
    InUse,
    EvictionQueued,
    Evicting,
    Evicted,
    Cancelled,
    Failed,
}

pub fn legal_transition(from: &ResidencyState, to: &ResidencyState) -> bool {
    use ResidencyState::*;
    match from {
        Indexed => matches!(to, NvmeReadQueued | RamReady | Evicted),
        NvmeReadQueued => matches!(to, NvmeReading | Cancelled | Failed),
        NvmeReading => matches!(to, RamReady | Cancelled | Failed),
        RamReady => matches!(
            to,
            VramReservationQueued | UploadQueued | EvictionQueued | Indexed
        ),
        VramReservationQueued => matches!(to, VramReserved | RamReady | Cancelled | Failed),
        VramReserved => matches!(to, UploadQueued | EvictionQueued | Failed),
        UploadQueued => matches!(to, VramUploading | Cancelled | Failed),
        VramUploading => matches!(to, Uploaded | Cancelled | Failed),
        Uploaded => matches!(to, ComputeReady | EvictionQueued | Failed),
        ComputeReady => matches!(to, InUse | EvictionQueued | RamReady | Indexed),
        InUse => matches!(to, ComputeReady | EvictionQueued | Failed),
        EvictionQueued => matches!(to, Evicting | Cancelled | Failed),
        Evicting => matches!(to, RamReady | Indexed | Evicted | Failed),
        Evicted => matches!(to, NvmeReadQueued | Indexed),
        Cancelled => matches!(to, Indexed | Evicted),
        Failed => false,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Replica {
    pub location: PageLocation,
    pub representation: RepresentationId,
    pub generation: Generation,
    pub explicit_replication: bool,
    pub replica_id: String,
    pub checksum_sha256: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidencyRecord {
    pub page_id: PageId,
    pub representation: RepresentationId,
    pub generation: Generation,
    pub state: ResidencyState,
    pub authoritative: PageLocation,
    pub replicas: BTreeMap<PageLocation, Replica>,
    pub leases: BTreeMap<u64, PageLease>,
    pub transition_history: Vec<String>,
}

impl ResidencyRecord {
    pub fn new(page_id: PageId, representation: RepresentationId, generation: Generation) -> Self {
        Self {
            page_id,
            representation,
            generation,
            state: ResidencyState::Indexed,
            authoritative: PageLocation::NvmeCold,
            replicas: BTreeMap::new(),
            leases: BTreeMap::new(),
            transition_history: Vec::new(),
        }
    }

    pub fn locations(&self) -> std::collections::BTreeSet<PageLocation> {
        let mut locations = std::collections::BTreeSet::new();
        locations.insert(self.authoritative.clone());
        locations.extend(self.replicas.keys().cloned());
        locations
    }

    pub fn has_replica(&self, location: &PageLocation) -> bool {
        self.replicas.contains_key(location) || &self.authoritative == location
    }

    pub fn transition(&mut self, next: ResidencyState, reason: impl Into<String>) -> Result<()> {
        if !legal_transition(&self.state, &next) {
            return Err(ResidencyError::IllegalTransition {
                from: self.state.clone(),
                to: next,
            });
        }
        let reason = reason.into();
        self.transition_history
            .push(format!("{:?}->{:?}:{}", self.state, next, reason));
        self.state = next;
        Ok(())
    }

    pub fn add_replica(
        &mut self,
        location: PageLocation,
        generation: Generation,
        replica_id: impl Into<String>,
        checksum_sha256: Option<String>,
    ) -> Result<()> {
        if location == self.authoritative {
            return Err(ResidencyError::Invalid(
                "authoritative location is not a replica".into(),
            ));
        }
        if generation != self.generation {
            return Err(ResidencyError::StaleGeneration {
                page_id: self.page_id.clone(),
                expected: self.generation,
                got: generation,
            });
        }
        let replica_id = replica_id.into();
        if replica_id.is_empty() {
            return Err(ResidencyError::Invalid("replica id is empty".into()));
        }
        self.replicas.insert(
            location.clone(),
            Replica {
                location,
                representation: self.representation.clone(),
                generation,
                explicit_replication: true,
                replica_id,
                checksum_sha256,
            },
        );
        Ok(())
    }

    pub fn remove_replica(&mut self, location: &PageLocation) -> Result<()> {
        if location == &self.authoritative {
            return Err(ResidencyError::Invalid(
                "cannot remove authoritative location".into(),
            ));
        }
        self.replicas.remove(location);
        Ok(())
    }

    pub fn lease(
        &mut self,
        lease_id: u64,
        epoch: Epoch,
        location: PageLocation,
        purpose: impl Into<String>,
    ) -> Result<PageLease> {
        if !self.has_replica(&location) {
            return Err(ResidencyError::NotResident(location));
        }
        let lease = PageLease {
            lease_id,
            page_id: self.page_id.clone(),
            generation: self.generation,
            epoch,
            location,
            purpose: purpose.into(),
        };
        self.leases.insert(lease_id, lease.clone());
        Ok(lease)
    }

    pub fn release_lease(&mut self, lease: &PageLease) -> Result<()> {
        lease.validate_against(self)?;
        self.leases.remove(&lease.lease_id);
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        if self.authoritative != PageLocation::NvmeCold
            && !self.has_replica(&PageLocation::NvmeCold)
        {
            return Err(ResidencyError::Invalid(
                "non-NVMe authority lacks explicit NVMe replica".into(),
            ));
        }
        for (location, replica) in &self.replicas {
            if !replica.explicit_replication || location != &replica.location {
                return Err(ResidencyError::Invalid(
                    "replica lacks explicit location metadata".into(),
                ));
            }
            if replica.generation != self.generation
                || replica.representation != self.representation
            {
                return Err(ResidencyError::StaleGeneration {
                    page_id: self.page_id.clone(),
                    expected: self.generation,
                    got: replica.generation,
                });
            }
        }
        match self.state {
            ResidencyState::RamReady if !self.replicas.keys().any(PageLocation::is_ram) => {
                return Err(ResidencyError::Invalid(
                    "RAM_READY without RAM replica".into(),
                ))
            }
            ResidencyState::Uploaded | ResidencyState::ComputeReady | ResidencyState::InUse
                if !self.replicas.keys().any(PageLocation::is_vram) =>
            {
                return Err(ResidencyError::Invalid(
                    "VRAM-ready state without VRAM replica".into(),
                ))
            }
            ResidencyState::InUse if self.leases.is_empty() => {
                return Err(ResidencyError::Invalid("IN_USE without a lease".into()))
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ResourceSnapshot {
    pub epoch: Epoch,
    pub ram_pageable_capacity: u64,
    pub ram_pageable_used: u64,
    pub ram_pinned_capacity: u64,
    pub ram_pinned_used: u64,
    pub vram_page_capacity: u64,
    pub vram_page_used: u64,
    pub vram_slot_capacity: u64,
    pub vram_slot_used: u64,
    pub scratch_capacity: u64,
    pub scratch_used: u64,
    pub nvme_queue_depth: u32,
    pub nvme_queue_capacity: u32,
    pub transfer_queue_depth: u32,
    pub transfer_queue_capacity: u32,
    pub compute_queue_depth: u32,
    pub compute_queue_capacity: u32,
    pub requested_bytes: u64,
    pub unique_bytes: u64,
    pub useful_bytes: u64,
    pub wasted_bytes: u64,
    pub hidden_time_ns: u64,
    pub exposed_time_ns: u64,
    pub vram_hits: u64,
    pub ram_hits: u64,
    pub nvme_misses: u64,
}

impl ResourceSnapshot {
    pub fn ram_pinned_slack(&self) -> u64 {
        self.ram_pinned_capacity
            .saturating_sub(self.ram_pinned_used)
    }
    pub fn vram_slot_slack(&self) -> u64 {
        self.vram_slot_capacity.saturating_sub(self.vram_slot_used)
    }
    pub fn nvme_queue_slack(&self) -> u32 {
        self.nvme_queue_capacity
            .saturating_sub(self.nvme_queue_depth)
    }
    pub fn transfer_queue_slack(&self) -> u32 {
        self.transfer_queue_capacity
            .saturating_sub(self.transfer_queue_depth)
    }
    pub fn compute_queue_slack(&self) -> u32 {
        self.compute_queue_capacity
            .saturating_sub(self.compute_queue_depth)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EvictionReason {
    Capacity,
    Deadline,
    StaleGeneration,
    ExplicitRelease,
    Pressure,
    SpeculativeWaste,
    InUse,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResidencyError {
    Invalid(String),
    IllegalTransition {
        from: ResidencyState,
        to: ResidencyState,
    },
    StaleGeneration {
        page_id: PageId,
        expected: Generation,
        got: Generation,
    },
    NotResident(PageLocation),
    ResourceExhausted(String),
    QueueFull(String),
    TicketState(String),
    Io(String),
    Cancelled,
    Unsupported(String),
}

impl fmt::Display for ResidencyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(formatter, "invalid residency state: {message}"),
            Self::IllegalTransition { from, to } => {
                write!(formatter, "illegal transition {from:?}->{to:?}")
            }
            Self::StaleGeneration {
                page_id,
                expected,
                got,
            } => write!(
                formatter,
                "stale generation for {page_id:?}: expected {expected:?}, got {got:?}"
            ),
            Self::NotResident(location) => {
                write!(formatter, "page is not resident at {location:?}")
            }
            Self::ResourceExhausted(message)
            | Self::QueueFull(message)
            | Self::TicketState(message)
            | Self::Io(message)
            | Self::Unsupported(message) => write!(formatter, "{message}"),
            Self::Cancelled => write!(formatter, "transfer cancelled"),
        }
    }
}

impl std::error::Error for ResidencyError {}
