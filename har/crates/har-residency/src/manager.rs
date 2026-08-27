//! Ownership ledger, persistent slots and explicit transfer transitions.

use crate::types::*;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SlotState {
    Free,
    Reserved,
    Uploading,
    Uploaded,
    ComputeReady,
    InUse,
    Evicting,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VramSlot {
    pub slot_id: u32,
    pub capacity_bytes: u64,
    pub state: SlotState,
    pub page_id: Option<PageId>,
    pub generation: Option<Generation>,
    pub projection: Option<ExpertProjection>,
    pub reserved_bytes: u64,
    pub pinned_users: u32,
    pub upload_ticket: Option<u64>,
    pub checksum_sha256: Option<String>,
    pub last_use_epoch: Epoch,
    pub speculative: bool,
}

impl VramSlot {
    fn new(slot_id: u32, capacity_bytes: u64) -> Self {
        Self {
            slot_id,
            capacity_bytes,
            state: SlotState::Free,
            page_id: None,
            generation: None,
            projection: None,
            reserved_bytes: 0,
            pinned_users: 0,
            upload_ticket: None,
            checksum_sha256: None,
            last_use_epoch: Epoch(0),
            speculative: false,
        }
    }

    pub fn ready(&self) -> bool {
        matches!(self.state, SlotState::ComputeReady | SlotState::InUse)
    }

    pub fn evictable(&self) -> bool {
        matches!(self.state, SlotState::ComputeReady | SlotState::Uploaded)
            && self.pinned_users == 0
    }
}

#[derive(Clone, Debug)]
pub struct VramSlotTable {
    slots: Vec<VramSlot>,
    next_epoch: Epoch,
}

impl VramSlotTable {
    pub fn new(slot_count: u32, slot_bytes: u64) -> Result<Self> {
        if slot_count == 0 || slot_bytes == 0 {
            return Err(ResidencyError::Invalid(
                "VRAM slot count and size must be non-zero".into(),
            ));
        }
        Ok(Self {
            slots: (0..slot_count)
                .map(|id| VramSlot::new(id, slot_bytes))
                .collect(),
            next_epoch: Epoch(0),
        })
    }

    pub fn capacity_bytes(&self) -> u64 {
        self.slots.iter().map(|slot| slot.capacity_bytes).sum()
    }

    pub fn slots(&self) -> &[VramSlot] {
        &self.slots
    }

    pub fn get(&self, slot_id: u32) -> Result<&VramSlot> {
        self.slots
            .get(slot_id as usize)
            .ok_or_else(|| ResidencyError::Invalid(format!("unknown VRAM slot {slot_id}")))
    }

    fn get_mut(&mut self, slot_id: u32) -> Result<&mut VramSlot> {
        self.slots
            .get_mut(slot_id as usize)
            .ok_or_else(|| ResidencyError::Invalid(format!("unknown VRAM slot {slot_id}")))
    }

    pub fn find_ready(&self, page_id: &PageId, generation: Generation) -> Option<&VramSlot> {
        self.slots.iter().find(|slot| {
            slot.ready()
                && slot.page_id.as_ref() == Some(page_id)
                && slot.generation == Some(generation)
        })
    }

    pub fn reserve(
        &mut self,
        page_id: &PageId,
        generation: Generation,
        projection: ExpertProjection,
        bytes: u64,
        speculative: bool,
    ) -> Result<u32> {
        if bytes == 0 {
            return Err(ResidencyError::Invalid(
                "cannot reserve zero-byte slot".into(),
            ));
        }
        if let Some(slot) = self.slots.iter().find(|slot| {
            slot.page_id.as_ref() == Some(page_id)
                && slot.generation == Some(generation)
                && slot.projection.as_ref() == Some(&projection)
                && !matches!(slot.state, SlotState::Free)
        }) {
            return Ok(slot.slot_id);
        }
        // Slot reuse is never implicit.  The residency manager must first
        // evict the old page and remove its replica metadata; otherwise a
        // free-list operation could leave two pages claiming one slot.
        let candidate = self
            .slots
            .iter()
            .position(|slot| slot.state == SlotState::Free);
        let index = candidate.ok_or_else(|| {
            ResidencyError::ResourceExhausted("no free or evictable VRAM slot".into())
        })?;
        let slot = &mut self.slots[index];
        if bytes > slot.capacity_bytes {
            return Err(ResidencyError::ResourceExhausted(format!(
                "page {bytes} bytes exceeds slot {} bytes",
                slot.capacity_bytes
            )));
        }
        *slot = VramSlot::new(slot.slot_id, slot.capacity_bytes);
        slot.state = SlotState::Reserved;
        slot.page_id = Some(page_id.clone());
        slot.generation = Some(generation);
        slot.projection = Some(projection);
        slot.reserved_bytes = bytes;
        slot.speculative = speculative;
        slot.last_use_epoch = self.next_epoch;
        Ok(slot.slot_id)
    }

    pub fn queue_upload(&mut self, slot_id: u32, ticket_id: u64) -> Result<()> {
        let slot = self.get_mut(slot_id)?;
        if slot.state != SlotState::Reserved {
            return Err(ResidencyError::Invalid(
                "upload requires a reserved slot".into(),
            ));
        }
        slot.state = SlotState::Uploading;
        slot.upload_ticket = Some(ticket_id);
        Ok(())
    }

    pub fn complete_upload(
        &mut self,
        slot_id: u32,
        ticket_id: u64,
        checksum: Option<String>,
    ) -> Result<()> {
        let slot = self.get_mut(slot_id)?;
        if slot.state != SlotState::Uploading || slot.upload_ticket != Some(ticket_id) {
            return Err(ResidencyError::StaleGeneration {
                page_id: slot
                    .page_id
                    .clone()
                    .unwrap_or_else(|| PageId::weights("unknown", 0)),
                expected: slot.generation.unwrap_or(Generation(0)),
                got: Generation(0),
            });
        }
        slot.state = SlotState::Uploaded;
        slot.checksum_sha256 = checksum;
        Ok(())
    }

    pub fn mark_compute_ready(
        &mut self,
        slot_id: u32,
        page_id: &PageId,
        generation: Generation,
    ) -> Result<()> {
        let slot = self.get_mut(slot_id)?;
        validate_slot_identity(slot, page_id, generation)?;
        if slot.state != SlotState::Uploaded {
            return Err(ResidencyError::Invalid(
                "compute readiness requires uploaded slot".into(),
            ));
        }
        slot.state = SlotState::ComputeReady;
        Ok(())
    }

    pub fn acquire(
        &mut self,
        slot_id: u32,
        page_id: &PageId,
        generation: Generation,
        epoch: Epoch,
    ) -> Result<PageLease> {
        let slot = self.get_mut(slot_id)?;
        validate_slot_identity(slot, page_id, generation)?;
        if slot.state != SlotState::ComputeReady {
            return Err(ResidencyError::Invalid("slot is not compute ready".into()));
        }
        slot.state = SlotState::InUse;
        slot.pinned_users = slot.pinned_users.saturating_add(1);
        slot.last_use_epoch = epoch;
        Ok(PageLease {
            lease_id: ((slot_id as u64) << 32) ^ epoch.0,
            page_id: page_id.clone(),
            generation,
            epoch,
            location: PageLocation::VramSlot(slot_id),
            purpose: "gpu-compute".into(),
        })
    }

    pub fn release(&mut self, lease: &PageLease) -> Result<()> {
        let slot = self.get_mut(match lease.location {
            PageLocation::VramSlot(id) => id,
            _ => {
                return Err(ResidencyError::Invalid(
                    "VRAM slot lease has wrong location".into(),
                ))
            }
        })?;
        validate_slot_identity(slot, &lease.page_id, lease.generation)?;
        if slot.state != SlotState::InUse || slot.pinned_users == 0 {
            return Err(ResidencyError::Invalid(
                "slot release without an active pin".into(),
            ));
        }
        slot.pinned_users -= 1;
        if slot.pinned_users == 0 {
            slot.state = SlotState::ComputeReady;
        }
        Ok(())
    }

    pub fn evict(&mut self, slot_id: u32, page_id: &PageId, generation: Generation) -> Result<()> {
        let slot = self.get_mut(slot_id)?;
        validate_slot_identity(slot, page_id, generation)?;
        if !slot.evictable() {
            return Err(ResidencyError::ResourceExhausted(
                "slot is pinned or not evictable".into(),
            ));
        }
        let capacity = slot.capacity_bytes;
        *slot = VramSlot::new(slot_id, capacity);
        Ok(())
    }
}

fn validate_slot_identity(slot: &VramSlot, page_id: &PageId, generation: Generation) -> Result<()> {
    if slot.page_id.as_ref() != Some(page_id) || slot.generation != Some(generation) {
        return Err(ResidencyError::StaleGeneration {
            page_id: page_id.clone(),
            expected: slot.generation.unwrap_or(Generation(0)),
            got: generation,
        });
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct ResidencyManager {
    pub records: HashMap<PageId, ResidencyRecord>,
    pub tickets: HashMap<u64, TransferTicket>,
    pub slots: VramSlotTable,
    pub snapshot: ResourceSnapshot,
    next_ticket: u64,
    next_lease: u64,
    next_epoch: u64,
    ram_pageable_budget: u64,
    ram_pinned_budget: u64,
}

impl ResidencyManager {
    pub fn new(
        ram_pageable_budget: u64,
        ram_pinned_budget: u64,
        slot_count: u32,
        slot_bytes: u64,
        nvme_queue_capacity: u32,
        transfer_queue_capacity: u32,
        compute_queue_capacity: u32,
    ) -> Result<Self> {
        let slots = VramSlotTable::new(slot_count, slot_bytes)?;
        let snapshot = ResourceSnapshot {
            ram_pageable_capacity: ram_pageable_budget,
            ram_pinned_capacity: ram_pinned_budget,
            vram_slot_capacity: slots.capacity_bytes(),
            nvme_queue_capacity,
            transfer_queue_capacity,
            compute_queue_capacity,
            ..ResourceSnapshot::default()
        };
        Ok(Self {
            records: HashMap::new(),
            tickets: HashMap::new(),
            slots,
            snapshot,
            next_ticket: 1,
            next_lease: 1,
            next_epoch: 1,
            ram_pageable_budget,
            ram_pinned_budget,
        })
    }

    pub fn register(&mut self, slice: &StorageSlice, generation: Generation) -> Result<()> {
        slice.validate()?;
        if self.records.contains_key(&slice.page_id) {
            return Err(ResidencyError::Invalid("page already registered".into()));
        }
        self.records.insert(
            slice.page_id.clone(),
            ResidencyRecord::new(slice.page_id.clone(), slice.representation(), generation),
        );
        Ok(())
    }

    pub fn record(&self, page_id: &PageId) -> Result<&ResidencyRecord> {
        self.records
            .get(page_id)
            .ok_or_else(|| ResidencyError::Invalid("unknown page".into()))
    }

    pub fn record_mut(&mut self, page_id: &PageId) -> Result<&mut ResidencyRecord> {
        self.records
            .get_mut(page_id)
            .ok_or_else(|| ResidencyError::Invalid("unknown page".into()))
    }

    #[allow(clippy::too_many_arguments)]
    fn new_ticket(
        &mut self,
        page_id: &PageId,
        kind: TransferKind,
        source: PageLocation,
        destination: PageLocation,
        bytes: u64,
        mandatory: bool,
        speculative: bool,
    ) -> Result<u64> {
        let record = self.record(page_id)?.clone();
        let ticket_id = self.next_ticket;
        self.next_ticket += 1;
        self.tickets.insert(
            ticket_id,
            TransferTicket::new(
                ticket_id,
                page_id.clone(),
                record.generation,
                source,
                destination,
                kind,
                bytes,
                mandatory,
                speculative,
            ),
        );
        Ok(ticket_id)
    }

    pub fn publish_ram_pageable(
        &mut self,
        page_id: &PageId,
        checksum: Option<String>,
    ) -> Result<()> {
        let record = self.record(page_id)?.clone();
        if self
            .snapshot
            .ram_pageable_used
            .saturating_add(record.representation.bytes)
            > self.ram_pageable_budget
        {
            return Err(ResidencyError::ResourceExhausted(
                "pageable RAM budget would be exceeded".into(),
            ));
        }
        if !record.has_replica(&PageLocation::RamPageable) {
            self.record_mut(page_id)?.add_replica(
                PageLocation::RamPageable,
                record.generation,
                format!("ram-pageable:{:?}", page_id.ordinal),
                checksum,
            )?;
            self.snapshot.ram_pageable_used = self
                .snapshot
                .ram_pageable_used
                .saturating_add(record.representation.bytes);
        }
        if record.state == ResidencyState::Indexed {
            self.record_mut(page_id)?
                .transition(ResidencyState::RamReady, "pageable RAM replica published")?;
        }
        Ok(())
    }

    pub fn promote_pageable_to_pinned(
        &mut self,
        page_id: &PageId,
        checksum: Option<String>,
    ) -> Result<()> {
        let record = self.record(page_id)?.clone();
        if !record.has_replica(&PageLocation::RamPageable) {
            return Err(ResidencyError::NotResident(PageLocation::RamPageable));
        }
        if self
            .snapshot
            .ram_pinned_used
            .saturating_add(record.representation.bytes)
            > self.ram_pinned_budget
        {
            return Err(ResidencyError::ResourceExhausted(
                "pinned RAM budget would be exceeded".into(),
            ));
        }
        self.record_mut(page_id)?
            .remove_replica(&PageLocation::RamPageable)?;
        self.record_mut(page_id)?.add_replica(
            PageLocation::RamPinned,
            record.generation,
            format!("ram-pinned:promote:{:?}", page_id.ordinal),
            checksum,
        )?;
        self.snapshot.ram_pageable_used = self
            .snapshot
            .ram_pageable_used
            .saturating_sub(record.representation.bytes);
        self.snapshot.ram_pinned_used = self
            .snapshot
            .ram_pinned_used
            .saturating_add(record.representation.bytes);
        Ok(())
    }

    pub fn queue_nvme_read(
        &mut self,
        page_id: &PageId,
        mandatory: bool,
        speculative: bool,
    ) -> Result<u64> {
        if self.snapshot.nvme_queue_depth >= self.snapshot.nvme_queue_capacity {
            return Err(ResidencyError::QueueFull(
                "NVMe queue has no measured slack".into(),
            ));
        }
        let bytes = self.record(page_id)?.representation.bytes;
        let ticket_id = self.new_ticket(
            page_id,
            TransferKind::NvmeRead,
            PageLocation::NvmeCold,
            PageLocation::RamPinned,
            bytes,
            mandatory,
            speculative,
        )?;
        self.record_mut(page_id)?
            .transition(ResidencyState::NvmeReadQueued, "NVMe read queued")?;
        self.snapshot.nvme_queue_depth += 1;
        self.snapshot.requested_bytes += bytes;
        self.snapshot.unique_bytes += bytes;
        self.snapshot.nvme_misses += 1;
        Ok(ticket_id)
    }

    pub fn start_nvme_read(&mut self, ticket_id: u64) -> Result<()> {
        let ticket = self
            .tickets
            .get(&ticket_id)
            .ok_or_else(|| ResidencyError::Invalid("unknown transfer ticket".into()))?;
        let page_id = ticket.page_id.clone();
        let ticket_generation = ticket.generation;
        let current_generation = self.record(&page_id)?.generation;
        if ticket_generation != current_generation {
            return Err(ResidencyError::StaleGeneration {
                page_id,
                expected: current_generation,
                got: ticket_generation,
            });
        }
        self.tickets.get_mut(&ticket_id).unwrap().start()?;
        self.record_mut(&page_id)?
            .transition(ResidencyState::NvmeReading, "NVMe read started")?;
        Ok(())
    }

    pub fn complete_nvme_read(&mut self, ticket_id: u64, checksum: Option<String>) -> Result<()> {
        let (page_id, generation, bytes) = {
            let ticket = self
                .tickets
                .get(&ticket_id)
                .ok_or_else(|| ResidencyError::Invalid("unknown transfer ticket".into()))?;
            (
                ticket.page_id.clone(),
                ticket.generation,
                ticket.requested_bytes,
            )
        };
        if self.snapshot.ram_pinned_used.saturating_add(bytes) > self.ram_pinned_budget {
            self.tickets
                .get_mut(&ticket_id)
                .unwrap()
                .fail("pinned RAM budget would be exceeded")?;
            self.snapshot.nvme_queue_depth = self.snapshot.nvme_queue_depth.saturating_sub(1);
            return Err(ResidencyError::ResourceExhausted(
                "pinned RAM budget would be exceeded".into(),
            ));
        }
        let ticket = self.tickets.get_mut(&ticket_id).unwrap();
        ticket.first_byte()?;
        ticket.complete()?;
        let record = self.record_mut(&page_id)?;
        if record.generation != generation {
            return Err(ResidencyError::StaleGeneration {
                page_id,
                expected: record.generation,
                got: generation,
            });
        }
        record.add_replica(
            PageLocation::RamPinned,
            generation,
            format!("ram-pinned:{}", ticket_id),
            checksum,
        )?;
        record.transition(
            ResidencyState::RamReady,
            "NVMe payload committed to pinned RAM",
        )?;
        self.snapshot.nvme_queue_depth = self.snapshot.nvme_queue_depth.saturating_sub(1);
        self.snapshot.ram_pinned_used = self.snapshot.ram_pinned_used.saturating_add(bytes);
        Ok(())
    }

    pub fn reserve_vram(
        &mut self,
        page_id: &PageId,
        projection: ExpertProjection,
        _mandatory: bool,
        speculative: bool,
    ) -> Result<u32> {
        let record = self.record(page_id)?.clone();
        if record.state != ResidencyState::RamReady {
            return Err(ResidencyError::IllegalTransition {
                from: record.state,
                to: ResidencyState::VramReservationQueued,
            });
        }
        if !record.has_replica(&PageLocation::RamPinned) {
            return Err(ResidencyError::NotResident(PageLocation::RamPinned));
        }
        if self.snapshot.ram_pinned_used > self.ram_pinned_budget {
            return Err(ResidencyError::ResourceExhausted(
                "pinned RAM budget is already exceeded".into(),
            ));
        }
        self.record_mut(page_id)?.transition(
            ResidencyState::VramReservationQueued,
            "VRAM reservation queued",
        )?;
        let result = self.slots.reserve(
            page_id,
            record.generation,
            projection,
            record.representation.bytes,
            speculative,
        );
        match result {
            Ok(slot_id) => {
                self.record_mut(page_id)?.transition(
                    ResidencyState::VramReserved,
                    "persistent VRAM slot reserved",
                )?;
                self.snapshot.vram_slot_used = self.snapshot.vram_slot_used.saturating_add(0);
                Ok(slot_id)
            }
            Err(error) => {
                self.record_mut(page_id)?.transition(
                    ResidencyState::RamReady,
                    "VRAM reservation refused; RAM retained",
                )?;
                Err(error)
            }
        }
    }

    pub fn queue_upload(
        &mut self,
        page_id: &PageId,
        slot_id: u32,
        mandatory: bool,
        speculative: bool,
    ) -> Result<u64> {
        if self.snapshot.transfer_queue_depth >= self.snapshot.transfer_queue_capacity {
            return Err(ResidencyError::QueueFull(
                "transfer queue has no measured slack".into(),
            ));
        }
        let record = self.record(page_id)?.clone();
        let ticket_id = self.new_ticket(
            page_id,
            TransferKind::RamToVram,
            PageLocation::RamPinned,
            PageLocation::VramSlot(slot_id),
            record.representation.bytes,
            mandatory,
            speculative,
        )?;
        self.slots.queue_upload(slot_id, ticket_id)?;
        self.record_mut(page_id)?
            .transition(ResidencyState::UploadQueued, "RAM to VRAM upload queued")?;
        self.snapshot.transfer_queue_depth += 1;
        Ok(ticket_id)
    }

    pub fn start_upload(&mut self, ticket_id: u64) -> Result<()> {
        let page_id = self
            .tickets
            .get(&ticket_id)
            .ok_or_else(|| ResidencyError::Invalid("unknown transfer ticket".into()))?
            .page_id
            .clone();
        self.tickets.get_mut(&ticket_id).unwrap().start()?;
        self.record_mut(&page_id)?
            .transition(ResidencyState::VramUploading, "RAM to VRAM upload started")?;
        Ok(())
    }

    pub fn complete_upload(
        &mut self,
        ticket_id: u64,
        slot_id: u32,
        checksum: Option<String>,
    ) -> Result<()> {
        let (page_id, generation, bytes) = {
            let ticket = self
                .tickets
                .get(&ticket_id)
                .ok_or_else(|| ResidencyError::Invalid("unknown transfer ticket".into()))?;
            (
                ticket.page_id.clone(),
                ticket.generation,
                ticket.requested_bytes,
            )
        };
        if self.record(&page_id)?.generation != generation {
            return Err(ResidencyError::StaleGeneration {
                page_id: page_id.clone(),
                expected: self.record(&page_id)?.generation,
                got: generation,
            });
        }
        self.slots
            .complete_upload(slot_id, ticket_id, checksum.clone())?;
        let ticket = self.tickets.get_mut(&ticket_id).unwrap();
        ticket.first_byte()?;
        ticket.complete()?;
        let record = self.record_mut(&page_id)?;
        if record.generation != generation {
            return Err(ResidencyError::StaleGeneration {
                page_id,
                expected: record.generation,
                got: generation,
            });
        }
        record.add_replica(
            PageLocation::VramSlot(slot_id),
            generation,
            format!("vram-slot:{}", slot_id),
            checksum,
        )?;
        record.transition(ResidencyState::Uploaded, "VRAM upload completed")?;
        self.snapshot.transfer_queue_depth = self.snapshot.transfer_queue_depth.saturating_sub(1);
        self.snapshot.ram_pinned_used = self.snapshot.ram_pinned_used.saturating_sub(bytes);
        self.snapshot.vram_slot_used = self.snapshot.vram_slot_used.saturating_add(bytes);
        Ok(())
    }

    pub fn mark_compute_ready(&mut self, page_id: &PageId, slot_id: u32) -> Result<()> {
        let record = self.record(page_id)?.clone();
        self.slots
            .mark_compute_ready(slot_id, page_id, record.generation)?;
        self.record_mut(page_id)?.transition(
            ResidencyState::ComputeReady,
            "GPU readiness event published",
        )?;
        Ok(())
    }

    pub fn acquire_compute(&mut self, page_id: &PageId, slot_id: u32) -> Result<PageLease> {
        if self.snapshot.compute_queue_depth >= self.snapshot.compute_queue_capacity {
            return Err(ResidencyError::QueueFull(
                "compute queue has no measured slack".into(),
            ));
        }
        let generation = self.record(page_id)?.generation;
        let epoch = Epoch(self.next_epoch);
        self.next_epoch += 1;
        let mut lease = self.slots.acquire(slot_id, page_id, generation, epoch)?;
        lease.lease_id = self.next_lease;
        self.next_lease += 1;
        self.record_mut(page_id)?
            .leases
            .insert(lease.lease_id, lease.clone());
        self.record_mut(page_id)?
            .transition(ResidencyState::InUse, "GPU compute lease acquired")?;
        self.snapshot.compute_queue_depth += 1;
        Ok(lease)
    }

    pub fn release_compute(&mut self, lease: PageLease) -> Result<()> {
        lease.validate_against(self.record(&lease.page_id)?)?;
        self.slots.release(&lease)?;
        let record = self.record_mut(&lease.page_id)?;
        record.leases.remove(&lease.lease_id);
        record.transition(ResidencyState::ComputeReady, "GPU compute lease released")?;
        self.snapshot.compute_queue_depth = self.snapshot.compute_queue_depth.saturating_sub(1);
        Ok(())
    }

    pub fn evict_vram(&mut self, page_id: &PageId, slot_id: u32) -> Result<EvictionReason> {
        let record = self.record(page_id)?.clone();
        if !record.has_replica(&PageLocation::VramSlot(slot_id)) {
            return Err(ResidencyError::NotResident(PageLocation::VramSlot(slot_id)));
        }
        if !matches!(
            record.state,
            ResidencyState::ComputeReady | ResidencyState::Uploaded
        ) {
            return Err(ResidencyError::Invalid(
                "VRAM eviction while page is in use or not ready".into(),
            ));
        }
        self.record_mut(page_id)?
            .transition(ResidencyState::EvictionQueued, "VRAM eviction queued")?;
        self.record_mut(page_id)?
            .transition(ResidencyState::Evicting, "VRAM eviction started")?;
        let bytes = record.representation.bytes;
        self.slots.evict(slot_id, page_id, record.generation)?;
        self.record_mut(page_id)?
            .remove_replica(&PageLocation::VramSlot(slot_id))?;
        self.record_mut(page_id)?.transition(
            ResidencyState::RamReady,
            "VRAM evicted; pinned RAM retained",
        )?;
        self.snapshot.vram_slot_used = self.snapshot.vram_slot_used.saturating_sub(bytes);
        Ok(EvictionReason::ExplicitRelease)
    }

    pub fn evict_pageable_ram(&mut self, page_id: &PageId) -> Result<EvictionReason> {
        let record = self.record(page_id)?.clone();
        if !record.has_replica(&PageLocation::RamPageable) {
            return Err(ResidencyError::NotResident(PageLocation::RamPageable));
        }
        if !matches!(
            record.state,
            ResidencyState::RamReady | ResidencyState::ComputeReady
        ) {
            return Err(ResidencyError::Invalid(
                "pageable RAM eviction while page is in use".into(),
            ));
        }
        self.record_mut(page_id)?.transition(
            ResidencyState::EvictionQueued,
            "pageable RAM eviction queued",
        )?;
        self.record_mut(page_id)?
            .transition(ResidencyState::Evicting, "pageable RAM eviction started")?;
        self.record_mut(page_id)?
            .remove_replica(&PageLocation::RamPageable)?;
        let remains_vram = self
            .record(page_id)?
            .replicas
            .keys()
            .any(PageLocation::is_vram);
        if remains_vram {
            self.record_mut(page_id)?.transition(
                ResidencyState::ComputeReady,
                "pageable RAM evicted; VRAM retained",
            )?;
        } else {
            self.record_mut(page_id)?
                .transition(ResidencyState::Indexed, "pageable RAM replica evicted")?;
        }
        self.snapshot.ram_pageable_used = self
            .snapshot
            .ram_pageable_used
            .saturating_sub(record.representation.bytes);
        Ok(EvictionReason::ExplicitRelease)
    }

    pub fn evict_ram(&mut self, page_id: &PageId) -> Result<EvictionReason> {
        let record = self.record(page_id)?.clone();
        if !record.has_replica(&PageLocation::RamPinned) {
            return Err(ResidencyError::NotResident(PageLocation::RamPinned));
        }
        if !matches!(
            record.state,
            ResidencyState::RamReady | ResidencyState::ComputeReady
        ) {
            return Err(ResidencyError::Invalid(
                "RAM eviction while page is in use".into(),
            ));
        }
        self.record_mut(page_id)?
            .transition(ResidencyState::EvictionQueued, "RAM eviction queued")?;
        self.record_mut(page_id)?
            .transition(ResidencyState::Evicting, "RAM eviction started")?;
        self.record_mut(page_id)?
            .remove_replica(&PageLocation::RamPinned)?;
        let remains_vram = self
            .record(page_id)?
            .replicas
            .keys()
            .any(PageLocation::is_vram);
        if remains_vram {
            self.record_mut(page_id)?.transition(
                ResidencyState::ComputeReady,
                "RAM replica evicted; VRAM retained",
            )?;
        } else {
            self.record_mut(page_id)?
                .transition(ResidencyState::Indexed, "RAM replica evicted")?;
        }
        self.snapshot.ram_pinned_used = self
            .snapshot
            .ram_pinned_used
            .saturating_sub(record.representation.bytes);
        Ok(EvictionReason::ExplicitRelease)
    }

    pub fn cancel_ticket(&mut self, ticket_id: u64, reason: impl Into<String>) -> Result<()> {
        let ticket = self
            .tickets
            .get_mut(&ticket_id)
            .ok_or_else(|| ResidencyError::Invalid("unknown transfer ticket".into()))?;
        ticket.cancel(reason)?;
        match ticket.kind {
            TransferKind::NvmeRead => {
                self.snapshot.nvme_queue_depth = self.snapshot.nvme_queue_depth.saturating_sub(1)
            }
            TransferKind::RamToVram => {
                self.snapshot.transfer_queue_depth =
                    self.snapshot.transfer_queue_depth.saturating_sub(1)
            }
            _ => {}
        }
        Ok(())
    }

    pub fn advance_generation(&mut self, page_id: &PageId, generation: Generation) -> Result<()> {
        let (old_generation, leases, state, bytes, had_pageable, had_pinned) = {
            let record = self.record(page_id)?;
            (
                record.generation,
                record.leases.len(),
                record.state.clone(),
                record.representation.bytes,
                record.has_replica(&PageLocation::RamPageable),
                record.has_replica(&PageLocation::RamPinned),
            )
        };
        if leases != 0 || state == ResidencyState::InUse {
            return Err(ResidencyError::ResourceExhausted(
                "cannot advance a page while it is leased".into(),
            ));
        }
        if generation <= old_generation {
            return Err(ResidencyError::Invalid(
                "generation must increase monotonically".into(),
            ));
        }
        let stale_slots: Vec<u32> = self
            .slots
            .slots()
            .iter()
            .filter(|slot| {
                slot.page_id.as_ref() == Some(page_id) && slot.generation == Some(old_generation)
            })
            .map(|slot| slot.slot_id)
            .collect();
        for slot_id in stale_slots {
            if self.slots.get(slot_id)?.evictable() {
                self.slots.evict(slot_id, page_id, old_generation)?;
            } else {
                return Err(ResidencyError::ResourceExhausted(
                    "cannot advance a page with a busy slot".into(),
                ));
            }
        }
        if had_pageable {
            self.snapshot.ram_pageable_used = self.snapshot.ram_pageable_used.saturating_sub(bytes);
        }
        if had_pinned {
            self.snapshot.ram_pinned_used = self.snapshot.ram_pinned_used.saturating_sub(bytes);
        }
        let record = self.record_mut(page_id)?;
        record.replicas.clear();
        record.generation = generation;
        record.state = ResidencyState::Indexed;
        record
            .transition_history
            .push(format!("generation advanced to {:?}", generation));
        Ok(())
    }

    /// Feed one explicit completed residency observation into REMORA. This
    /// method only reads the manager snapshot; ownership/state transitions
    /// remain governed by this manager.
    pub fn observe_remora(
        &self,
        bridge: &mut crate::remora::ResidencyMetabolism,
        observation: crate::remora::ResidencyObservation,
    ) -> har_metabolism::snapshot::MetabolismSnapshot {
        bridge.observe(&self.snapshot, observation)
    }

    pub fn validate(&self) -> Result<()> {
        for record in self.records.values() {
            record.validate()?;
        }
        for slot in self
            .slots
            .slots()
            .iter()
            .filter(|slot| slot.state != SlotState::Free)
        {
            let page_id = slot.page_id.as_ref().ok_or_else(|| {
                ResidencyError::Invalid("occupied slot has no page identity".into())
            })?;
            let record = self.record(page_id)?;
            if slot.generation != Some(record.generation) {
                return Err(ResidencyError::StaleGeneration {
                    page_id: page_id.clone(),
                    expected: record.generation,
                    got: slot.generation.unwrap_or(Generation(0)),
                });
            }
            if slot.ready() && !record.has_replica(&PageLocation::VramSlot(slot.slot_id)) {
                return Err(ResidencyError::Invalid(
                    "ready slot lacks explicit record replica".into(),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slice(kind: PageKind, ordinal: u64, bytes: u64) -> StorageSlice {
        StorageSlice {
            page_id: PageId {
                model_root: ModelRoot::new("fixture"),
                kind,
                ordinal,
            },
            model_id: "fixture".into(),
            source_path: "fixture.gguf".into(),
            shard: "0".into(),
            tensor: "blk.0.ffn_gate_exps.weight".into(),
            offset: 0,
            payload_bytes: bytes,
            alignment: 4096,
            quant_type: "Q8_0".into(),
            layer: Some(0),
            expert: Some(0),
            projection: ExpertProjection::Gate,
            checksum_sha256: None,
            parent_offset: None,
            parent_payload_bytes: None,
        }
    }

    #[test]
    fn state_machine_rejects_skipping_physical_stage() {
        assert!(!legal_transition(
            &ResidencyState::Indexed,
            &ResidencyState::ComputeReady
        ));
        assert!(legal_transition(
            &ResidencyState::NvmeReading,
            &ResidencyState::RamReady
        ));
        assert!(!legal_transition(
            &ResidencyState::InUse,
            &ResidencyState::Indexed
        ));
    }

    #[test]
    fn generation_advance_invalidates_old_completion() {
        let page = slice(PageKind::Weights, 7, 128);
        let page_id = page.page_id.clone();
        let mut manager = ResidencyManager::new(1024, 1024, 1, 256, 1, 1, 1).unwrap();
        manager.register(&page, Generation(0)).unwrap();
        let ticket = manager.queue_nvme_read(&page_id, true, false).unwrap();
        manager.advance_generation(&page_id, Generation(1)).unwrap();
        assert!(matches!(
            manager.start_nvme_read(ticket),
            Err(ResidencyError::StaleGeneration { .. })
        ));
    }

    #[test]
    fn kv_and_weight_pages_cannot_alias() {
        let weights = PageId::weights("same-model", 9);
        let kv = PageId::kv("same-model", 9);
        assert_ne!(weights, kv);
        assert_ne!(weights.kind, kv.kind);
    }

    #[test]
    fn stale_slot_generation_is_rejected() {
        let page = slice(PageKind::Weights, 1, 64);
        let mut manager = ResidencyManager::new(1024, 1024, 1, 128, 1, 1, 1).unwrap();
        manager.register(&page, Generation(0)).unwrap();
        let ticket = manager.queue_nvme_read(&page.page_id, true, false).unwrap();
        manager.start_nvme_read(ticket).unwrap();
        manager.complete_nvme_read(ticket, None).unwrap();
        let slot = manager
            .reserve_vram(&page.page_id, ExpertProjection::Gate, true, false)
            .unwrap();
        let upload = manager
            .queue_upload(&page.page_id, slot, true, false)
            .unwrap();
        manager.start_upload(upload).unwrap();
        manager.complete_upload(upload, slot, None).unwrap();
        manager.mark_compute_ready(&page.page_id, slot).unwrap();
        let lease = manager.acquire_compute(&page.page_id, slot).unwrap();
        manager.release_compute(lease).unwrap();
        manager
            .advance_generation(&page.page_id, Generation(1))
            .unwrap();
        assert!(manager
            .slots
            .acquire(slot, &page.page_id, Generation(1), Epoch(2))
            .is_err());
    }
}
