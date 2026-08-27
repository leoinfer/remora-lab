//! Adapter names for compiler's `har-memory` typed state contract.
//!
//! compiler's state machine remains the public plan vocabulary.  HAR extends
//! it with generation-tagged page/slot states where the older vocabulary has
//! no distinct pageable/slot/uploaded state.

use crate::types::{PageLocation, ResidencyError, ResidencyState, Result};

pub fn page_location_from_model(tier: &har_core::MemoryTier) -> Result<PageLocation> {
    Ok(match tier {
        har_core::MemoryTier::NvmeCold => PageLocation::NvmeCold,
        har_core::MemoryTier::RamMapped => PageLocation::RamPageable,
        har_core::MemoryTier::RamPinned => PageLocation::RamPinned,
        har_core::MemoryTier::VramResident => PageLocation::VramPage(0),
        har_core::MemoryTier::VramSlot => PageLocation::VramSlot(0),
        har_core::MemoryTier::ReconstructionScratch => PageLocation::Scratch(0),
        har_core::MemoryTier::CpuHeap => PageLocation::RamPageable,
    })
}

pub fn residency_state_to_model(state: &ResidencyState) -> Result<har_core::ResidencyState> {
    Ok(match state {
        ResidencyState::Indexed => har_core::ResidencyState::Indexed,
        ResidencyState::NvmeReadQueued => har_core::ResidencyState::ReadQueued,
        ResidencyState::NvmeReading => har_core::ResidencyState::Reading,
        ResidencyState::RamReady => har_core::ResidencyState::ReadyHost,
        ResidencyState::VramReservationQueued
        | ResidencyState::VramReserved
        | ResidencyState::UploadQueued => har_core::ResidencyState::TransferQueued,
        ResidencyState::VramUploading | ResidencyState::Uploaded => {
            har_core::ResidencyState::CopyingToVram
        }
        ResidencyState::ComputeReady => har_core::ResidencyState::ReadyVram,
        ResidencyState::InUse => har_core::ResidencyState::Computing,
        ResidencyState::EvictionQueued | ResidencyState::Evicting => {
            har_core::ResidencyState::Evicting
        }
        ResidencyState::Evicted => har_core::ResidencyState::Unavailable,
        ResidencyState::Cancelled | ResidencyState::Failed => har_core::ResidencyState::Error,
    })
}

pub fn check_model_generation(record: &har_memory::ResidencyRecord, expected: u64) -> Result<()> {
    if record.machine.generation != expected {
        return Err(ResidencyError::Invalid(format!(
            "compiler generation mismatch: expected {expected}, got {}",
            record.machine.generation
        )));
    }
    Ok(())
}
