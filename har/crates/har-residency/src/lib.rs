//! HAR physical page store and MoE wavefront control plane.
//!
//! This crate contains no operation-text interpreter. Compiled page plans
//! enter through `StorageSlice`; execution adapters consume readiness events.

pub mod compat;
pub mod expert_lru;
pub mod manager;
pub mod mtp;
pub mod page_store;
pub mod remora;
pub mod scheduler;
pub mod types;

pub use compat::{check_model_generation, page_location_from_model, residency_state_to_model};
pub use expert_lru::{LruCacheStats, LruExpertCache};
pub use manager::{ResidencyManager, SlotState, VramSlot, VramSlotTable};
pub use mtp::{ExpertUnionEstimate, MtpResourceInterface, PageResidencyView, ResidencySnapshot};
pub use page_store::{InMemoryPageStore, PageStore};
pub use remora::{ResidencyMetabolism, ResidencyObservation};
pub use scheduler::{
    PageSource, ProjectionReplay, SlackPolicy, WavefrontEvent, WavefrontScheduler, WavefrontWork,
    WorkStage,
};
pub use types::{
    legal_transition, Epoch, EvictionReason, ExpertProjection, Generation, ModelRoot, PageId,
    PageKind, PageLease, PageLocation, Replica, RepresentationId, ResidencyError, ResidencyRecord,
    ResidencyState, ResourceSnapshot, Result, StorageSlice, TransferKind, TransferStatus,
    TransferTicket,
};
