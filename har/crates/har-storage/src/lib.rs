//! HAR-owned source stores: original split GGUF first, aligned sidecar later.

pub mod direct_io;
pub mod gguf;
pub mod sidecar;

pub use direct_io::{
    DirectIoCapabilities, DirectIoEngine, IoAccounting, ReadHandle, ReadRequest, ReadResult,
};
pub use gguf::{ExpertIndex, OriginalGgufStore};
pub use sidecar::AlignedSidecarStore;
