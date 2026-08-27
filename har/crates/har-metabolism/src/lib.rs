//! REMORA metabolism runtime control.
//!
//! Native, bounded, deterministic implementation of the REMORA metabolism
//! family (Portion, Reserve, Reclaim, Refrigerator, Salvage, Waste Ledger,
//! Moving Maintenance Setpoint, Uncertainty-Adjusted Safe Surplus, and
//! Fast/Slow Adaptation Clocks) as a first-class control subsystem inside
//! the HAR runtime.
//!
//! Design rules inherited from the canonical definition
//! (`docs/remora_metabolism/PROVENANCE_AND_SCOPE.md`):
//!
//! - Charge only exact committed tokens / verified authority work.
//! - No double counting, no imagined overlap credit, no speculative reuse
//!   credit before actual reuse is observed.
//! - Unknown inputs are not zero; unknown critical inputs fail closed.
//! - Energy stays `UNKNOWN` unless a `GPU_ONLY` source is present.
//! - Deterministic: a recorded trace replays to identical decisions.
//! - REMORA is a controller; it never bypasses HAR exactness gates.

pub mod artifact;
pub mod baseline;
pub mod clock;
pub mod common;
pub mod controller;
pub mod energy;
pub mod energy_measurement;
pub mod error;
pub mod ledger;
pub mod portion;
pub mod reclaim;
pub mod reserve;
pub mod salvage;
pub mod setpoint;
pub mod snapshot;
pub mod surplus;
pub mod trace;

pub use controller::MetabolismController;
pub use energy_measurement::{EnergyTracker, PowerSample};
pub use error::{MetabolismError, MetabolismResult};
pub use trace::TraceMode;

pub const METABOLISM_SCHEMA: &str = "har.metabolism.v1";
pub const TRACE_SCHEMA: &str = "har.metabolism.trace.v1";
