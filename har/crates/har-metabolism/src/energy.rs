//! Energy labels (REMORA-08).
//!
//! Energy stays `UNKNOWN` unless a `GPU_ONLY` source is present.  The V1
//! runtime has no on-device energy counter, so every energy field defaults
//! to `Unknown` and only the `scope` enum carries intent.

use serde::{Deserialize, Serialize};

/// The energy accounting scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnergyScope {
    /// No physical energy source; every field is UNKNOWN.
    Unknown,
    /// GPU-only source present (e.g. RDNA4 SM residency counters) and
    /// measured fields are valid.
    GpuOnly,
    /// Power-integrated estimate from hwmon / rocm-smi.  Fields are valid
    /// but annotated as estimated rather than a true hardware energy counter.
    Estimated,
}

/// The energy label.  `joules_per_exact_token` is only meaningful under
/// `EnergyScope::GpuOnly`; under `Unknown` it is semantically forbidden to
/// fill it (always None).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnergyLabel {
    pub scope: EnergyScope,
    pub joules_per_exact_token: Option<f64>,
    pub joules_per_accepted_token: Option<f64>,
    pub energy_delay_product: Option<f64>,
}

impl Default for EnergyLabel {
    fn default() -> Self {
        Self {
            scope: EnergyScope::Unknown,
            joules_per_exact_token: None,
            joules_per_accepted_token: None,
            energy_delay_product: None,
        }
    }
}

impl EnergyLabel {
    pub fn unknown() -> Self {
        Self::default()
    }

    /// Record a measured GPU-only source.  Called ONLY by code that actually
    /// read a device energy counter; every other path must stay Unknown.
    pub fn gpu_only(joules_per_exact_token: f64) -> Self {
        Self {
            scope: EnergyScope::GpuOnly,
            joules_per_exact_token: Some(joules_per_exact_token),
            joules_per_accepted_token: None,
            energy_delay_product: None,
        }
    }

    pub fn is_known(&self) -> bool {
        self.scope == EnergyScope::GpuOnly && self.joules_per_exact_token.is_some()
    }
}
