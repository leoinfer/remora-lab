//! Baseline policies (REMORA-02): what an unmanaged pipeline would do.
//!
//! Baselines are used only to tell the difference the controller makes --
//! they are never executed by the controller.  Each baseline defines its own
//! (policy-only, deterministic) response to the same input observations.
//! V1 implements the required baselines:
//!   NoOptimization, LruOnly, FixedPrefetch, FixedReserve, RemoraV1.
//! `RemoraV1` is itself *not* allowed to move toward the REMORA controller
//! (it is the shipped baseline that the REMORA controller beats).

use crate::common::MiB;
use serde::{Deserialize, Serialize};

/// Which baseline policy is being compared.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BaselinePolicy {
    NoOptimization,
    LruOnly,
    FixedPrefetch,
    FixedReserve,
    RemoraV1,
}

/// One observation fed to a baseline policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaselineObservation {
    pub vram_free_mib: MiB,
    pub pending_work_bytes: u64,
    pub was_cache_hit: bool,
}

/// The policy's deterministic response.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BaselineAction {
    Admit { bytes: u64 },
    AdmitNone,
    Evict { bytes: u64 },
}

/// Symmetrical comparison of a baseline vs the controller on the same trace.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaselineComparison {
    pub policy: BaselinePolicy,
    pub controller_score: u64,
    pub baseline_score: u64,
}

/// Baseline engine: computes what the baseline policy would have done.
#[derive(Clone, Copy, Debug, Default)]
pub struct Baseline;

impl Baseline {
    pub fn new() -> Self {
        Baseline
    }

    /// Deterministic response of a baseline policy to a single observation.
    pub fn step(
        &self,
        policy: BaselinePolicy,
        obs: BaselineObservation,
        window_bytes: u64,
    ) -> BaselineAction {
        match policy {
            BaselinePolicy::NoOptimization => BaselineAction::AdmitNone,
            BaselinePolicy::LruOnly => {
                if obs.was_cache_hit {
                    BaselineAction::AdmitNone
                } else {
                    BaselineAction::Evict {
                        bytes: obs.pending_work_bytes,
                    }
                }
            }
            BaselinePolicy::FixedPrefetch => {
                if obs.was_cache_hit {
                    BaselineAction::AdmitNone
                } else {
                    BaselineAction::Admit {
                        bytes: obs.pending_work_bytes,
                    }
                }
            }
            BaselinePolicy::FixedReserve => {
                // A fixed reservation admits up to the window regardless of
                // hit outcome (the "always over-provision" baseline).
                BaselineAction::Admit {
                    bytes: window_bytes,
                }
            }
            BaselinePolicy::RemoraV1 => {
                // The shipped V1: admit only on a miss, with no surplus -
                // this is what the controller is compared against.
                if obs.was_cache_hit {
                    BaselineAction::AdmitNone
                } else {
                    BaselineAction::Admit {
                        bytes: obs.pending_work_bytes,
                    }
                }
            }
        }
    }
}
