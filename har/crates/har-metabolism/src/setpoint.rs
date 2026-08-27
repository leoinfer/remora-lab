//! Moving Maintenance Setpoint (REMORA-21).
//!
//! Computes the maintenance budget (VRAM/RAM/compute/queue) required to keep
//! the pipeline's peak context available, from runtime observables only.
//! It is a setpoint, not a policy: it never decides what gets done, it only
//! reports the *resources required for what must be kept resident*.

use crate::common::MiB;
use serde::{Deserialize, Serialize};

/// Setup for the maintenance setpoint estimator.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaintenanceParams {
    pub vram_per_kv_mib: MiB,
    pub vram_activation_floor_mib: MiB,
    pub ram_activation_floor_mib: MiB,
    pub compute_ms_per_token: u64,
    pub queue_depth_min: u32,
    pub queue_depth_max: u32,
}

impl Default for MaintenanceParams {
    fn default() -> Self {
        Self {
            vram_per_kv_mib: 2,
            vram_activation_floor_mib: 512,
            ram_activation_floor_mib: 1024,
            compute_ms_per_token: 2,
            queue_depth_min: 2,
            queue_depth_max: 8,
        }
    }
}

/// Observables used to move the setpoint.  Unknown -> use params floor.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaintenanceObservations {
    /// Number of live KV tokens (bounded by the machinery, never infinite).
    pub kv_occupancy: u64,
    /// MTP-acceptance ratio in permille over the last slow window.
    pub mtp_acceptance_permille: Option<u64>,
    /// Queue depth (ratio of active resident workers).
    pub queue_depth: u32,
}

/// The computed maintenance budget.  All in MiB (or tokens/compute ms).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaintenanceSetpoint {
    pub vram_mib: MiB,
    pub ram_mib: MiB,
    pub compute_ms_per_token: u64,
    pub queue_depth: u32,
}

impl MaintenanceSetpoint {
    pub fn total_mib(&self) -> MiB {
        self.vram_mib.saturating_add(self.ram_mib)
    }
}

/// Moving maintenance setpoint estimator on top of fast/slow clocks.
/// Deterministic and monotone: a bad (higher) MTP acceptance never shrinks
/// the maintenance envelope below the base floor.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SetpointEstimator {
    pub params: MaintenanceParams,
}

impl SetpointEstimator {
    pub fn new(params: MaintenanceParams) -> Self {
        Self { params }
    }

    /// Compute the maintenance setpoint from current observables.
    ///
    /// vram = activation_floor + kv_occupancy * v_per_kv (scaled by MTP
    /// acceptance, which raises expectation that accept-stream inflates KV).
    /// ram  = activation_floor XOR spill when vram exceeds a cap (uses RAM as
    ///         the second-tier spill; the setpoint estimator does not claim
    ///         free RAM, it only claims the maintenance footprint).
    pub fn compute(&self, obs: &MaintenanceObservations) -> MaintenanceSetpoint {
        let qd = obs
            .queue_depth
            .clamp(self.params.queue_depth_min, self.params.queue_depth_max);
        let mtp = obs.mtp_acceptance_permille.unwrap_or(0);
        let mtp_scale = 1000_u64.saturating_add(mtp) / 1000_u64;
        let kv_vram = obs.kv_occupancy.saturating_mul(self.params.vram_per_kv_mib) * mtp_scale;
        let vram = self
            .params
            .vram_activation_floor_mib
            .saturating_add(kv_vram);
        let ram = self
            .params
            .ram_activation_floor_mib
            .saturating_add(vram / 2);
        MaintenanceSetpoint {
            vram_mib: vram,
            ram_mib: ram,
            compute_ms_per_token: qd as u64,
            queue_depth: qd,
        }
    }
}
