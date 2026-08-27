//! Explicit REMORA observation bridge for residency events.
//!
//! The bridge observes the residency manager's resource snapshot but never
//! authorizes a transfer or bypasses the residency state machine. It only
//! advances the deterministic metabolism controller and returns an additive
//! snapshot for runtime telemetry.

use crate::types::ResourceSnapshot;
use har_metabolism::clock::{ClockBasis, FastObservation};
use har_metabolism::controller::{ControllerConfig, MetabolismController};
use har_metabolism::setpoint::MaintenanceObservations;
use har_metabolism::snapshot::MetabolismSnapshot;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyObservation {
    pub transfer_cost_ns: u64,
    pub compute_cost_ms: u64,
    /// Optional live KV occupancy in tokens. Residency byte usage is not
    /// substituted here because it has different units.
    pub kv_occupancy: Option<u64>,
    pub useful: bool,
    pub miss: bool,
}

impl Default for ResidencyObservation {
    fn default() -> Self {
        Self {
            transfer_cost_ns: 0,
            compute_cost_ms: 0,
            kv_occupancy: None,
            useful: true,
            miss: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ResidencyMetabolism {
    pub controller: MetabolismController,
}

impl ResidencyMetabolism {
    pub fn new(config: ControllerConfig, basis: ClockBasis) -> Self {
        let mut controller = MetabolismController::new(config);
        controller.set_basis(basis);
        Self { controller }
    }

    pub fn with_defaults(model_root: impl Into<String>, graph_identity: impl Into<String>) -> Self {
        Self::new(
            ControllerConfig::default(),
            ClockBasis {
                model_root: model_root.into(),
                graph_identity: graph_identity.into(),
                worker_set: "har-residency".into(),
            },
        )
    }

    /// Observe one completed residency/compute unit. `resources` is copied
    /// read-only; the controller cannot mutate residency ownership.
    pub fn observe(
        &mut self,
        resources: &ResourceSnapshot,
        observation: ResidencyObservation,
    ) -> MetabolismSnapshot {
        let queue_depth = resources
            .transfer_queue_depth
            .saturating_add(resources.compute_queue_depth)
            .saturating_add(resources.nvme_queue_depth);
        self.controller.update_setpoint(&MaintenanceObservations {
            kv_occupancy: observation.kv_occupancy.unwrap_or(0),
            mtp_acceptance_permille: None,
            queue_depth,
        });
        self.controller.observe(
            FastObservation {
                transfer_cost_ns: observation.transfer_cost_ns,
                compute_cost_ms: observation.compute_cost_ms,
                was_useful: observation.useful,
                miss: observation.miss,
            },
            None,
        );
        self.controller.snapshot()
    }

    pub fn snapshot(&self) -> MetabolismSnapshot {
        self.controller.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_advances_only_the_controller_clock() {
        let mut bridge = ResidencyMetabolism::with_defaults("model", "graph");
        let snapshot = bridge.observe(
            &ResourceSnapshot::default(),
            ResidencyObservation {
                transfer_cost_ns: 12,
                compute_cost_ms: 3,
                kv_occupancy: None,
                useful: true,
                miss: false,
            },
        );
        assert_eq!(snapshot.fast_epoch, 1);
        assert_eq!(snapshot.exact_tokens, 0);
        assert_eq!(
            snapshot.energy,
            har_metabolism::energy::EnergyLabel::unknown()
        );
    }
}
