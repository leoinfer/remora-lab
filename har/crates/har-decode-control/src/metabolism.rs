//! Decode-side REMORA admission and observation boundary.
//!
//! Optional work can be deferred here, but this gate never authorizes
//! speculative output or changes exact token acceptance. Exact observations
//! are recorded only after the decode owner has committed them.

use har_metabolism::artifact::ReuseClass;
use har_metabolism::clock::{ClockBasis, FastObservation};
use har_metabolism::controller::{ControllerConfig, MetabolismController};
use har_metabolism::ledger::LedgerClass;
use har_metabolism::portion::{PortionDecision, PortionInput};
use har_metabolism::snapshot::MetabolismSnapshot;
use har_metabolism::{MetabolismError, MetabolismResult};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecodeMetabolismConfig {
    pub enabled: bool,
    pub optional_work_requires_known_cost: bool,
}

impl Default for DecodeMetabolismConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            optional_work_requires_known_cost: true,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DecodeObservation {
    pub token_identity: Option<String>,
    pub work_mib_ms: u64,
    pub generation: u64,
    pub transfer_cost_ns: u64,
    pub compute_cost_ms: u64,
    pub exact_commit: bool,
    pub miss: bool,
}

#[derive(Clone, Debug)]
pub struct DecodeMetabolismGate {
    pub controller: MetabolismController,
    pub config: DecodeMetabolismConfig,
}

impl DecodeMetabolismGate {
    pub fn new(
        config: DecodeMetabolismConfig,
        controller_config: ControllerConfig,
        basis: ClockBasis,
    ) -> Self {
        let mut controller = MetabolismController::new(controller_config);
        controller.set_basis(basis);
        Self { controller, config }
    }

    pub fn with_defaults(model_root: impl Into<String>, graph_identity: impl Into<String>) -> Self {
        Self::new(
            DecodeMetabolismConfig::default(),
            ControllerConfig::default(),
            ClockBasis {
                model_root: model_root.into(),
                graph_identity: graph_identity.into(),
                worker_set: "har-decode-control".into(),
            },
        )
    }

    /// Decide only whether optional work may be attempted. A deferred or
    /// rejected decision must not be interpreted as an exactness fallback.
    pub fn decide_optional(
        &mut self,
        expected_tokens: u64,
        transfer_cost_bytes: u64,
        compute_cost_ms: u64,
        class: ReuseClass,
    ) -> MetabolismResult<PortionDecision> {
        if !self.config.enabled {
            return Ok(PortionDecision::Deferred {
                reason: "REMORA disabled",
            });
        }
        if self.config.optional_work_requires_known_cost
            && (transfer_cost_bytes == 0 || compute_cost_ms == 0)
        {
            return Ok(PortionDecision::FailClosed("optional cost is unknown"));
        }
        self.controller.decide_portion(
            PortionInput {
                artifact_id: "har.decode.optional",
                expected_tokens,
                transfer_cost_bytes,
                compute_cost_ms,
            },
            class,
        )
    }

    /// Record a committed observation after exact decode ownership has been
    /// established. `exact_commit=false` is retained as a rejected/miss clock
    /// observation; it never creates useful-token ledger credit.
    pub fn observe(
        &mut self,
        observation: DecodeObservation,
    ) -> MetabolismResult<MetabolismSnapshot> {
        if observation.exact_commit {
            let identity = observation
                .token_identity
                .clone()
                .filter(|identity| !identity.is_empty())
                .ok_or(MetabolismError::FailClosed(
                    "exact token identity is unknown",
                ))?;
            self.controller.record_spent(
                LedgerClass::AuthoritativeUseful,
                identity,
                observation.work_mib_ms,
                1,
                observation.generation,
            )?;
        }
        self.controller.observe(
            FastObservation {
                transfer_cost_ns: observation.transfer_cost_ns,
                compute_cost_ms: observation.compute_cost_ms,
                was_useful: observation.exact_commit,
                miss: observation.miss,
            },
            None,
        );
        Ok(self.controller.snapshot())
    }

    pub fn snapshot(&self) -> MetabolismSnapshot {
        self.controller.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_optional_cost_fails_closed() {
        let mut gate = DecodeMetabolismGate::with_defaults("model", "graph");
        let result = gate
            .decide_optional(1, 0, 4, ReuseClass::ConditionalReusable)
            .unwrap();
        assert_eq!(
            result,
            PortionDecision::FailClosed("optional cost is unknown")
        );
    }

    #[test]
    fn exact_observation_advances_once() {
        let mut gate = DecodeMetabolismGate::with_defaults("model", "graph");
        let snapshot = gate
            .observe(DecodeObservation {
                token_identity: Some("token@0".into()),
                work_mib_ms: 1,
                generation: 1,
                exact_commit: true,
                transfer_cost_ns: 5,
                compute_cost_ms: 2,
                miss: false,
            })
            .unwrap();
        assert_eq!(snapshot.fast_epoch, 1);
        assert_eq!(snapshot.exact_tokens, 1);
    }
}
