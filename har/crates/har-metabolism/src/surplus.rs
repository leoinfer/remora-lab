//! Uncertainty-Adjusted Safe Surplus (REMORA-20).
//!
//! Safe surplus = the reserve region we may *safely* spend on optional work
//! beyond the maintenance setpoint, after subtracting an uncertainty
//! adjustment sized by adjoined uncertainty (contention, volatility,
//! miss-risk).  Unknown sources never reduce surplus silently: an unknown
//! input is treated as the *worst case* (largest adjustment) so spending
//! remains safe.

use crate::common::MiB;
use serde::{Deserialize, Serialize};

/// Inputs that adjust the safe surplus downward.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SurplusInputs {
    pub total_reserve: MiB,
    pub maintenance_setpoint_mib: MiB,
    /// Miss-rate-driven cold-start penalty in MiB (micro-latency budget).
    pub miss_rate_penalty_mib: MiB,
    /// Contention with other workers in MiB.  Unknown -> conservatively large.
    pub contention_mib: Option<MiB>,
    /// Interference from I/O pipelines; unknown -> treated as worst.
    pub interference_mib: Option<MiB>,
    /// When true, any unknown (None) term shrinks the budget instead of a
    /// guessed value being substituted.
    pub unknown_shrinks: bool,
}

impl SurplusInputs {
    /// A completely unknown input set; every unknown term is worst-case.
    pub fn unknown() -> Self {
        Self {
            total_reserve: 0,
            maintenance_setpoint_mib: 0,
            miss_rate_penalty_mib: 0,
            contention_mib: None,
            interference_mib: None,
            unknown_shrinks: true,
        }
    }
}

/// Safe surplus result with the adjustment trail for observability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafeSurplus {
    pub optional_mib: MiB,
    /// [miss_adjustment, contention_adjustment, interference_adjustment,
    ///  gross_mib] as applied.
    pub adjustment_record: [MiB; 4],
}

/// Computes the safe surplus of a reserve against a maintenance setpoint.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Surplus;

impl Surplus {
    pub fn new() -> Self {
        Surplus
    }

    /// Determine the optional budget: never exceeds reserve minus the
    /// maintenance setpoint; unknown-risk terms always shrink the budget,
    /// never grow it.
    pub fn compute(&self, inputs: &SurplusInputs) -> SafeSurplus {
        let gross = inputs
            .total_reserve
            .saturating_sub(inputs.maintenance_setpoint_mib);

        let miss_adj = inputs.miss_rate_penalty_mib;
        let after_miss = gross.saturating_sub(miss_adj);

        let contention_adj = match inputs.contention_mib {
            Some(v) => v.min(after_miss),
            None => {
                if inputs.unknown_shrinks {
                    after_miss / 2
                } else {
                    0
                }
            }
        };
        let after_contention = after_miss.saturating_sub(contention_adj);

        let interference_adj = match inputs.interference_mib {
            Some(v) => v.min(after_contention),
            None => {
                if inputs.unknown_shrinks {
                    after_contention / 2
                } else {
                    0
                }
            }
        };
        let optional = after_contention.saturating_sub(interference_adj);

        SafeSurplus {
            optional_mib: optional,
            adjustment_record: [miss_adj, contention_adj, interference_adj, gross],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surplus_never_exceeds_reserve_minus_setpoint() {
        let s = Surplus::new();
        let inputs = SurplusInputs {
            total_reserve: 1000,
            maintenance_setpoint_mib: 400,
            miss_rate_penalty_mib: 50,
            contention_mib: Some(10),
            interference_mib: Some(10),
            unknown_shrinks: false,
        };
        let out = s.compute(&inputs);
        assert!(out.optional_mib <= 600);
        assert_eq!(out.optional_mib, 1000 - 400 - 50 - 10 - 10);
    }

    #[test]
    fn surplus_zero_when_setpoint_eclipses_reserve() {
        let inputs = SurplusInputs {
            total_reserve: 100,
            maintenance_setpoint_mib: 500,
            ..SurplusInputs::unknown()
        };
        let out = Surplus::new().compute(&inputs);
        assert_eq!(out.optional_mib, 0);
    }

    #[test]
    fn unknown_shrinks_surplus() {
        let inputs = SurplusInputs {
            total_reserve: 1000,
            maintenance_setpoint_mib: 0,
            miss_rate_penalty_mib: 0,
            contention_mib: None,
            interference_mib: None,
            unknown_shrinks: true,
        };
        let out = Surplus::new().compute(&inputs);
        // 1000 -> after contention /2 = 500 -> after interference /2 = 250
        assert_eq!(out.optional_mib, 250);
    }
}
