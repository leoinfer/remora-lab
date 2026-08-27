//! Portion (REMORA-01): the per-token budget decision.
//!
//! Portion answers "should this optional increment of computation be
//! admitted?".  It is a bounded, fail-closed controller decision: an unknown
//! critical input rejects the admission rather than assuming a small value.
//! The decision never bypasses a HAR exactness gate; it only controls how
//! much optional (speculative/prefetch) budget is spent.

use crate::artifact::ReuseClass;
use crate::common::Tokens;
use crate::error::{MetabolismError, MetabolismResult};
use crate::reserve::{ReserveDim, ReserveTable};
use serde::{Deserialize, Serialize};

/// A single portion input: all quantities are *before* resident environment
/// work (maintenance) is discounted.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PortionInput {
    /// Identity of the artifact being considered.
    pub artifact_id: &'static str,
    /// Expected exact tokens this portion would redundantly (but exactly)
    /// re-examine.
    pub expected_tokens: Tokens,
    /// Measured transfer cost in Mb moved (H2D/D2H/NVMe) for this increment.
    pub transfer_cost_bytes: u64,
    /// Measured (or estimated) compute cost in ms that would be spent.
    pub compute_cost_ms: u64,
}

/// The Portion decision.  `Admit` is the only gate through which optional
/// spending flows; `Deferring` means budget again soon; everything else is
/// reject/unknown.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PortionDecision {
    /// Admit, but the `expected_tokens` field remains advisory.  Actual
    /// credit is only booked by the ledger from CHANGES (never from intent).
    Admit {
        expected_credit_tokens: Tokens,
    },
    Deferred {
        reason: &'static str,
    },
    Reject {
        reason: &'static str,
    },
    FailClosed(&'static str),
}

/// Portion consumer over a reserve table and the arriving token stream.
/// This is the only way optional work is monetized in REMORA.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Portion {
    pub approved_total_tokens: u64,
    pub approved_spends: u64,
    pub rejected_spends: u64,
}

impl Portion {
    pub fn new() -> Self {
        Self::default()
    }

    /// Decide whether to admit a portion.  Deterministic.  Unknown critical
    /// inputs (e.g. an artifact with no expected-token identity) fail closed.
    pub fn decide(
        &mut self,
        input: PortionInput,
        reserve: &ReserveTable,
        class: ReuseClass,
    ) -> MetabolismResult<PortionDecision> {
        if class == ReuseClass::Unrecoverable {
            return Err(MetabolismError::FailClosed("artifact unrecoverable"));
        }
        if input.artifact_id.is_empty() {
            return Err(MetabolismError::FailClosed("missing artifact identity"));
        }
        if input.expected_tokens == 0 {
            self.rejected_spends += 1;
            return Ok(PortionDecision::Deferred {
                reason: "zero expected tokens",
            });
        }
        if !reserve.check(ReserveDim::GpuCompute, 1) {
            self.rejected_spends += 1;
            return Ok(PortionDecision::Reject {
                reason: "compute reserve saturated",
            });
        }
        // The portion increment must be affordable now.  Any admission
        // reserves compute so the optional work does not pile up.
        if !reserve.check(ReserveDim::Vram, 1) {
            return Ok(PortionDecision::Deferred {
                reason: "vram reserve saturated",
            });
        }
        self.approved_total_tokens = self
            .approved_total_tokens
            .saturating_add(input.expected_tokens);
        self.approved_spends += 1;
        Ok(PortionDecision::Admit {
            expected_credit_tokens: input.expected_tokens,
        })
    }
}
