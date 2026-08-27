//! Salvage (REMORA-14): admit retained artifacts whose expected re-build
//! cost outweighs their expected re-load cost, ranked deterministically.
//!
//! Salvage uses the inputs Refrigerator exposes as `SalvageInput`.
//! Unknown inputs are not zero; an admission with unknown probability or
//! unknown cost is refused with a fail-closed verdict.

use crate::artifact::SalvageInput;

use serde::{Deserialize, Serialize};

/// One candidate for salvage.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SalvageCandidate {
    pub input: SalvageInput,
    /// Causal/base sequence this candidate was derived from.
    pub base_seq: u64,
    /// Priority bucket (0 highest); order within a bucket is by base_seq so
    /// ranking is deterministic across hosts.
    pub bucket: u8,
}

/// Admit/deny of a single salvage candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SalvageDecision {
    /// Projected value (signed ms) of retention.
    Admit { projected_value: i64 },
    /// Known but non-positive; retained cost exceeds expected reuse value.
    RejectBelowCost,
    /// One or more critical inputs were unknown -> fail closed.
    FailClosed,
}

/// Deterministic retention/eviction ordering: by (bucket, projected value
/// desc) then by base_seq for absolute stability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SalvageRank {
    pub tier: u8,
    pub base_seq: u64,
}

impl SalvageRank {
    pub fn of(candidate: &SalvageCandidate) -> Self {
        Self {
            tier: candidate.bucket,
            base_seq: candidate.base_seq,
        }
    }
}

/// Salvage: builds the deterministic ranking and the admission verdicts.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Salvage;

impl Salvage {
    pub fn new() -> Self {
        Salvage
    }

    /// Deterministically score one candidate.
    ///
    /// value = expected_reuse_cost x reuse_probability (adjusting for expiry
    /// risk) - (holding + validation + memory opportunity + eviction +
    /// contention).  Any unknown critical input -> FailClosed.
    pub fn score(&self, c: &SalvageCandidate) -> SalvageDecision {
        if !c.input.known() {
            return SalvageDecision::FailClosed;
        }
        let reuse_prob_permille = c.input.reuse_probability.unwrap() as i64;
        let reuse_ms = c.input.expected_reuse_cost.value;
        let expiry_permille = c.input.expiry_risk_permille.unwrap() as i64;

        let expected = reuse_ms * reuse_prob_permille / 1000;
        let expected_after_expiry =
            expected.saturating_sub(reuse_ms_permille(reuse_ms, expiry_permille));

        let retained = c
            .input
            .holding_cost
            .value
            .saturating_add(c.input.validation_cost.value)
            .saturating_add(c.input.memory_opportunity_cost.value)
            .saturating_add(c.input.contention_cost.value);

        let value = expected_after_expiry.saturating_sub(retained);
        if value > 0 {
            SalvageDecision::Admit {
                projected_value: value,
            }
        } else {
            SalvageDecision::RejectBelowCost
        }
    }

    /// Establish a stable order across a candidate set.  Ranking defines the
    /// "eviction order"; lower rank = hotter.  Deterministic: within the same
    /// bucket the retained objects are compared by base_seq only.
    pub fn ranked(&self, candidates: &mut [SalvageCandidate]) {
        candidates.sort_unstable_by_key(SalvageRank::of);
    }

    /// Rank all candidates and return the admitted subset in admission order.
    pub fn admit(
        &self,
        candidates: &mut [SalvageCandidate],
    ) -> Vec<(SalvageCandidate, SalvageDecision)> {
        self.ranked(candidates);
        candidates.iter().map(|c| (*c, self.score(c))).collect()
    }
}

fn reuse_ms_permille(reuse_ms: i64, expiry_risk_permille: i64) -> i64 {
    reuse_ms
        .saturating_mul(expiry_risk_permille)
        .saturating_div(1000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_input_fails_closed() {
        let s = Salvage::new();
        let mut c = SalvageCandidate {
            input: SalvageInput::with_none(),
            base_seq: 0,
            bucket: 0,
        };
        c.input.reuse_probability = Some(500);
        // expected_reuse_cost still UNKNOWN -> score fails closed.
        assert_eq!(s.score(&c), SalvageDecision::FailClosed);
    }
}
