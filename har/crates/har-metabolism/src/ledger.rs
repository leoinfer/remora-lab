//! Waste Ledger (REMORA-05): the single authoritative book of record for
//! spent resources and the conservation identity for the whole controller.

use crate::common::Tokens;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Class of recorded work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LedgerClass {
    /// Authoritative useful compute that produced exact output tokens.
    AuthoritativeUseful,
    /// Authoritative but discarded work (e.g. expert projection never hit).
    AuthoritativeDiscarded,
    /// Speculative work that was consumed by nothing.
    SpeculationMiss,
}

/// A single append-only ledger entry.  Never amended.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub seq: u64,
    pub gen: u64,
    pub class: LedgerClass,
    pub token_identity: String,
    pub work_mib_ms: u64,
    pub tokens: Tokens,
}

/// Aggregated ledger totals from *observed* events only.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerTotals {
    pub spent_work_mib_ms: u64,
    pub useful_tokens: Tokens,
    pub wasted_tokens: Tokens,
    pub speculation_misses: u64,
    /// Waste from speculative compute (ms).
    pub waste_spec_ms: u64,
    /// Waste from prefetch that went unused (MiB).
    pub wasted_mib: u64,
    /// ms of credit granted for observed reuse.
    pub reuse_credit_ms: u64,
    /// ms of credit granted for observed overlap.
    pub overlap_credit_ms: u64,
}

/// Evidence required to grant one unit of reuse credit.  Critical inputs are
/// not allowed to be unknown: a witness that is all zeros is a FAIL-CLOSED
/// path, it does not "pass because the count is small".
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreditEvidence {
    /// Identity of the artifact whose reuse was observed (must be from an
    /// actual ledger entry, not an assumed "it overlapped").
    pub witness_identity: String,
    /// Redeemed exact tokens, only if a prior reclaim was recorded.
    pub redeems_tokens: Tokens,
    pub bytes_moved: u64,
    pub replay_gen: u64,
}

impl CreditEvidence {
    pub fn is_known(&self) -> bool {
        !self.witness_identity.is_empty()
    }
}

/// The waste ledger: append-only.  Never guesses.  Credit is granted only
/// when a prior reclaim was recorded for the same identity, and reuse was
/// actually observed.  Stale duplicates are refused (no double credit).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WasteLedger {
    pub entries: Vec<LedgerEntry>,
    pub totals: LedgerTotals,
    /// Identities already recorded as reclaimed (witness -> base entry).
    pub reclaimed: HashMap<String, u64>,
    /// Identities that already received reuse credit.
    pub reuse_credit_given: HashMap<String, Tokens>,
    /// Identities that already received overlap credit.
    pub overlap_credit_given: HashMap<String, u64>,
    next_seq: u64,
}

type LResult<T> = std::result::Result<T, crate::error::MetabolismError>;

impl WasteLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a ledger entry.  `class` must be one of the observed classes;
    /// speculation is recorded separately from exact work.
    pub fn record_spent(
        &mut self,
        class: LedgerClass,
        token_identity: String,
        work_mib_ms: u64,
        tokens: Tokens,
        gen: u64,
    ) -> LResult<u64> {
        let seq = self.next_seq;
        self.next_seq = self.next_seq.saturating_add(1);
        let entry = LedgerEntry {
            seq,
            gen,
            class,
            token_identity,
            work_mib_ms,
            tokens,
        };
        match class {
            LedgerClass::AuthoritativeUseful => {
                self.totals.useful_tokens = self.totals.useful_tokens.saturating_add(tokens);
            }
            LedgerClass::AuthoritativeDiscarded => {
                self.totals.wasted_tokens = self.totals.wasted_tokens.saturating_add(tokens);
            }
            LedgerClass::SpeculationMiss => {
                self.totals.speculation_misses = self.totals.speculation_misses.saturating_add(1);
                self.totals.waste_spec_ms = self.totals.waste_spec_ms.saturating_add(tokens);
                self.totals.wasted_mib = self.totals.wasted_mib.saturating_add(work_mib_ms);
            }
        }
        self.totals.spent_work_mib_ms = self.totals.spent_work_mib_ms.saturating_add(work_mib_ms);
        self.entries.push(entry);
        Ok(seq)
    }

    /// `record_reclaim` is called when Refrigerator classifies an entry as
    /// reclaimable, before Salvage admits it.  It never grants credit.
    pub fn record_reclaim(&mut self, identity: &str, base_seq: u64) -> LResult<bool> {
        if identity.is_empty() {
            return Err(crate::error::MetabolismError::InvalidArtifact(
                "empty reclaim identity".into(),
            ));
        }
        if self.reclaimed.contains_key(identity) {
            return Err(crate::error::MetabolismError::DoubleCredit(format!(
                "reclaim for {identity} already recorded"
            )));
        }
        self.reclaimed.insert(identity.to_string(), base_seq);
        Ok(true)
    }

    /// Record a reuse credit.  Requires prior reclaim AND the identity must
    /// not already have crediting.  A stale duplicate is refused.
    pub fn grant_reuse_credit(&mut self, evidence: &CreditEvidence) -> LResult<bool> {
        if !evidence.is_known() {
            return Err(crate::error::MetabolismError::FailClosed(
                "unknown credit evidence",
            ));
        }
        if !self.reclaimed.contains_key(&evidence.witness_identity) {
            return Err(crate::error::MetabolismError::InvalidArtifact(
                "reuse credit requires a prior reclaim for the same identity".into(),
            ));
        }
        if self
            .reuse_credit_given
            .contains_key(&evidence.witness_identity)
            || self
                .overlap_credit_given
                .contains_key(&evidence.witness_identity)
        {
            return Err(crate::error::MetabolismError::DoubleCredit(format!(
                "identity {} already credited",
                evidence.witness_identity
            )));
        }
        self.reuse_credit_given
            .insert(evidence.witness_identity.clone(), evidence.redeems_tokens);
        self.totals.reuse_credit_ms = self
            .totals
            .reuse_credit_ms
            .saturating_add(evidence.redeems_tokens);
        Ok(true)
    }

    /// Record an overlap credit (RE: same-tick replay of two pieces of work).
    /// Requires actual measured overlap bytes (seconds-ticked) -- never assumed.
    pub fn grant_overlap_credit(
        &mut self,
        witness_identity: String,
        bytes_moved: u64,
    ) -> LResult<bool> {
        if bytes_moved == 0 {
            return Err(crate::error::MetabolismError::FailClosed(
                "overlap credit without measured bytes",
            ));
        }
        if self.reuse_credit_given.contains_key(&witness_identity)
            || self.overlap_credit_given.contains_key(&witness_identity)
        {
            return Err(crate::error::MetabolismError::DoubleCredit(format!(
                "credit identity {witness_identity} already credited"
            )));
        }
        self.overlap_credit_given
            .insert(witness_identity, bytes_moved);
        self.totals.overlap_credit_ms = self.totals.overlap_credit_ms.saturating_add(bytes_moved);
        Ok(true)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}
