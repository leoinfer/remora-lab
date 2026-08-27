//! Reclaim (REMORA-12): classify *spent* authoritative work for reuse.
//!
//! Reclaim never invents credit.  It consumes a *spent* ledger entry and the
//! current dependency state, and produces a reuse class.  Reuse class is
//! derived from the actual dependency state at reclaim time, never from
//! "this was computed already".

use crate::artifact::ArtifactEnvelope;
use crate::error::{MetabolismError, MetabolismResult};
use crate::ledger::{LedgerClass, WasteLedger};
use serde::{Deserialize, Serialize};

/// A reclaim decision: what may be salvaged out of spent work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReclaimDecision {
    /// Prior entry confirmed reusable under the current dependency closure.
    ExactReusable,
    /// Reusable under a validity rule.
    ConditionalReusable,
    /// Informational only (a legibility trace; never an exact authority).
    InformationalOnly,
    /// Not reusable under the current dependencies.
    Unrecoverable,
}

/// Reclaim operates on the ledger's *spent* entries.  It is a helper that
/// answers a deterministic question: given an already-recorded spent ledger
/// entry and the current artifact dependency state, can this be presented
/// as reusable?
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Reclaim {}

impl Reclaim {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn classify(
        &self,
        ledger: &WasteLedger,
        entry_seq: u64,
        current: &ArtifactEnvelope,
    ) -> MetabolismResult<ReclaimDecision> {
        let entry = ledger
            .entries
            .iter()
            .find(|e| e.seq == entry_seq)
            .ok_or_else(|| {
                MetabolismError::Invariant(format!("sequence {entry_seq} not recorded"))
            })?;

        // An authoritative spent entry can be recycled only if it was exact
        // work to begin with.  Speculation misses are never reusable.
        match entry.class {
            LedgerClass::SpeculationMiss => return Ok(ReclaimDecision::Unrecoverable),
            LedgerClass::AuthoritativeDiscarded => return Ok(ReclaimDecision::Unrecoverable),
            LedgerClass::AuthoritativeUseful => { /* continue */ }
        }

        // Reuse under present dependencies requires the artifact envelope to
        // be valid in the current context (this is checked by Refrigerator
        // elsewhere), but Reclaim alone must not over-credit: it can only
        // claim reuse when the current context says so.
        match current.correctness_class {
            crate::artifact::ReuseClass::ExactReusable => Ok(ReclaimDecision::ExactReusable),
            crate::artifact::ReuseClass::ConditionalReusable => {
                Ok(ReclaimDecision::ConditionalReusable)
            }
            crate::artifact::ReuseClass::InformationalOnly => {
                Ok(ReclaimDecision::InformationalOnly)
            }
            crate::artifact::ReuseClass::Unrecoverable => Ok(ReclaimDecision::Unrecoverable),
        }
    }
}
