//! Artifact lifecycle: provenance envelope (REMORA Refrigerator), reuse
//! classification (REMORA Reclaim) and salvage scoring (REMORA Salvage).
//!
//! An artifact is a retained piece of work (an expert projection, a KV page,
//! a compiled kernel, a snapshot).  Its identity is a closed dependency tuple;
//! exactness is a dependency-closure property, not a token match.

use crate::common::{Estimate, Ms};
use serde::{Deserialize, Serialize};

/// Provenance envelope for a retained artifact (REMORA-13 / PFM packet).
///
/// A changed causal leaf forces a MISS or INVALIDATED state.  Prompt/model
/// hash alone is never sufficient.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactEnvelope {
    pub artifact_id: String,
    pub payload_hash: String,
    pub causal_prefix: String,
    pub dependency_root: String,
    pub token_identity: String,
    pub model_root: String,
    pub tensor_root: String,
    pub package_root: String,
    pub runtime_version: String,
    pub graph_identity: String,
    pub backend_identity: String,
    pub kernel_identity: String,
    pub precision: String,
    pub quantization: String,
    pub device: String,
    pub creation_cost: Ms,
    pub creation_gen: u64,
    pub validation_rule: String,
    pub expiry_gen: u64,
    pub correctness_class: ReuseClass,
}

/// Reclaimed-artifact reuse classification (REMORA-12).
///
/// The classification is ALWAYS derived from current dependency state, never
/// from "this was computed already".
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ReuseClass {
    /// All causal dependencies match exactly; unconditional exact credit.
    ExactReusable,
    /// Dependencies hold only under a validity rule.
    ConditionalReusable,
    /// Informational only; never an exact output authority.
    InformationalOnly,
    /// Not recoverable under the current dependencies.
    Unrecoverable,
}

/// Artifact validity verdict produced by the Refrigerator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Validity {
    Valid,
    Miss,
    Invalidated,
}

/// Salvage scoring inputs remain individually observable (REMORA-14).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SalvageInput {
    /// Estimated probability (per mille) of reuse given retained.
    pub reuse_probability: Option<u32>,
    /// Avoided reload/recompute if reused.
    pub expected_reuse_cost: Estimate,
    pub holding_cost: Estimate,
    pub validation_cost: Estimate,
    pub memory_opportunity_cost: Estimate,
    pub eviction_cost: Estimate,
    pub contention_cost: Estimate,
    /// Higher risk reduces expected salvage by hardening the retained cost.
    pub expiry_risk_permille: Option<u32>,
}

impl SalvageInput {
    pub fn with_none() -> Self {
        Self {
            reuse_probability: None,
            expected_reuse_cost: Estimate::UNKNOWN_ZERO,
            holding_cost: Estimate::UNKNOWN_ZERO,
            validation_cost: Estimate::UNKNOWN_ZERO,
            memory_opportunity_cost: Estimate::UNKNOWN_ZERO,
            eviction_cost: Estimate::UNKNOWN_ZERO,
            contention_cost: Estimate::UNKNOWN_ZERO,
            expiry_risk_permille: None,
        }
    }
    pub fn known(&self) -> bool {
        self.reuse_probability.is_some()
            && self.expected_reuse_cost.known()
            && self.holding_cost.known()
            && self.validation_cost.known()
            && self.memory_opportunity_cost.known()
            && self.eviction_cost.known()
            && self.contention_cost.known()
            && self.expiry_risk_permille.is_some()
    }
}
