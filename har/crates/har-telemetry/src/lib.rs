//! Telemetry is explicit and bounded.  It is not consulted for correctness
//! decisions except through a copied, generation-tagged resource snapshot.

use har_core::{canonical_sha256, sha256_bytes, sha256_f32, EpochNamespace, Result};
use har_events::{EventBuffer, RuntimeEvent};
use serde::{Deserialize, Serialize};
use std::time::Instant;

pub const TELEMETRY_SCHEMA: &str = "har.telemetry.v1";
pub const SPECULATION_TELEMETRY_SCHEMA: &str = "har.speculation_telemetry.v1";

/// Accepted-token speculation telemetry. Mirrors the HAR accepted-token
/// trace contract block section. Pure
/// instrumentation: never consulted for correctness decisions, never changes
/// output semantics. Fail-closed: `record_speculation` rejects (returns Err)
/// any record violating the trace-contract invariants:
///   accepted_tokens + rejected_tokens == candidate_horizon
///   kv_namespace_commits + kv_namespace_rollbacks <= candidate_horizon
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpeculationTelemetry {
    pub candidate_horizon: u64, // H
    pub accepted_tokens: u64,   // A
    pub rejected_tokens: u64,   // H - A
    pub verification_cost_ns: u64,
    pub draft_cost_ns: u64,
    pub kv_namespace_commits: u64,
    pub kv_namespace_rollbacks: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpeculationTotals {
    pub blocks: u64,
    pub candidates: u64,
    pub accepted_tokens: u64,
    pub rejected_tokens: u64,
    pub verification_cost_ns: u64,
    pub draft_cost_ns: u64,
    pub kv_namespace_commits: u64,
    pub kv_namespace_rollbacks: u64,
}

/// REMORA metabolism telemetry. One immutable, additive snapshot row. Every
/// field is a measured or explicitly UNKNOWN quantity; no value is invented.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct MetabolismTotals {
    pub exact_tokens: u64,
    pub maintenance_vram_mib: u64,
    pub reserve_vram_mib: u64,
    pub reserve_ram_mib: u64,
    pub safe_surplus_mib: u64,
    pub optional_budget_mib: u64,
    pub reclaimed: u64,
    pub salvaged: u64,
    pub waste_spec_compute_ms: u64,
    pub waste_prefetch_unused_mib: u64,
    pub reuse_credit_ms: u64,
    pub overlap_credit_ms: u64,
    pub reserve_debt_mib: u64,
    pub fast_epoch: u64,
    pub slow_epoch: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct OperationTelemetry {
    pub operation_id: String,
    pub input_hash: String,
    pub output_hash: String,
    pub reference_output_hash: String,
    pub input_elements: u64,
    pub output_elements: u64,
    pub bytes_moved: u64,
    pub dispatch_count: u64,
    pub residency_event_count: u64,
    pub elapsed_ns: u64,
    pub exact: bool,
    pub normalized_error_ppm: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResourceTelemetry {
    pub requested_bytes: u64,
    pub unique_bytes: u64,
    pub useful_bytes: u64,
    pub wasted_bytes: u64,
    pub nvme_reads: u64,
    pub ram_hits: u64,
    pub vram_hits: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TelemetrySnapshot {
    pub schema: String,
    pub epoch: EpochNamespace,
    pub plan_hash: String,
    pub operations: Vec<OperationTelemetry>,
    pub resources: ResourceTelemetry,
    pub events: EventBuffer,
    #[serde(default)]
    pub speculation: Vec<SpeculationTelemetry>,
    #[serde(default)]
    pub speculation_totals: SpeculationTotals,
    #[serde(default)]
    pub metabolism: MetabolismTotals,
}
impl TelemetrySnapshot {
    pub fn canonical_hash(&self) -> Result<String> {
        canonical_sha256(self)
    }
}

pub struct TelemetryCollector {
    pub epoch: EpochNamespace,
    pub plan_hash: String,
    pub events: EventBuffer,
    pub operations: Vec<OperationTelemetry>,
    pub resources: ResourceTelemetry,
    pub speculation: Vec<SpeculationTelemetry>,
    pub speculation_totals: SpeculationTotals,
    pub metabolism: MetabolismTotals,
}
impl TelemetryCollector {
    pub fn new(epoch: EpochNamespace, plan_hash: impl Into<String>, event_capacity: usize) -> Self {
        Self {
            epoch,
            plan_hash: plan_hash.into(),
            events: EventBuffer::with_capacity(event_capacity),
            operations: Vec::with_capacity(64),
            resources: ResourceTelemetry {
                requested_bytes: 0,
                unique_bytes: 0,
                useful_bytes: 0,
                wasted_bytes: 0,
                nvme_reads: 0,
                ram_hits: 0,
                vram_hits: 0,
            },
            speculation: Vec::with_capacity(16),
            speculation_totals: SpeculationTotals::default(),
            metabolism: MetabolismTotals::default(),
        }
    }
    pub fn record_event(&mut self, event: RuntimeEvent) {
        self.events.push(event);
    }
    /// Replace the additive latest REMORA row with an observed controller
    /// snapshot. The telemetry collector does not make policy decisions.
    pub fn record_metabolism(&mut self, totals: MetabolismTotals) {
        self.metabolism = totals;
    }
    pub fn record_operation(&mut self, telemetry: OperationTelemetry) {
        self.resources.requested_bytes = self
            .resources
            .requested_bytes
            .saturating_add(telemetry.bytes_moved);
        self.operations.push(telemetry);
    }
    /// Fail-closed accepted-token record. Returns Err (and records nothing)
    /// when the trace-contract invariants do not hold.
    pub fn record_speculation(
        &mut self,
        telemetry: SpeculationTelemetry,
    ) -> std::result::Result<(), String> {
        if telemetry.accepted_tokens + telemetry.rejected_tokens != telemetry.candidate_horizon {
            return Err(format!(
                "accepted+rejected {} != horizon {} (contract invariant 1)",
                telemetry.accepted_tokens + telemetry.rejected_tokens,
                telemetry.candidate_horizon
            ));
        }
        if telemetry.kv_namespace_commits + telemetry.kv_namespace_rollbacks
            > telemetry.candidate_horizon
        {
            return Err(format!(
                "commits+rollbacks {} > horizon {} (contract invariant 2)",
                telemetry.kv_namespace_commits + telemetry.kv_namespace_rollbacks,
                telemetry.candidate_horizon
            ));
        }
        self.speculation_totals.blocks += 1;
        self.speculation_totals.candidates += telemetry.candidate_horizon;
        self.speculation_totals.accepted_tokens += telemetry.accepted_tokens;
        self.speculation_totals.rejected_tokens += telemetry.rejected_tokens;
        self.speculation_totals.verification_cost_ns += telemetry.verification_cost_ns;
        self.speculation_totals.draft_cost_ns += telemetry.draft_cost_ns;
        self.speculation_totals.kv_namespace_commits += telemetry.kv_namespace_commits;
        self.speculation_totals.kv_namespace_rollbacks += telemetry.kv_namespace_rollbacks;
        self.speculation.push(telemetry);
        Ok(())
    }
    pub fn snapshot(self) -> TelemetrySnapshot {
        TelemetrySnapshot {
            schema: TELEMETRY_SCHEMA.into(),
            epoch: self.epoch,
            plan_hash: self.plan_hash,
            operations: self.operations,
            resources: self.resources,
            events: self.events,
            speculation: self.speculation,
            speculation_totals: self.speculation_totals,
            metabolism: self.metabolism,
        }
    }
}

pub fn hash_f32(values: &[f32]) -> String {
    sha256_f32(values)
}
pub fn hash_bytes(values: &[u8]) -> String {
    sha256_bytes(values)
}
pub fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use har_core::{EpochNamespace, ModelRoot};

    fn epoch() -> EpochNamespace {
        EpochNamespace {
            model_root: ModelRoot::new("m", "h"),
            graph_generation: 1,
            decode_epoch: 1,
            sequence_id: 1,
        }
    }
    fn spec(h: u64, a: u64) -> SpeculationTelemetry {
        SpeculationTelemetry {
            candidate_horizon: h,
            accepted_tokens: a,
            rejected_tokens: h - a,
            verification_cost_ns: 100,
            draft_cost_ns: 50,
            kv_namespace_commits: a,
            kv_namespace_rollbacks: h - a,
        }
    }

    #[test]
    fn f32_hash_is_stable() {
        assert_eq!(hash_f32(&[1.0, 2.0]), hash_f32(&[1.0, 2.0]));
        assert_ne!(hash_f32(&[1.0]), hash_f32(&[2.0]));
    }

    #[test]
    fn speculation_invariant_fail_closed() {
        let mut c = TelemetryCollector::new(epoch(), "p", 8);
        let bad = SpeculationTelemetry {
            candidate_horizon: 5,
            accepted_tokens: 4,
            rejected_tokens: 2,
            ..spec(5, 4)
        };
        assert!(
            c.record_speculation(bad).is_err(),
            "accepted+rejected != horizon must be rejected"
        );
        let bad2 = SpeculationTelemetry {
            candidate_horizon: 5,
            accepted_tokens: 5,
            rejected_tokens: 0,
            kv_namespace_commits: 6,
            kv_namespace_rollbacks: 0,
            ..spec(5, 5)
        };
        assert!(
            c.record_speculation(bad2).is_err(),
            "commits+rollbacks > horizon must be rejected"
        );
        assert!(
            c.speculation.is_empty(),
            "rejected records must not be stored"
        );
        assert_eq!(c.speculation_totals, SpeculationTotals::default());
    }

    #[test]
    fn speculation_totals_monotone_and_exact() {
        let mut c = TelemetryCollector::new(epoch(), "p", 8);
        c.record_speculation(spec(8, 6)).unwrap();
        c.record_speculation(spec(8, 7)).unwrap();
        c.record_speculation(spec(4, 4)).unwrap();
        let s = c.snapshot();
        assert_eq!(s.speculation_totals.blocks, 3);
        assert_eq!(s.speculation_totals.candidates, 20);
        assert_eq!(s.speculation_totals.accepted_tokens, 17);
        assert_eq!(s.speculation_totals.rejected_tokens, 3);
        assert_eq!(s.speculation.len(), 3);
        assert_eq!(s.speculation_totals.verification_cost_ns, 300);
        assert_eq!(s.speculation_totals.draft_cost_ns, 150);
    }

    #[test]
    fn snapshot_backward_compatible_deserialize() {
        // Old-format snapshot (no speculation fields) must still deserialize.
        let old = r#"{"schema":"har.telemetry.v1","epoch":{"model_root":{"name":"m","sha256":"h"},"graph_generation":1,"decode_epoch":1,"sequence_id":1},"plan_hash":"p","operations":[],"resources":{"requested_bytes":0,"unique_bytes":0,"useful_bytes":0,"wasted_bytes":0,"nvme_reads":0,"ram_hits":0,"vram_hits":0},"events":{"schema":"har.events.v1","capacity":0,"events":[],"dropped":0}}"#;
        let snap: TelemetrySnapshot =
            serde_json::from_str(old).expect("old snapshot must deserialize");
        assert!(snap.speculation.is_empty());
        assert_eq!(snap.speculation_totals, SpeculationTotals::default());
    }

    #[test]
    fn identical_collectors_produce_identical_canonical_hashes() {
        let mk = || {
            let mut c = TelemetryCollector::new(epoch(), "p", 8);
            c.record_speculation(spec(8, 6)).unwrap();
            c.snapshot()
        };
        assert_eq!(
            mk().canonical_hash().unwrap(),
            mk().canonical_hash().unwrap()
        );
    }
}
