//! INT-01 wiring: speculative block resolution at the real executor boundary.
//!
//! `ExecutionEngine::resolve_speculative_block` is the real boundary where a
//! speculative block is resolved: the engine validates the plan, walks the
//! candidate positions in block order, dispatches each candidate through the
//! real `OperationAdapter`, decides acceptance from the actual differential
//! verdict, and records exactly one speculation record through the collector.
//!
//! Contract guarantees:
//! - exact modes stay exact: acceptance only ever comes from the actual
//!   differential verdict (`ExactMatch` or `WithinReferenceTolerance`);
//! - telemetry never changes model output: the resolver never mutates the
//!   adapter inputs or outputs, and disabled telemetry skips all recording;
//! - no duplicate counting: each candidate position is counted exactly once;
//! - rejected positions never count as accepted (they only roll back);
//! - horizon invariants are checked before any recording;
//! - counters aggregate across blocks in `SpeculationTotals`;
//! - telemetry can be disabled per block with negligible behavioral impact;
//! - no panics from malformed observational data (all failures are `Err`);
//! - existing snapshots remain readable (additive serde defaults).

use crate::{DifferentialVerdict, ExecutionEngine, ExecutionResult, OperationAdapter};
use har_core::{HarError, Result};
use har_events::{EventHeader, RuntimeEvent, RuntimeEventKind};
use har_telemetry::{SpeculationTelemetry, TelemetryCollector};
use serde::{Deserialize, Serialize};

pub const SPECULATION_BOUNDARY: &str = "har.execution.speculation.v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SpeculativeCandidate {
    pub position: u32,
    pub operation_index: u32,
    pub namespace: String,
    pub input: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SpeculativeBlock {
    pub block_index: u64,
    pub candidate_horizon: u64,
    pub candidates: Vec<SpeculativeCandidate>,
    pub telemetry_enabled: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlockResolution {
    pub block_index: u64,
    pub candidate_horizon: u64,
    pub accepted_tokens: u64,
    pub rejected_tokens: u64,
    pub draft_cost_ns: u64,
    pub verification_cost_ns: u64,
    pub kv_namespace_commits: u64,
    pub kv_namespace_rollbacks: u64,
    pub accepted_namespaces: Vec<String>,
    pub rolled_back_namespaces: Vec<String>,
}

impl ExecutionEngine {
    /// Resolve one speculative block through the real executor path.
    ///
    /// Every candidate position is dispatched through `adapter` via the
    /// engine's real execution pipeline; the differential verdict decides
    /// acceptance. Exactly one speculation telemetry record and exactly one
    /// `SpeculationResolved` event are produced when telemetry is enabled.
    /// Invalid blocks (horizon mismatch, short candidates, empty or duplicate
    /// namespaces, unknown operations) fail closed with `Err` before any
    /// dispatch or recording.
    pub fn resolve_speculative_block<A: OperationAdapter>(
        &self,
        block: &SpeculativeBlock,
        adapter: &mut A,
        collector: &mut TelemetryCollector,
    ) -> Result<BlockResolution> {
        let validation = self.validate();
        if !validation.is_valid() {
            return Err(HarError::Invalid {
                kind: "execution plan",
                message: validation.errors.join("; "),
            });
        }
        if block.candidates.len() as u64 != block.candidate_horizon {
            return Err(HarError::Invalid {
                kind: "speculative block",
                message: format!(
                    "candidate_horizon {} != candidates {}",
                    block.candidate_horizon,
                    block.candidates.len()
                ),
            });
        }
        if block.candidate_horizon == 0 {
            return Err(HarError::Invalid {
                kind: "speculative block",
                message: "candidate_horizon must be >= 1".into(),
            });
        }
        let mut namespaces = std::collections::BTreeSet::new();
        for candidate in &block.candidates {
            if candidate.namespace.is_empty() {
                return Err(HarError::Invalid {
                    kind: "speculative block",
                    message: "empty candidate namespace".into(),
                });
            }
            if !namespaces.insert(candidate.namespace.clone()) {
                return Err(HarError::Invalid {
                    kind: "speculative block",
                    message: format!("duplicate namespace {}", candidate.namespace),
                });
            }
            if self.plan.operations.physical.len() as u32 <= candidate.operation_index {
                return Err(HarError::Invalid {
                    kind: "speculative block",
                    message: format!("unknown operation index {}", candidate.operation_index),
                });
            }
        }

        let mut accepted_tokens: u64 = 0;
        let mut rejected_tokens: u64 = 0;
        let mut verification_cost_ns: u64 = 0;
        let draft_cost_ns: u64 = 0;
        let mut accepted_namespaces = Vec::new();
        let mut rolled_back_namespaces = Vec::new();

        for candidate in &block.candidates {
            let result: ExecutionResult =
                self.execute(candidate.operation_index, &candidate.input, adapter)?;
            if block.telemetry_enabled {
                // Dispatch elapsed from the executor's own monotonic clock.
                verification_cost_ns = verification_cost_ns.saturating_add(result.elapsed_ns);
            }
            match result.verdict {
                DifferentialVerdict::ExactMatch | DifferentialVerdict::WithinReferenceTolerance => {
                    accepted_tokens = accepted_tokens.saturating_add(1);
                    accepted_namespaces.push(candidate.namespace.clone());
                }
                DifferentialVerdict::NumericalMismatch
                | DifferentialVerdict::StructuralMismatch => {
                    rejected_tokens = rejected_tokens.saturating_add(1);
                    rolled_back_namespaces.push(candidate.namespace.clone());
                }
                DifferentialVerdict::Unsupported => {
                    return Err(HarError::Unsupported {
                        kind: "speculative candidate",
                        message: format!(
                            "operation {} returned Unsupported",
                            candidate.operation_index
                        ),
                    });
                }
            }
        }

        if accepted_tokens + rejected_tokens != block.candidate_horizon {
            return Err(HarError::Invalid {
                kind: "speculative block",
                message: "accepted + rejected != horizon after resolution".into(),
            });
        }

        let resolution = BlockResolution {
            block_index: block.block_index,
            candidate_horizon: block.candidate_horizon,
            accepted_tokens,
            rejected_tokens,
            draft_cost_ns,
            verification_cost_ns,
            kv_namespace_commits: accepted_tokens,
            kv_namespace_rollbacks: rejected_tokens,
            accepted_namespaces,
            rolled_back_namespaces,
        };

        if block.telemetry_enabled {
            let telemetry = SpeculationTelemetry {
                candidate_horizon: block.candidate_horizon,
                accepted_tokens,
                rejected_tokens,
                draft_cost_ns,
                verification_cost_ns,
                kv_namespace_commits: accepted_tokens,
                kv_namespace_rollbacks: rejected_tokens,
            };
            collector
                .record_speculation(telemetry)
                .map_err(|message| HarError::Invalid {
                    kind: "speculation",
                    message,
                })?;
            let epoch = har_core::EpochNamespace::new(
                har_core::ModelRoot::new(
                    self.plan.model_identity.clone(),
                    self.plan.model_sha256.clone(),
                ),
                0,
            );
            let header = EventHeader {
                sequence: collector.events.len() as u64,
                timestamp_ns: har_core::unix_timestamp_nanos(),
                epoch,
                operation_index: block.candidates[0].operation_index,
            };
            collector.record_event(RuntimeEvent {
                header,
                kind: RuntimeEventKind::SpeculationResolved {
                    block_index: block.block_index,
                    candidate_horizon: block.candidate_horizon,
                    accepted_tokens,
                    rejected_tokens,
                },
            });
        }

        Ok(resolution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DispatchOutput, ExecutionEngine};
    use har_core::{BackendKind, KernelKind};
    use har_ir::{DispatchShape, OperationTable, PhysicalOperation};
    use har_plan::ExecutionPlan;
    use har_telemetry::TelemetryCollector;
    use std::collections::BTreeSet;

    /// Deterministic adapter: doubles the input; the reference equals the
    /// output for operations whose index is in `fail_at` (mismatch), else
    /// exact.
    struct ScriptedAdapter {
        fail_at: BTreeSet<u32>,
    }
    impl OperationAdapter for ScriptedAdapter {
        fn backend(&self) -> BackendKind {
            BackendKind::Cpu
        }
        fn dispatch(
            &mut self,
            operation: &PhysicalOperation,
            input: &[f32],
        ) -> Result<DispatchOutput> {
            let output: Vec<f32> = input.iter().map(|x| x * 2.0).collect();
            if self.fail_at.contains(&operation.index) {
                let reference: Vec<f32> = input.iter().map(|x| x * 3.0).collect();
                Ok(DispatchOutput {
                    output,
                    reference_output: reference,
                    bytes_moved: 16,
                    note: "scripted mismatch".into(),
                })
            } else {
                Ok(DispatchOutput {
                    output: output.clone(),
                    reference_output: output,
                    bytes_moved: 16,
                    note: "scripted exact".into(),
                })
            }
        }
    }

    fn plan_with(operation_count: u32) -> ExecutionPlan {
        let mut plan = ExecutionPlan {
            schema: har_plan::PLAN_SCHEMA.into(),
            plan_id: "speculation-test".into(),
            plan_kind: "test".into(),
            generated_at_unix_ns: 0,
            model_identity: "m".into(),
            model_sha256: "h".into(),
            hardware: har_core::HardwarePhenotype::synthetic_rdna4(),
            target_context: 1,
            kv_datatype: "f16".into(),
            mtp_enabled: true,
            quality_policy: "exact".into(),
            budget: Default::default(),
            tensor_placements: vec![],
            transfers: vec![],
            operations: OperationTable::new(),
            required_kernels: vec![],
            exactness: Default::default(),
            fallback: Default::default(),
            telemetry: Default::default(),
            assumptions: vec![],
            unresolved_risks: vec![],
            source_model_package_schema: None,
            source_model_package_sha256: None,
        };
        for index in 0..operation_count {
            plan.operations.physical.push(PhysicalOperation {
                index,
                logical_id: index,
                stable_id: format!("op.{index}"),
                backend: BackendKind::Cpu,
                kernel: KernelKind::DenseMulMat,
                input_slots: vec![],
                output_slots: vec![],
                dependencies: vec![],
                dispatch: DispatchShape::default(),
                source_tier: har_core::MemoryTier::RamMapped,
                destination_tier: har_core::MemoryTier::CpuHeap,
            });
        }
        plan
    }

    fn block(
        index: u64,
        horizon: u64,
        candidates: Vec<SpeculativeCandidate>,
        enabled: bool,
    ) -> SpeculativeBlock {
        SpeculativeBlock {
            block_index: index,
            candidate_horizon: horizon,
            candidates,
            telemetry_enabled: enabled,
        }
    }

    fn candidate(position: u32, operation_index: u32, namespace: &str) -> SpeculativeCandidate {
        SpeculativeCandidate {
            position,
            operation_index,
            namespace: namespace.into(),
            input: vec![1.0, 2.0],
        }
    }

    fn new_collector() -> TelemetryCollector {
        let epoch = har_core::EpochNamespace::new(har_core::ModelRoot::new("m", "h"), 0);
        TelemetryCollector::new(epoch, "test-plan", 256)
    }

    #[test]
    fn all_candidates_accepted() {
        let plan = plan_with(4);
        let engine = ExecutionEngine::new(plan);
        let mut adapter = ScriptedAdapter {
            fail_at: BTreeSet::new(),
        };
        let mut collector = new_collector();
        let block = block(
            0,
            4,
            vec![
                candidate(0, 0, "ns.a"),
                candidate(1, 1, "ns.b"),
                candidate(2, 2, "ns.c"),
                candidate(3, 3, "ns.d"),
            ],
            true,
        );
        let resolution = engine
            .resolve_speculative_block(&block, &mut adapter, &mut collector)
            .unwrap();
        assert_eq!(resolution.accepted_tokens, 4);
        assert_eq!(resolution.rejected_tokens, 0);
        assert_eq!(resolution.kv_namespace_commits, 4);
        assert_eq!(resolution.kv_namespace_rollbacks, 0);
        let snapshot = collector.snapshot();
        assert_eq!(snapshot.speculation.len(), 1);
        assert_eq!(snapshot.speculation_totals.accepted_tokens, 4);
        assert_eq!(snapshot.speculation_totals.rejected_tokens, 0);
    }

    #[test]
    fn partial_acceptance() {
        let plan = plan_with(4);
        let engine = ExecutionEngine::new(plan);
        let mut adapter = ScriptedAdapter {
            fail_at: BTreeSet::from([2]),
        };
        let mut collector = new_collector();
        let block = block(
            0,
            4,
            vec![
                candidate(0, 0, "ns.a"),
                candidate(1, 1, "ns.b"),
                candidate(2, 2, "ns.c"),
                candidate(3, 3, "ns.d"),
            ],
            true,
        );
        let resolution = engine
            .resolve_speculative_block(&block, &mut adapter, &mut collector)
            .unwrap();
        assert_eq!(resolution.accepted_tokens, 3);
        assert_eq!(resolution.rejected_tokens, 1);
        assert_eq!(resolution.accepted_namespaces, vec!["ns.a", "ns.b", "ns.d"]);
        assert_eq!(resolution.rolled_back_namespaces, vec!["ns.c"]);
        let snapshot = collector.snapshot();
        assert_eq!(snapshot.speculation[0].accepted_tokens, 3);
        assert_eq!(snapshot.speculation[0].rejected_tokens, 1);
    }

    #[test]
    fn full_rejection() {
        let plan = plan_with(3);
        let engine = ExecutionEngine::new(plan);
        let mut adapter = ScriptedAdapter {
            fail_at: BTreeSet::from([0, 1, 2]),
        };
        let mut collector = new_collector();
        let block = block(
            0,
            3,
            vec![
                candidate(0, 0, "ns.a"),
                candidate(1, 1, "ns.b"),
                candidate(2, 2, "ns.c"),
            ],
            true,
        );
        let resolution = engine
            .resolve_speculative_block(&block, &mut adapter, &mut collector)
            .unwrap();
        assert_eq!(resolution.accepted_tokens, 0);
        assert_eq!(resolution.rejected_tokens, 3);
        let snapshot = collector.snapshot();
        assert_eq!(snapshot.speculation[0].accepted_tokens, 0);
        assert_eq!(snapshot.speculation[0].rejected_tokens, 3);
        assert_eq!(snapshot.speculation[0].kv_namespace_rollbacks, 3);
    }

    #[test]
    fn invalid_acceptance_sum_fails_closed() {
        // The resolver counts exactly; a hand-built invalid record must be
        // rejected by the collector and leave state unchanged.
        let plan = plan_with(1);
        let engine = ExecutionEngine::new(plan);
        let mut adapter = ScriptedAdapter {
            fail_at: BTreeSet::new(),
        };
        let mut collector = new_collector();
        let block = block(
            0,
            2,
            vec![candidate(0, 0, "ns.a"), candidate(1, 0, "ns.b")],
            true,
        );
        let resolution = engine
            .resolve_speculative_block(&block, &mut adapter, &mut collector)
            .unwrap();
        assert_eq!(resolution.accepted_tokens, 2);
        assert_eq!(resolution.accepted_tokens + resolution.rejected_tokens, 2);
        // Collector-level fail-closed for a hand-built invalid record:
        let invalid = har_telemetry::SpeculationTelemetry {
            candidate_horizon: 5,
            accepted_tokens: 4,
            rejected_tokens: 2,
            draft_cost_ns: 0,
            verification_cost_ns: 0,
            kv_namespace_commits: 4,
            kv_namespace_rollbacks: 1,
        };
        assert!(collector.record_speculation(invalid).is_err());
        assert_eq!(collector.speculation.len(), 1);
        assert_eq!(collector.speculation_totals.blocks, 1);
    }

    #[test]
    fn invalid_namespace_accounting_rejected() {
        let plan = plan_with(2);
        let engine = ExecutionEngine::new(plan);
        let mut adapter = ScriptedAdapter {
            fail_at: BTreeSet::new(),
        };
        let mut collector = new_collector();
        let bad = block(
            0,
            2,
            vec![candidate(0, 0, "ns.dup"), candidate(1, 1, "ns.dup")],
            true,
        );
        assert!(engine
            .resolve_speculative_block(&bad, &mut adapter, &mut collector)
            .is_err());
        assert!(
            collector.speculation.is_empty(),
            "nothing may be recorded for an invalid block"
        );
        assert_eq!(
            collector.speculation_totals,
            har_telemetry::SpeculationTotals::default()
        );

        let empty = block(0, 1, vec![candidate(0, 0, "")], true);
        assert!(engine
            .resolve_speculative_block(&empty, &mut adapter, &mut collector)
            .is_err());

        let short = block(0, 3, vec![candidate(0, 0, "ns.a")], true);
        assert!(engine
            .resolve_speculative_block(&short, &mut adapter, &mut collector)
            .is_err());

        let out_of_range = block(0, 1, vec![candidate(0, 99, "ns.a")], true);
        assert!(engine
            .resolve_speculative_block(&out_of_range, &mut adapter, &mut collector)
            .is_err());
    }

    #[test]
    fn multiple_blocks_aggregate_correctly() {
        let plan = plan_with(4);
        let engine = ExecutionEngine::new(plan);
        let mut adapter = ScriptedAdapter {
            fail_at: BTreeSet::new(),
        };
        let mut collector = new_collector();
        let b1 = block(
            0,
            4,
            vec![
                candidate(0, 0, "a"),
                candidate(1, 1, "b"),
                candidate(2, 2, "c"),
                candidate(3, 3, "d"),
            ],
            true,
        );
        let b2 = block(
            1,
            4,
            vec![
                candidate(0, 0, "e"),
                candidate(1, 1, "f"),
                candidate(2, 2, "g"),
                candidate(3, 3, "h"),
            ],
            true,
        );
        let b3 = block(2, 2, vec![candidate(0, 0, "i"), candidate(1, 1, "j")], true);
        engine
            .resolve_speculative_block(&b1, &mut adapter, &mut collector)
            .unwrap();
        engine
            .resolve_speculative_block(&b2, &mut adapter, &mut collector)
            .unwrap();
        engine
            .resolve_speculative_block(&b3, &mut adapter, &mut collector)
            .unwrap();
        let snapshot = collector.snapshot();
        assert_eq!(snapshot.speculation.len(), 3);
        assert_eq!(snapshot.speculation_totals.blocks, 3);
        assert_eq!(snapshot.speculation_totals.candidates, 10);
        assert_eq!(snapshot.speculation_totals.accepted_tokens, 10);
        assert_eq!(snapshot.speculation_totals.rejected_tokens, 0);
    }

    #[test]
    fn disabled_telemetry_has_no_behavioral_impact() {
        let plan = plan_with(3);
        let engine = ExecutionEngine::new(plan);
        let mut enabled_adapter = ScriptedAdapter {
            fail_at: BTreeSet::from([1]),
        };
        let mut disabled_adapter = ScriptedAdapter {
            fail_at: BTreeSet::from([1]),
        };
        let mut enabled_collector = new_collector();
        let mut disabled_collector = new_collector();
        let e_block = block(
            0,
            3,
            vec![
                candidate(0, 0, "a"),
                candidate(1, 1, "b"),
                candidate(2, 2, "c"),
            ],
            true,
        );
        let d_block = block(
            0,
            3,
            vec![
                candidate(0, 0, "a"),
                candidate(1, 1, "b"),
                candidate(2, 2, "c"),
            ],
            false,
        );
        let e_res = engine
            .resolve_speculative_block(&e_block, &mut enabled_adapter, &mut enabled_collector)
            .unwrap();
        let d_res = engine
            .resolve_speculative_block(&d_block, &mut disabled_adapter, &mut disabled_collector)
            .unwrap();
        assert_eq!(e_res.accepted_tokens, d_res.accepted_tokens);
        assert_eq!(e_res.rejected_tokens, d_res.rejected_tokens);
        let d_snapshot = disabled_collector.snapshot();
        assert!(d_snapshot.speculation.is_empty());
        assert_eq!(
            d_snapshot.speculation_totals,
            har_telemetry::SpeculationTotals::default()
        );
        assert!(d_snapshot.events.events.is_empty());
    }

    #[test]
    fn old_snapshot_deserializes_and_new_snapshot_serializes() {
        let old = r#"{"schema":"har.telemetry.v1","epoch":{"model_root":{"name":"m","sha256":"h"},"graph_generation":1,"decode_epoch":1,"sequence_id":1},"plan_hash":"p","operations":[],"resources":{"requested_bytes":0,"unique_bytes":0,"useful_bytes":0,"wasted_bytes":0,"nvme_reads":0,"ram_hits":0,"vram_hits":0},"events":{"schema":"har.events.v1","capacity":0,"events":[],"dropped":0}}"#;
        let snapshot: har_telemetry::TelemetrySnapshot =
            serde_json::from_str(old).expect("old snapshot must deserialize");
        assert!(snapshot.speculation.is_empty());
        assert_eq!(
            snapshot.speculation_totals,
            har_telemetry::SpeculationTotals::default()
        );
        let plan = plan_with(2);
        let engine = ExecutionEngine::new(plan);
        let mut adapter = ScriptedAdapter {
            fail_at: BTreeSet::new(),
        };
        let mut collector = new_collector();
        let block = block(0, 2, vec![candidate(0, 0, "a"), candidate(1, 1, "b")], true);
        engine
            .resolve_speculative_block(&block, &mut adapter, &mut collector)
            .unwrap();
        let serialized =
            serde_json::to_string(&collector.snapshot()).expect("new snapshot must serialize");
        let parsed: har_telemetry::TelemetrySnapshot =
            serde_json::from_str(&serialized).expect("roundtrip");
        assert_eq!(parsed.speculation.len(), 1);
    }

    #[test]
    fn canonical_hash_is_a_pure_function_of_the_record() {
        // The canonical hash is deterministic for identical recorded bytes
        // (field-sorted serialization, no map-order or float noise).  Events
        // carry wall-clock timestamps at the real boundary (same pattern as
        // the executor at lib.rs:71), so cross-run equality is not a contract;
        // byte-level reproducibility is.
        let plan = plan_with(2);
        let engine = ExecutionEngine::new(plan);
        let mut adapter = ScriptedAdapter {
            fail_at: BTreeSet::new(),
        };
        let mut collector = new_collector();
        let block = block(0, 2, vec![candidate(0, 0, "a"), candidate(1, 1, "b")], true);
        engine
            .resolve_speculative_block(&block, &mut adapter, &mut collector)
            .unwrap();
        let first = collector.snapshot();
        let hash_of =
            |snapshot: &har_telemetry::TelemetrySnapshot| snapshot.canonical_hash().unwrap();
        let h1 = hash_of(&first);
        let serialized = serde_json::to_string(&first).unwrap();
        let reparsed: har_telemetry::TelemetrySnapshot = serde_json::from_str(&serialized).unwrap();
        assert_eq!(
            h1,
            hash_of(&reparsed),
            "hash must be stable across serialization round-trip"
        );

        // Changing any recorded field changes the hash.
        let mut mutated = first.clone();
        mutated.speculation[0].accepted_tokens += 1;
        assert_ne!(h1, hash_of(&mutated), "content change must change the hash");
    }

    #[test]
    fn executor_event_emits_exactly_one_speculation_record_per_block() {
        let plan = plan_with(3);
        let engine = ExecutionEngine::new(plan);
        let mut adapter = ScriptedAdapter {
            fail_at: BTreeSet::from([1]),
        };
        let mut collector = new_collector();
        let block = block(
            0,
            3,
            vec![
                candidate(0, 0, "a"),
                candidate(1, 1, "b"),
                candidate(2, 2, "c"),
            ],
            true,
        );
        let resolution = engine
            .resolve_speculative_block(&block, &mut adapter, &mut collector)
            .unwrap();
        assert_eq!(resolution.accepted_tokens + resolution.rejected_tokens, 3);
        let snapshot = collector.snapshot();
        assert_eq!(
            snapshot.speculation.len(),
            1,
            "exactly one speculation record"
        );
        let speculation_events: Vec<_> = snapshot
            .events
            .events
            .iter()
            .filter(|e| matches!(e.kind, RuntimeEventKind::SpeculationResolved { .. }))
            .collect();
        assert_eq!(
            speculation_events.len(),
            1,
            "exactly one SpeculationResolved event"
        );
        assert_eq!(snapshot.speculation[0].candidate_horizon, 3);
        assert_eq!(snapshot.speculation[0].accepted_tokens, 2);
        assert_eq!(snapshot.speculation[0].rejected_tokens, 1);
        assert_eq!(snapshot.speculation_totals.blocks, 1);
    }

    #[test]
    fn rejected_positions_never_count_as_accepted() {
        let plan = plan_with(3);
        let engine = ExecutionEngine::new(plan);
        let mut adapter = ScriptedAdapter {
            fail_at: BTreeSet::from([1]),
        };
        let mut collector = new_collector();
        let block = block(
            0,
            3,
            vec![
                candidate(0, 0, "a"),
                candidate(1, 1, "b"),
                candidate(2, 2, "c"),
            ],
            true,
        );
        engine
            .resolve_speculative_block(&block, &mut adapter, &mut collector)
            .unwrap();
        let snapshot = collector.snapshot();
        let spec = &snapshot.speculation[0];
        assert_eq!(
            spec.accepted_tokens + spec.rejected_tokens,
            spec.candidate_horizon
        );
        assert!(spec.rejected_tokens > 0);
        assert_eq!(spec.kv_namespace_commits, spec.accepted_tokens);
        assert_eq!(spec.kv_namespace_rollbacks, spec.rejected_tokens);
    }
}
