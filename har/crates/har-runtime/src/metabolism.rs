//! Runtime-owned REMORA controller and telemetry join.
//!
//! Exact token ownership stays with the runtime/decode owner. REMORA records
//! only after that owner calls `record_exact_token`; it never authorizes a
//! fallback or changes mathematical dispatch.

use har_core::{EpochNamespace, HardwarePhenotype, ModelRoot};
use har_events::{EventHeader, RuntimeEvent, RuntimeEventKind};
use har_metabolism::clock::{ClockBasis, FastObservation};
use har_metabolism::controller::{ControllerConfig, MetabolismController};
use har_metabolism::snapshot::MetabolismSnapshot;
use har_plan::LoadedPlan;
use har_telemetry::{TelemetryCollector, TelemetrySnapshot};

pub struct RuntimeMetabolism {
    pub controller: MetabolismController,
    pub telemetry: TelemetryCollector,
}

impl RuntimeMetabolism {
    pub fn from_loaded(
        loaded: &LoadedPlan,
        hardware: &HardwarePhenotype,
        sequence_id: u64,
        event_capacity: usize,
    ) -> Self {
        Self::from_identity(
            &loaded.plan.model_identity,
            &loaded.plan.model_sha256,
            &loaded.plan_identity_hash,
            hardware,
            sequence_id,
            event_capacity,
        )
    }

    pub fn from_identity(
        model_name: &str,
        model_sha256: &str,
        plan_hash: &str,
        hardware: &HardwarePhenotype,
        sequence_id: u64,
        event_capacity: usize,
    ) -> Self {
        let vram_capacity_mib = (hardware.gpu.safe_allocatable_vram_bytes / 1_048_576).max(1);
        let ram_capacity_mib = (hardware.ram.total_bytes / 1_048_576).max(1);
        let protected_min_mib = vram_capacity_mib.min(2048);
        let config = ControllerConfig {
            vram_capacity_mib,
            ram_capacity_mib,
            protected_min_mib,
            gpu_compute_budget: 4096,
            slow_window: 256,
        };
        let model_root = ModelRoot::new(model_name, model_sha256);
        let epoch = EpochNamespace::new(model_root.clone(), sequence_id);
        let mut controller = MetabolismController::new(config);
        controller.set_basis(ClockBasis {
            model_root: model_root.identity(),
            graph_identity: plan_hash.to_owned(),
            worker_set: hardware.identity(),
        });
        Self {
            controller,
            telemetry: TelemetryCollector::new(epoch, plan_hash, event_capacity),
        }
    }

    /// Record one exact committed token. The caller supplies observed work;
    /// no timing or energy value is synthesized here.
    #[allow(clippy::too_many_arguments)]
    pub fn record_exact_token(
        &mut self,
        token_identity: impl Into<String>,
        work_mib_ms: u64,
        transfer_cost_ns: u64,
        compute_cost_ms: u64,
        generation: u64,
        operation_index: u32,
        timestamp_ns: u64,
    ) -> Result<MetabolismSnapshot, har_metabolism::MetabolismError> {
        let token_identity = token_identity.into();
        if token_identity.is_empty() {
            return Err(har_metabolism::MetabolismError::FailClosed(
                "exact token identity is unknown",
            ));
        }
        self.controller.record_spent(
            har_metabolism::ledger::LedgerClass::AuthoritativeUseful,
            token_identity,
            work_mib_ms,
            1,
            generation,
        )?;
        self.controller.observe(
            FastObservation {
                transfer_cost_ns,
                compute_cost_ms,
                was_useful: true,
                miss: false,
            },
            None,
        );
        let snapshot = self.controller.snapshot();
        self.record_snapshot_event(&snapshot, operation_index, timestamp_ns);
        Ok(snapshot)
    }

    /// Record a non-token observation without creating useful-token ledger
    /// credit. This is suitable for residency/transfer observations.
    pub fn record_observation(
        &mut self,
        transfer_cost_ns: u64,
        compute_cost_ms: u64,
        useful: bool,
        miss: bool,
        operation_index: u32,
        timestamp_ns: u64,
    ) -> MetabolismSnapshot {
        self.controller.observe(
            FastObservation {
                transfer_cost_ns,
                compute_cost_ms,
                was_useful: useful,
                miss,
            },
            None,
        );
        let snapshot = self.controller.snapshot();
        self.record_snapshot_event(&snapshot, operation_index, timestamp_ns);
        snapshot
    }

    pub fn snapshot(&self) -> MetabolismSnapshot {
        self.controller.snapshot()
    }

    pub fn take_telemetry(&mut self) -> TelemetrySnapshot {
        let replacement = TelemetryCollector::new(
            self.telemetry.epoch.clone(),
            self.telemetry.plan_hash.clone(),
            self.telemetry.events.capacity,
        );
        std::mem::replace(&mut self.telemetry, replacement).snapshot()
    }

    fn record_snapshot_event(
        &mut self,
        snapshot: &MetabolismSnapshot,
        operation_index: u32,
        timestamp_ns: u64,
    ) {
        self.telemetry.record_metabolism(snapshot.to_totals());
        self.telemetry.record_event(RuntimeEvent {
            header: EventHeader {
                sequence: snapshot.fast_epoch,
                timestamp_ns,
                epoch: EpochNamespace {
                    decode_epoch: snapshot.fast_epoch,
                    ..self.telemetry.epoch.clone()
                },
                operation_index,
            },
            kind: RuntimeEventKind::MetabolismSnapshot {
                exact_tokens: snapshot.exact_tokens,
                maintenance_vram_mib: snapshot.maintenance_vram_mib,
                reserve_vram_mib: snapshot.reserve_vram_mib,
                reserve_ram_mib: snapshot.reserve_ram_mib,
                safe_surplus_mib: snapshot.safe_surplus_mib,
                optional_budget_mib: snapshot.optional_budget_mib,
                reclaimed: snapshot.reclaimed,
                salvaged: snapshot.salvaged,
                waste_spec_compute_ms: snapshot.waste_spec_compute_ms,
                waste_prefetch_unused_mib: snapshot.waste_prefetch_unused_mib,
                reuse_credit_ms: snapshot.reuse_credit_ms,
                overlap_credit_ms: snapshot.overlap_credit_ms,
                reserve_debt_mib: snapshot.reserve_debt_mib,
                fast_epoch: snapshot.fast_epoch,
                slow_epoch: snapshot.slow_epoch,
                energy: format!("{:?}", snapshot.energy),
            },
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_token_record_updates_totals_and_event() {
        let mut state = RuntimeMetabolism::from_identity(
            "model",
            "sha",
            "plan",
            &HardwarePhenotype::synthetic_rdna4(),
            7,
            8,
        );
        let snapshot = state
            .record_exact_token("token@0", 3, 11, 2, 1, 4, 99)
            .unwrap();
        assert_eq!(snapshot.exact_tokens, 1);
        assert_eq!(snapshot.fast_epoch, 1);
        assert_eq!(state.telemetry.metabolism.exact_tokens, 1);
        assert!(matches!(
            state.telemetry.events.events[0].kind,
            RuntimeEventKind::MetabolismSnapshot {
                exact_tokens: 1,
                ..
            }
        ));
        assert_eq!(state.telemetry.events.events[0].header.timestamp_ns, 99);
    }

    #[test]
    fn energy_is_explicitly_unknown_by_default() {
        let state = RuntimeMetabolism::from_identity(
            "model",
            "sha",
            "plan",
            &HardwarePhenotype::synthetic_rdna4(),
            1,
            1,
        );
        assert_eq!(
            state.snapshot().energy,
            har_metabolism::energy::EnergyLabel::unknown()
        );
    }
}
