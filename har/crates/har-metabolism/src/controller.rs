//! MetabolismController (REMORA-25): the single composable control surface.
//!
//! It composes Reserve (tiers + mobilization), Waste Ledger (authoritative
//! book), Fast/Slow clocks, Moving Maintenance Setpoint, Uncertainty-Adjusted
//! Safe Surplus, Portion, Reclaim, Salvage and artifact state under one roof.
//! It is deliberately free of wall-clock time and guesswork: every output
//! comes from an observed input or an explicit UNKNOWN.

use crate::artifact::{ArtifactEnvelope, ReuseClass};
use crate::clock::{ClockBasis, FastClock, FastObservation, SlowClock};
use crate::common::{ClockTicks, MiB, Tokens};
use crate::energy::EnergyLabel;
use crate::error::MetabolismResult;
use crate::ledger::{CreditEvidence, LedgerClass, WasteLedger};
use crate::portion::{Portion, PortionDecision, PortionInput};
use crate::reclaim::{Reclaim, ReclaimDecision};
use crate::reserve::{ReserveDim, ReserveTable};
use crate::salvage::{Salvage, SalvageCandidate, SalvageDecision};
use crate::setpoint::{MaintenanceObservations, MaintenanceSetpoint, SetpointEstimator};
use crate::snapshot::MetabolismSnapshot;
use crate::surplus::{Surplus, SurplusInputs};
use serde::{Deserialize, Serialize};

/// Controller configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ControllerConfig {
    pub vram_capacity_mib: MiB,
    pub ram_capacity_mib: MiB,
    pub protected_min_mib: MiB,
    pub gpu_compute_budget: u64,
    pub slow_window: u32,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            vram_capacity_mib: 32768,
            ram_capacity_mib: 65536,
            protected_min_mib: 2048,
            gpu_compute_budget: 4096,
            slow_window: 256,
        }
    }
}

/// The controller: deterministic, wall-clock-free, fail-closed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetabolismController {
    pub reserve: ReserveTable,
    pub ledger: WasteLedger,
    pub fast_clock: FastClock,
    pub slow_clock: SlowClock,
    pub basis: ClockBasis,
    pub setpoint: SetpointEstimator,
    pub portion: Portion,
    pub reclaim: Reclaim,
    pub salvage: Salvage,
    pub surplus: Surplus,
    pub energy: EnergyLabel,
    pub config: ControllerConfig,
    /// Recent maintenance estimate (recomputed at slow-window roll).
    pub maintenance: MaintenanceSetpoint,
}

impl MetabolismController {
    pub fn new(config: ControllerConfig) -> Self {
        let reserve = ReserveTable::new(&[
            (
                ReserveDim::Vram,
                config.vram_capacity_mib,
                config.protected_min_mib,
            ),
            (
                ReserveDim::Ram,
                config.ram_capacity_mib,
                config.protected_min_mib,
            ),
            (ReserveDim::GpuCompute, config.gpu_compute_budget, 0),
            (ReserveDim::QueueFenceSlack, 64, 0),
            (ReserveDim::NvmeReadBudget, 1 << 20, 0),
        ]);
        let maintenance = SetpointEstimator::default().compute(&MaintenanceObservations::default());
        Self {
            reserve,
            ledger: WasteLedger::new(),
            fast_clock: FastClock::new(),
            slow_clock: SlowClock::new(config.slow_window),
            basis: ClockBasis::default(),
            setpoint: SetpointEstimator::default(),
            portion: Portion::new(),
            reclaim: Reclaim::new(),
            salvage: Salvage::new(),
            surplus: Surplus::new(),
            energy: EnergyLabel::unknown(),
            config,
            maintenance,
        }
    }

    pub fn ticks(&self) -> ClockTicks {
        ClockTicks {
            fast: self.fast_clock.epoch,
            slow: self.slow_clock.epoch,
        }
    }

    /// Resolve the maintenance setpoint from observations and update the
    /// running estimate.
    pub fn update_setpoint(&mut self, obs: &MaintenanceObservations) -> MaintenanceSetpoint {
        self.maintenance = self.setpoint.compute(obs);
        self.maintenance
    }

    /// Compute the safe surplus against the current maintenance setpoint.
    pub fn safe_surplus(&self, inputs: &SurplusInputs) -> crate::surplus::SafeSurplus {
        self.surplus.compute(inputs)
    }

    /// Decide whether to admit a portion of optional work.
    pub fn decide_portion(
        &mut self,
        input: PortionInput,
        class: ReuseClass,
    ) -> MetabolismResult<PortionDecision> {
        self.portion.decide(input, &self.reserve, class)
    }

    /// Record an observed block-scale observation; advances the fast clock
    /// by exactly one.  When the slow window rolls, maintenance is resolved.
    pub fn observe(&mut self, obs: FastObservation, mtp_acceptance: Option<u64>) {
        self.fast_clock.observe(obs);
        let rolled = self.slow_clock.advance(&self.basis, &self.fast_clock);
        if rolled {
            let base = MaintenanceObservations {
                kv_occupancy: 0,
                mtp_acceptance_permille: mtp_acceptance,
                queue_depth: self.slow_clock.queue_depth_avg as u32,
            };
            self.maintenance = self.setpoint.compute(&base);
        }
    }

    /// Switch/assert the clock basis (model root / graph identity / worker
    /// set).  A basis change invalidates the slow clock's window.
    pub fn set_basis(&mut self, basis: ClockBasis) {
        self.basis = basis;
        self.slow_clock.invalidate();
    }

    /// Reclaim decision for a spent ledger entry against current context.
    pub fn reclaim(
        &mut self,
        entry_seq: u64,
        current: &ArtifactEnvelope,
    ) -> MetabolismResult<ReclaimDecision> {
        self.reclaim.classify(&self.ledger, entry_seq, current)
    }

    /// Record spent work in the ledger (authoritative or speculative).
    pub fn record_spent(
        &mut self,
        class: LedgerClass,
        token_identity: String,
        work_mib_ms: u64,
        tokens: Tokens,
        gen: u64,
    ) -> MetabolismResult<u64> {
        self.ledger
            .record_spent(class, token_identity, work_mib_ms, tokens, gen)
    }

    /// Record a reclaim record that later evidence may build on.
    pub fn record_reclaim(&mut self, identity: &str, base_seq: u64) -> MetabolismResult<bool> {
        self.ledger.record_reclaim(identity, base_seq)
    }

    /// Record a reuse credit (evidence-gated).
    pub fn grant_reuse_credit(&mut self, evidence: &CreditEvidence) -> MetabolismResult<bool> {
        self.ledger.grant_reuse_credit(evidence)
    }

    /// Record an overlap credit; requires actually measured bytes.
    pub fn grant_overlap_credit(
        &mut self,
        witness_identity: String,
        bytes_moved: u64,
    ) -> MetabolismResult<bool> {
        self.ledger
            .grant_overlap_credit(witness_identity, bytes_moved)
    }

    /// Score a salvage candidate (no state change).
    pub fn salvage_score(&self, candidate: &SalvageCandidate) -> SalvageDecision {
        self.salvage.score(candidate)
    }

    /// Produce a snapshot row (an immutable, additive telemetry record).
    pub fn snapshot(&self) -> MetabolismSnapshot {
        let ticks = self.ticks();
        let vram = self.reserve.account(ReserveDim::Vram);
        let ram = self.reserve.account(ReserveDim::Ram);
        let safe = Surplus::new().compute(&SurplusInputs {
            total_reserve: self
                .reserve
                .total_committed()
                .saturating_add(vram.available()),
            maintenance_setpoint_mib: self.maintenance.total_mib(),
            miss_rate_penalty_mib: 0,
            contention_mib: None,
            interference_mib: None,
            unknown_shrinks: true,
        });
        MetabolismSnapshot::new(
            self.ledger.totals.useful_tokens,
            self.maintenance.total_mib(),
            vram.committed,
            ram.committed,
            safe.optional_mib,
            safe.optional_mib,
            self.ledger.reclaimed.len() as u64,
            0,
            self.ledger.totals.waste_spec_ms,
            self.ledger.totals.wasted_mib,
            self.ledger.totals.reuse_credit_ms,
            self.ledger.totals.overlap_credit_ms,
            vram.debt,
            ticks,
            self.energy,
        )
    }
}
