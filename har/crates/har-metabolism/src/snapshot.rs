//! REMORA runtime snapshot (REMORA-15): the immutable, additive telemetry
//! row that mirrors `har_telemetry::MetabolismTotals`.  Every field is
//! measured or explicitly UNKNOWN inside the controller; nothing is invented.

use crate::common::{ClockTicks, MiB};
use crate::energy::EnergyLabel;
use serde::{Deserialize, Serialize};

/// The full metabolism snapshot of the running controller.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetabolismSnapshot {
    pub exact_tokens: u64,
    pub maintenance_vram_mib: MiB,
    pub reserve_vram_mib: MiB,
    pub reserve_ram_mib: MiB,
    pub safe_surplus_mib: MiB,
    pub optional_budget_mib: MiB,
    pub reclaimed: u64,
    pub salvaged: u64,
    pub waste_spec_compute_ms: u64,
    pub waste_prefetch_unused_mib: MiB,
    pub reuse_credit_ms: u64,
    pub overlap_credit_ms: u64,
    pub reserve_debt_mib: MiB,
    pub fast_epoch: u64,
    pub slow_epoch: u64,
    pub energy: EnergyLabel,
}

impl MetabolismSnapshot {
    /// Convenience constructor for tests / adapters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        exact_tokens: u64,
        maintenance_vram_mib: MiB,
        reserve_vram_mib: MiB,
        reserve_ram_mib: MiB,
        safe_surplus_mib: MiB,
        optional_budget_mib: MiB,
        reclaimed: u64,
        salvaged: u64,
        waste_spec_compute_ms: u64,
        waste_prefetch_unused_mib: MiB,
        reuse_credit_ms: u64,
        overlap_credit_ms: u64,
        reserve_debt_mib: MiB,
        clocks: ClockTicks,
        energy: EnergyLabel,
    ) -> Self {
        Self {
            exact_tokens,
            maintenance_vram_mib,
            reserve_vram_mib,
            reserve_ram_mib,
            safe_surplus_mib,
            optional_budget_mib,
            reclaimed,
            salvaged,
            waste_spec_compute_ms,
            waste_prefetch_unused_mib,
            reuse_credit_ms,
            overlap_credit_ms,
            reserve_debt_mib,
            fast_epoch: clocks.fast,
            slow_epoch: clocks.slow,
            energy,
        }
    }

    /// Emit the telemetry totals row.  Nothing above is invented; energy
    /// scope is asserted in `energy.rs` (default UNKNOWN).
    pub fn to_totals(&self) -> har_telemetry::MetabolismTotals {
        har_telemetry::MetabolismTotals {
            exact_tokens: self.exact_tokens,
            maintenance_vram_mib: self.maintenance_vram_mib,
            reserve_vram_mib: self.reserve_vram_mib,
            reserve_ram_mib: self.reserve_ram_mib,
            safe_surplus_mib: self.safe_surplus_mib,
            optional_budget_mib: self.optional_budget_mib,
            reclaimed: self.reclaimed,
            salvaged: self.salvaged,
            waste_spec_compute_ms: self.waste_spec_compute_ms,
            waste_prefetch_unused_mib: self.waste_prefetch_unused_mib,
            reuse_credit_ms: self.reuse_credit_ms,
            overlap_credit_ms: self.overlap_credit_ms,
            reserve_debt_mib: self.reserve_debt_mib,
            fast_epoch: self.fast_epoch,
            slow_epoch: self.slow_epoch,
        }
    }
}
