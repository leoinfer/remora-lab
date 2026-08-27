//! Tiered Reserve (REMORA-16) and Reserve Mobilization (REMORA-17).
//!
//! Reserve is NOT just free memory: each dimension tracks capacity, committed,
//! available, protected minimum, debt, expiry/recovery, and a pressure class.
//! A reserve dimension supports hard fail-closed limits and a post-action
//! viability check for mobilization.

use crate::common::MiB;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The explicit reserve dimensions implemented in V1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ReserveDim {
    Vram,
    Ram,
    NvmeReadBudget,
    GpuCompute,
    CpuBudget,
    QueueFenceSlack,
    EnergyThermal,
}

impl ReserveDim {
    pub fn all() -> [ReserveDim; 7] {
        [
            ReserveDim::Vram,
            ReserveDim::Ram,
            ReserveDim::NvmeReadBudget,
            ReserveDim::GpuCompute,
            ReserveDim::CpuBudget,
            ReserveDim::QueueFenceSlack,
            ReserveDim::EnergyThermal,
        ]
    }
}

/// Pressure class derived from the ratio `(committed + debt) / capacity`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PressureClass {
    Low,
    Normal,
    Elevated,
    Critical,
}

impl PressureClass {
    pub fn from_ratio(used: MiB, capacity: MiB) -> Self {
        if capacity == 0 {
            return PressureClass::Critical;
        }
        let ratio = (used as f64) / (capacity as f64);
        if ratio >= 0.95 {
            PressureClass::Critical
        } else if ratio >= 0.8 {
            PressureClass::Elevated
        } else if ratio >= 0.6 {
            PressureClass::Normal
        } else {
            PressureClass::Low
        }
    }
}

/// Reserve state for one dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReserveAccount {
    pub capacity: MiB,
    pub committed: MiB,
    /// Protected minimum that cannot be used by optional work.
    pub protected_min: MiB,
    /// Unreturned debt (reserve borrowed into the protected zone).
    pub debt: MiB,
    /// Fast-clock epoch at which debt is expected to recover.
    pub recovery_epoch: u64,
    /// Hard fail-closed flag: when set, available is forced to zero.
    pub fail_closed: bool,
    pub pressure: PressureClass,
}

impl ReserveAccount {
    pub fn new(capacity: MiB, protected_min: MiB) -> Self {
        Self {
            capacity,
            committed: 0,
            protected_min,
            debt: 0,
            recovery_epoch: 0,
            fail_closed: false,
            pressure: PressureClass::from_ratio(0, capacity),
        }
    }

    /// Headroom beyond the protected floor that optional work may use.
    pub fn available(&self) -> MiB {
        if self.fail_closed {
            return 0;
        }
        self.capacity.saturating_sub(
            self.protected_min
                .saturating_add(self.committed)
                .saturating_add(self.debt),
        )
    }

    pub fn check(&self, need: MiB) -> bool {
        self.available() >= need
    }

    pub fn commit(&mut self, need: MiB) -> bool {
        if !self.check(need) {
            return false;
        }
        self.committed = self.committed.saturating_add(need);
        self.refresh_pressure();
        true
    }

    pub fn release(&mut self, need: MiB) {
        self.committed = self.committed.saturating_sub(need.min(self.committed));
        self.refresh_pressure();
    }

    /// Mobilize against the protected minimum.  Only used after a
    /// post-action viability check; the amount borrowed into the protected
    /// zone is recorded as debt.
    pub fn mobilize(&mut self, need: MiB, now: u64) -> bool {
        let physical_headroom = self.capacity.saturating_sub(self.committed);
        if need > physical_headroom {
            return false;
        }
        let borrow_before = self
            .committed
            .saturating_sub(self.capacity.saturating_sub(self.protected_min));
        self.committed = self.committed.saturating_add(need);
        let borrow_after = self
            .committed
            .saturating_sub(self.capacity.saturating_sub(self.protected_min));
        self.debt = self
            .debt
            .saturating_add(borrow_after.saturating_sub(borrow_before));
        self.recovery_epoch = now.saturating_add(100);
        self.refresh_pressure();
        true
    }

    /// Post-action viability check (REMORA-17): the action restores viability
    /// only if the reserve returns to a healthy pressure class.
    pub fn post_action_viable(&self, target: PressureClass) -> bool {
        self.pressure <= target
    }

    pub fn settle_debt(&mut self, at_epoch: u64) {
        if at_epoch >= self.recovery_epoch {
            self.debt = 0;
            self.recovery_epoch = 0;
            self.refresh_pressure();
        }
    }

    fn refresh_pressure(&mut self) {
        self.pressure =
            PressureClass::from_ratio(self.committed.saturating_add(self.debt), self.capacity);
    }
}

/// The reserve table.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReserveTable {
    pub accounts: BTreeMap<ReserveDim, ReserveAccount>,
    pub epoch: u64,
}

impl ReserveTable {
    pub fn new(capacities: &[(ReserveDim, MiB, MiB)]) -> Self {
        let accounts = capacities
            .iter()
            .map(|(dim, capacity, protected)| (*dim, ReserveAccount::new(*capacity, *protected)))
            .collect();
        Self { accounts, epoch: 0 }
    }

    pub fn account(&self, dim: ReserveDim) -> ReserveAccount {
        self.accounts
            .get(&dim)
            .copied()
            .unwrap_or_else(|| ReserveAccount::new(0, 0))
    }

    pub fn check(&self, dim: ReserveDim, need: MiB) -> bool {
        self.account(dim).check(need)
    }

    pub fn commit(&mut self, dim: ReserveDim, need: MiB) -> bool {
        if dim == ReserveDim::EnergyThermal {
            // energy reserve is evidence-gated; runtime bytes are not
            // committed without a physical source.
            return true;
        }
        self.accounts
            .entry(dim)
            .or_insert_with(|| ReserveAccount::new(0, 0))
            .commit(need)
    }

    pub fn release(&mut self, dim: ReserveDim, need: MiB) {
        if let Some(account) = self.accounts.get_mut(&dim) {
            account.release(need);
        }
    }

    pub fn mobilize(&mut self, dim: ReserveDim, need: MiB, now: u64) -> bool {
        if let Some(account) = self.accounts.get_mut(&dim) {
            account.mobilize(need, now)
        } else {
            false
        }
    }

    pub fn advance(&mut self, epoch: u64) {
        self.epoch = epoch;
        for account in self.accounts.values_mut() {
            account.settle_debt(epoch);
        }
    }

    /// Aggregate reserved footprint in MiB across Vram/Ram for telemetry.
    pub fn total_committed(&self) -> MiB {
        [ReserveDim::Vram, ReserveDim::Ram]
            .iter()
            .filter_map(|dim| self.accounts.get(dim))
            .map(|a| a.committed.saturating_add(a.debt))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_protects_the_floor() {
        let mut account = ReserveAccount::new(100, 20);
        // available = 100 - 20 = 80
        assert_eq!(account.available(), 80);
        assert!(account.commit(80));
        assert_eq!(account.available(), 0);
        assert!(!account.check(1));
        account.release(30);
        assert_eq!(account.available(), 30);
        assert!(account.check(30));
        assert!(!account.check(31));
    }

    #[test]
    fn mobilize_records_debt_and_recovers() {
        let mut account = ReserveAccount::new(100, 20);
        assert!(account.commit(80));
        assert!(account.mobilize(20, 0)); // borrow 20 into protected zone
        assert_eq!(account.debt, 20);
        assert_eq!(account.committed, 100);
        assert_eq!(account.available(), 0);
        assert_eq!(account.pressure, PressureClass::Critical);
        assert!(!account.post_action_viable(PressureClass::Normal));
        account.settle_debt(100);
        assert_eq!(account.debt, 0);
        assert_eq!(account.committed, 100);
        assert_eq!(account.available(), 0); // committed still holds the space
        account.release(100);
        assert_eq!(account.available(), 80);
    }

    #[test]
    fn table_commit_and_pressure() {
        let mut t = ReserveTable::new(&[(ReserveDim::Vram, 100, 20)]);
        assert!(t.commit(ReserveDim::Vram, 80));
        assert!(!t.check(ReserveDim::Vram, 1));
        assert!(t.account(ReserveDim::Vram).available() == 0);
        t.advance(200);
        assert!(t.account(ReserveDim::Vram).debt == 0);
    }
}
