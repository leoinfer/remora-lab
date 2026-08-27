//! Fast/Slow Adaptation Clocks (REMORA-22).
//!
//! - Fast clock: updated on every block-scale observation; only moves one
//!   tick per call, so replay determinism holds.
//! - Slow clock: a windowed aggregate over the fast stream that can be
//!   invalidated when the clock basis (model root / graph identity / worker
//!   set) changes; invalidation does not reset accumulated totals, but it
//!   forces a fresh window so adaptation does not carry stale samples.

use crate::common::{ClockTicks, MiB};
use serde::{Deserialize, Serialize};

/// Basis identity the slow clock is allowed to extrapolate on.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockBasis {
    pub model_root: String,
    pub graph_identity: String,
    pub worker_set: String,
}

/// A single block-scale observation for the fast clock.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FastObservation {
    pub transfer_cost_ns: u64,
    pub compute_cost_ms: u64,
    pub was_useful: bool,
    pub miss: bool,
}

/// Fast clock: monotonic epoch, updated once per observation, bounded
/// derivative.  `window` counts observations since last slow flush.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FastClock {
    pub epoch: u64,
    pub window: u32,
    pub accepted: u64,
    pub rejected: u64,
    pub last_acceptance_ratio: f64,
    pub transfer_ns_total: u64,
    pub compute_ms_total: u64,
}

impl Default for FastClock {
    fn default() -> Self {
        Self {
            epoch: 0,
            window: 0,
            accepted: 0,
            rejected: 0,
            last_acceptance_ratio: 1.0,
            transfer_ns_total: 0,
            compute_ms_total: 0,
        }
    }
}

impl FastClock {
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe one block-scale event; increments epoch by exactly one.
    pub fn observe(&mut self, obs: FastObservation) {
        if obs.was_useful {
            self.accepted += 1;
        } else {
            self.rejected += 1;
        }
        self.transfer_ns_total = self.transfer_ns_total.saturating_add(obs.transfer_cost_ns);
        self.compute_ms_total = self.compute_ms_total.saturating_add(obs.compute_cost_ms);
        self.window = self.window.saturating_add(1);
        self.epoch = self.epoch.saturating_add(1);
        let total = self.accepted.saturating_add(self.rejected);
        if total > 0 {
            // bounded moving average keeps replay deterministic
            self.last_acceptance_ratio = (self.accepted as f64) / (total as f64);
        }
    }

    pub fn ticks(&self) -> ClockTicks {
        ClockTicks {
            fast: self.epoch,
            slow: 0,
        }
    }
}

/// Slow clock: windowed, invalidatable, never extrapolates past its basis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SlowClock {
    pub basis: ClockBasis,
    pub epoch: u64,
    pub window: u32,
    pub max_window: u32,
    pub acceptance_sum: u64,
    pub samples: u64,
    pub hot_worker_bytes: MiB,
    pub queue_depth_avg: u64,
    pub transfer_latency_ns_avg: u64,
    pub valid: bool,
}

impl Default for SlowClock {
    fn default() -> Self {
        Self {
            basis: ClockBasis::default(),
            epoch: 0,
            window: 0,
            max_window: 256,
            acceptance_sum: 0,
            samples: 0,
            hot_worker_bytes: 0,
            queue_depth_avg: 0,
            transfer_latency_ns_avg: 0,
            valid: true,
        }
    }
}

impl SlowClock {
    pub fn new(max_window: u32) -> Self {
        Self {
            max_window,
            ..Self::default()
        }
    }

    /// Advance the slow clock from a fast-clock sample.  Invalidates (opens
    /// a new window) when basis changed.  Returns true when the window
    /// rolled (state flush).
    pub fn advance(&mut self, basis: &ClockBasis, fast: &FastClock) -> bool {
        if self.basis != *basis {
            self.basis = basis.clone();
            self.valid = true;
            self.window = 0;
            self.samples = 0;
            self.acceptance_sum = 0;
        }
        self.window = self.window.saturating_add(1);
        self.acceptance_sum = self.acceptance_sum.saturating_add(fast.accepted);
        self.samples = self.samples.saturating_add(1);
        self.epoch = self.epoch.saturating_add(1);
        if self.window >= self.max_window {
            self.window = 0;
            true
        } else {
            false
        }
    }

    /// Mean acceptance over the current window, or None when no samples.
    pub fn acceptance_ratio(&self) -> Option<f64> {
        if self.samples == 0 {
            None
        } else {
            Some(self.acceptance_sum as f64 / self.samples as f64)
        }
    }

    pub fn ticks(&self) -> ClockTicks {
        ClockTicks {
            fast: 0,
            slow: self.epoch,
        }
    }

    pub fn invalidate(&mut self) {
        self.valid = false;
        self.window = 0;
        self.samples = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_clock_advances_one_tick_per_observation() {
        let mut f = FastClock::new();
        f.observe(FastObservation {
            transfer_cost_ns: 1,
            compute_cost_ms: 1,
            was_useful: true,
            miss: false,
        });
        f.observe(FastObservation {
            transfer_cost_ns: 1,
            compute_cost_ms: 1,
            was_useful: false,
            miss: true,
        });
        assert_eq!(f.epoch, 2);
        assert_eq!(f.accepted, 1);
        assert_eq!(f.rejected, 1);
        assert_eq!(f.last_acceptance_ratio, 0.5);
    }

    #[test]
    fn slow_clock_invalidates_on_basis_change() {
        let mut s = SlowClock::new(4);
        let mut f = FastClock::new();
        let b1 = ClockBasis {
            model_root: "m1".into(),
            graph_identity: "g1".into(),
            worker_set: "w1".into(),
        };
        let b2 = ClockBasis {
            model_root: "m2".into(),
            graph_identity: "g2".into(),
            worker_set: "w1".into(),
        };
        f.observe(FastObservation {
            transfer_cost_ns: 1,
            compute_cost_ms: 1,
            was_useful: true,
            miss: false,
        });
        let _ = s.advance(&b1, &f);
        assert!(s.valid);
        let rolled = s.advance(&b2, &f);
        // basis change reset the window; samples start over
        assert_eq!(s.samples, 1);
        assert_eq!(s.epoch, 2);
        assert!(!rolled);
    }
}
