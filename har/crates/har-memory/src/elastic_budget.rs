//! Elastic VRAM budget with idle-only adjustment windows.
//!
//! Concept source: public memory-management references; no upstream
//! implementation is copied. Runtime VRAM repartitioning must happen only
//! while the engine is idle, must be validated before teardown, and must use a
//! rollback ladder instead of failing open. This module is the backend-neutral
//! policy machine; a device adapter supplies free VRAM at commit time.

use har_core::{HarError, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnginePhase {
    Idle,
    Prefill,
    Decode,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ElasticBudgetPolicy {
    /// Budget can never drop below this floor (decode/KV viability).
    pub min_vram_bytes: u64,
    /// Hard ceiling (device limit).
    pub max_vram_bytes: u64,
    /// Granularity of one adjustment step / rollback rung.
    pub step_bytes: u64,
    /// Free VRAM that must remain unused after any committed grow.
    pub headroom_bytes: u64,
}

impl ElasticBudgetPolicy {
    pub fn validate(&self) -> Result<()> {
        if self.step_bytes == 0 || self.min_vram_bytes > self.max_vram_bytes {
            return Err(HarError::Invalid {
                kind: "elastic budget policy",
                message: "step must be > 0 and min <= max".into(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum AdjustmentKind {
    Grow,
    Shrink,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BudgetAdjustment {
    pub epoch: u64,
    pub kind: AdjustmentKind,
    pub requested_bytes: u64,
    pub committed_bytes: u64,
    pub rollback_rungs: u32,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub struct PendingAdjustment {
    kind: AdjustmentKind,
    target_bytes: u64,
    reason: String,
}

/// State machine owning the current VRAM budget. Adjustments are requested by
/// any subsystem but only *committed* from [`EnginePhase::Idle`] with a fresh
/// free-VRAM measurement, mirroring validate-before-teardown semantics.
#[derive(Clone, Debug)]
pub struct ElasticVramBudget {
    policy: ElasticBudgetPolicy,
    phase: EnginePhase,
    current_bytes: u64,
    epoch: u64,
    pending: Option<PendingAdjustment>,
    log: Vec<BudgetAdjustment>,
}

impl ElasticVramBudget {
    pub fn new(policy: ElasticBudgetPolicy, initial_bytes: u64) -> Result<Self> {
        policy.validate()?;
        if initial_bytes < policy.min_vram_bytes || initial_bytes > policy.max_vram_bytes {
            return Err(HarError::Invalid {
                kind: "elastic budget",
                message: format!(
                    "initial {} outside [{}, {}]",
                    initial_bytes, policy.min_vram_bytes, policy.max_vram_bytes
                ),
            });
        }
        Ok(Self {
            policy,
            phase: EnginePhase::Idle,
            current_bytes: initial_bytes,
            epoch: 0,
            pending: None,
            log: Vec::new(),
        })
    }

    pub fn phase(&self) -> EnginePhase {
        self.phase
    }

    pub fn set_phase(&mut self, phase: EnginePhase) {
        // Pending adjustments simply wait: commit_idle enforces idleness.
        self.phase = phase;
    }

    /// Drop a queued adjustment without applying or logging it.
    pub fn cancel_pending(&mut self) {
        self.pending = None;
    }

    pub fn current_bytes(&self) -> u64 {
        self.current_bytes
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn log(&self) -> &[BudgetAdjustment] {
        &self.log
    }

    pub fn pending(&self) -> Option<&PendingAdjustment> {
        self.pending.as_ref()
    }

    /// Queue a grow/shrink toward `target_bytes`. Legal in any phase; the
    /// commit gate enforces idleness (request-early, apply-at-idle pattern).
    pub fn request(&mut self, target_bytes: u64, reason: impl Into<String>) -> Result<()> {
        let target = target_bytes.clamp(self.policy.min_vram_bytes, self.policy.max_vram_bytes);
        let kind = if target >= self.current_bytes {
            AdjustmentKind::Grow
        } else {
            AdjustmentKind::Shrink
        };
        if target == self.current_bytes {
            return Ok(());
        }
        if self.pending.is_some() {
            return Err(HarError::Invalid {
                kind: "elastic budget",
                message: "an adjustment is already pending".into(),
            });
        }
        self.pending = Some(PendingAdjustment {
            kind,
            target_bytes: target,
            reason: reason.into(),
        });
        Ok(())
    }

    /// Validate-before-teardown commit. `measured_free_vram_bytes` is the live
    /// reading from the device. A grow commits only if every rung of its
    /// rollback ladder still respects headroom; otherwise it walks down by
    /// `step_bytes` until something fits or the request is refused.
    pub fn commit_idle(&mut self, measured_free_vram_bytes: u64) -> Result<BudgetAdjustment> {
        if self.phase != EnginePhase::Idle {
            return Err(HarError::Invalid {
                kind: "elastic budget",
                message: format!("adjustments are idle-only, phase is {:?}", self.phase),
            });
        }
        let pending = self.pending.take().ok_or_else(|| HarError::Invalid {
            kind: "elastic budget",
            message: "no pending adjustment".into(),
        })?;
        let mut rollback_rungs = 0u32;
        let committed = match pending.kind {
            AdjustmentKind::Shrink => pending.target_bytes,
            AdjustmentKind::Grow => {
                let usable = measured_free_vram_bytes.saturating_sub(self.policy.headroom_bytes);
                let mut candidate = pending.target_bytes;
                loop {
                    let delta = candidate - self.current_bytes;
                    if delta <= usable || candidate == self.current_bytes {
                        break candidate;
                    }
                    candidate = candidate
                        .saturating_sub(self.policy.step_bytes)
                        .max(self.current_bytes);
                    rollback_rungs += 1;
                }
            }
        };
        if committed == self.current_bytes {
            return Err(HarError::Invalid {
                kind: "elastic budget",
                message: "grow refused: no rung satisfied headroom constraint".into(),
            });
        }
        self.current_bytes = committed;
        self.epoch += 1;
        let record = BudgetAdjustment {
            epoch: self.epoch,
            kind: pending.kind,
            requested_bytes: pending.target_bytes,
            committed_bytes: committed,
            rollback_rungs,
            reason: pending.reason,
        };
        self.log.push(record.clone());
        Ok(record)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> ElasticBudgetPolicy {
        ElasticBudgetPolicy {
            min_vram_bytes: 1 << 30,
            max_vram_bytes: 16 << 30,
            step_bytes: 512 << 20,
            headroom_bytes: 1 << 30,
        }
    }

    #[test]
    fn adjust_is_idle_only() {
        let mut b = ElasticVramBudget::new(policy(), 8 << 30).unwrap();
        b.request(10 << 30, "kv pressure").unwrap();
        b.set_phase(EnginePhase::Decode);
        assert!(b.commit_idle(u64::MAX).is_err());
        b.set_phase(EnginePhase::Idle);
        assert_eq!(b.commit_idle(u64::MAX).unwrap().committed_bytes, 10 << 30);
    }

    #[test]
    fn pending_adjustment_survives_phase_change_but_waits_for_idle() {
        let mut b = ElasticVramBudget::new(policy(), 4 << 30).unwrap();
        b.request(6 << 30, "prefetch").unwrap();
        b.set_phase(EnginePhase::Prefill);
        assert!(b.pending().is_some());
        assert!(b.commit_idle(u64::MAX).is_err());
        b.cancel_pending();
        assert!(b.pending().is_none());
        b.set_phase(EnginePhase::Idle);
        // Nothing was applied or logged by the cancelled request.
        assert_eq!(b.current_bytes(), 4 << 30);
        assert!(b.log().is_empty());
    }

    #[test]
    fn targets_are_clamped_to_policy_window() {
        let mut b = ElasticVramBudget::new(policy(), 8 << 30).unwrap();
        b.request(64 << 30, "over").unwrap(); // clamps to max => grow to 16GiB
        assert_eq!(b.commit_idle(u64::MAX).unwrap().committed_bytes, 16 << 30);
        b.request(0, "under").unwrap(); // clamps to min => shrink to 1GiB
        assert_eq!(b.commit_idle(u64::MAX).unwrap().committed_bytes, 1 << 30);
    }

    #[test]
    fn grow_rolls_back_down_the_ladder_until_headroom_fits() {
        let mut b = ElasticVramBudget::new(policy(), 8 << 30).unwrap();
        b.request(12 << 30, "optimistic").unwrap();
        // Only 2 GiB usable after headroom: rungs 12->11.5->11->10.5 fail, 10 GiB fits.
        let adj = b.commit_idle(3 << 30).unwrap();
        assert_eq!(adj.committed_bytes, 10 << 30);
        assert_eq!(adj.rollback_rungs, 4);
        assert_eq!(adj.requested_bytes, 12 << 30);
    }

    #[test]
    fn grow_is_refused_when_no_rung_fits() {
        let mut b = ElasticVramBudget::new(policy(), 8 << 30).unwrap();
        b.request(12 << 30, "impossible").unwrap();
        let err = b.commit_idle(0).unwrap_err();
        assert!(format!("{err}").contains("refused"));
        assert_eq!(b.current_bytes(), 8 << 30);
        assert!(b.pending().is_none());
        assert!(b.log().is_empty());
    }

    #[test]
    fn shrink_needs_no_free_memory() {
        let mut b = ElasticVramBudget::new(policy(), 8 << 30).unwrap();
        b.request(2 << 30, "release expert arena").unwrap();
        let adj = b.commit_idle(0).unwrap();
        assert_eq!(adj.committed_bytes, 2 << 30);
        assert_eq!(adj.rollback_rungs, 0);
    }

    #[test]
    fn equal_target_is_a_noop_without_epoch_bump() {
        let mut b = ElasticVramBudget::new(policy(), 8 << 30).unwrap();
        assert!(b.request(8 << 30, "same").is_ok());
        assert!(b.pending().is_none());
        assert!(b.commit_idle(u64::MAX).is_err());
        assert_eq!(b.epoch(), 0);
    }

    #[test]
    fn audit_log_records_epochs_in_order() {
        let mut b = ElasticVramBudget::new(policy(), 4 << 30).unwrap();
        b.request(6 << 30, "a").unwrap();
        b.commit_idle(u64::MAX).unwrap();
        b.request(5 << 30, "b").unwrap();
        b.commit_idle(u64::MAX).unwrap();
        let log = b.log();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].epoch, 1);
        assert_eq!(log[1].epoch, 2);
        assert_eq!(b.epoch(), 2);
    }
}
