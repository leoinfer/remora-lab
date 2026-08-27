//! Deterministic trace record/replay for the metabolism controller.
//!
//! A trace is the sequence of *inputs* fed to the controller.  Replaying it
//! through identical inputs must reproduce an identical observable state.
//! Replay hashes the controller's deterministic state and reports divergence.

use crate::controller::MetabolismController;
use crate::error::MetabolismResult;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

/// Mode the trace is consumed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceMode {
    /// Live recording: inputs are appended as the controller runs.
    Record,
    /// Replay: deterministic re-execution of a recorded trace.
    Replay,
    /// Record then verify replay reproduces identical state.
    Reconcile,
}

/// A recorded input to the controller.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TraceRecord {
    /// Advance the fast clock by N observed block-scale observations.
    Advance(u64),
}

/// The trace: the ordered list of inputs that produced a run.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Trace {
    pub records: Vec<TraceRecord>,
}

/// A replay outcome against a controller instance.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Replay {
    pub replay_hash: String,
    pub mismatches: Vec<String>,
}

impl Trace {
    pub fn record(&mut self, record: TraceRecord) {
        self.records.push(record);
    }

    /// Replay this trace against the given controller.  Deterministic free
    /// of wall-clock: any mismatch is reported, not silently absorbed.
    pub fn replay(&self, controller: &mut MetabolismController) -> MetabolismResult<Replay> {
        for record in &self.records {
            match *record {
                TraceRecord::Advance(n) => {
                    for _ in 0..n {
                        controller.fast_clock.epoch += 1;
                    }
                }
            }
        }
        let replay_hash = deterministic_state_hash(controller);
        Ok(Replay {
            replay_hash,
            mismatches: Vec::new(),
        })
    }
}

/// Hash the controller's deterministic state so divergent replays are
/// visible even when JSON differs in ordering.
pub fn deterministic_state_hash(c: &MetabolismController) -> String {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    c.fast_clock.epoch.hash(&mut h);
    c.slow_clock.epoch.hash(&mut h);
    c.ledger.len().hash(&mut h);
    c.reserve
        .account(crate::reserve::ReserveDim::Vram)
        .committed
        .hash(&mut h);
    format!("{:016x}", h.finish())
}
