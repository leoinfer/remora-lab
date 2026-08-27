//! Shared primitives for the metabolism subsystem.
//!
//! Everything here is a value object with deterministic behavior.  No wall
//! clock appears in any decision; time is represented by explicit tick/epoch
//! numbers in the trace so replay is reproducible.

use serde::{Deserialize, Serialize};

/// MiB unit alias.  All reserve/memory quantities are integer MiB or bytes.
pub type MiB = u64;
/// Milliseconds unit alias (compute/waste/credit time).
pub type Ms = u64;
/// Nanosecond unit alias (transfer/engineering time).
pub type Ns = u64;
/// Exact tokens (committed).
pub type Tokens = u64;

/// Deterministic value classification.  `Unknown` is never treated as zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Certainty {
    Measured,
    Estimated,
    Unknown,
}

/// An estimate with an identity.  Conservation rules require an estimate to
/// be either measured, or estimated with an explicit calibration boundary,
/// or unknown.  Unknown is not zero.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Estimate {
    pub value: i64,
    pub certainty: Certainty,
}

impl Estimate {
    pub const UNKNOWN_ZERO: Estimate = Estimate {
        value: 0,
        certainty: Certainty::Unknown,
    };

    pub fn measured(value: i64) -> Self {
        Self {
            value,
            certainty: Certainty::Measured,
        }
    }
    pub fn estimated(value: i64) -> Self {
        Self {
            value,
            certainty: Certainty::Estimated,
        }
    }
    pub fn known(&self) -> bool {
        self.certainty != Certainty::Unknown
    }
}

/// Probability of reuse, in thousandths (0..=1000).  Integer form keeps the
/// ranking deterministic and JSON-friendly.
pub type Permille = u32;

/// Shared deterministic clock ticks.  `fast` is advanced on every block-scale
/// observation; `slow` is advanced by the slow clock policy only after
/// `fast` has crossed a window boundary or identity changed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockTicks {
    pub fast: u64,
    pub slow: u64,
}

pub const MIB: MiB = 1;
pub const BYTES_PER_MIB: u64 = 1 << 20;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_is_not_zero_for_certainty() {
        let e = Estimate::UNKNOWN_ZERO;
        assert_eq!(e.value, 0);
        assert_eq!(e.certainty, Certainty::Unknown);
        assert!(!e.known());
        assert!(Estimate::measured(3).known());
    }
}
