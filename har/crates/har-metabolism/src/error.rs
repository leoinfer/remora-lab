//! Metabolism errors: all control-flow failures are typed and fail-closed.

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum MetabolismError {
    #[error("unknown required input: {0}")]
    UnknownInput(&'static str),
    #[error("reserve insufficient: {0}")]
    ReserveInsufficient(String),
    #[error("reserve debt exceeded: {0}")]
    ReserveDebt(String),
    #[error("fail closed: {0}")]
    FailClosed(&'static str),
    #[error("double credit attempted: {0}")]
    DoubleCredit(String),
    #[error("invalid artifact state: {0}")]
    InvalidArtifact(String),
    #[error("invariant violation: {0}")]
    Invariant(String),
    #[error("trace mismatch: {0}")]
    TraceMismatch(String),
}

pub type MetabolismResult<T> = Result<T, MetabolismError>;
