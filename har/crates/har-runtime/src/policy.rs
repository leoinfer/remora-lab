//! Explicit Rust-only runtime execution policy.
//!
//! The production runtime has one policy: native Rust execution is required.
//! Missing kernels, stale state, and unknown capacity fail closed. Historical
//! reference implementations may remain in offline artifacts, but they are
//! not representable as a production runtime policy.

use har_core::{BackendKind, FallbackContract, HarError, Result};
use serde::{Deserialize, Serialize};

pub const RUNTIME_POLICY_SCHEMA: &str = "har.runtime_policy.v1";

/// Explicit runtime execution policy.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RuntimePolicy {
    #[default]
    NativeRequired,
}

impl RuntimePolicy {
    pub fn schema(&self) -> &'static str {
        RUNTIME_POLICY_SCHEMA
    }

    /// Production Rust execution never permits a foreign fallback.
    pub fn permits_reference_fallback(&self) -> bool {
        false
    }

    /// NATIVE_REQUIRED hard invariants.
    pub fn required_fallback_count(&self) -> u64 {
        0
    }

    pub fn required_reference_adapter_invocations(&self) -> u64 {
        0
    }

    /// Fail-closed admission check: the observed counters must satisfy the
    /// policy.  In NATIVE_REQUIRED any fallback or reference invocation is a
    /// hard error; in NATIVE_OPTIONAL both are allowed but must be recorded.
    pub fn admit(&self, fallback_count: u64, reference_adapter_invocations: u64) -> Result<()> {
        if fallback_count != 0 {
            return Err(HarError::Invalid {
                kind: "runtime policy",
                message: format!(
                    "NATIVE_REQUIRED violated: fallback_count={fallback_count} (must be 0)"
                ),
            });
        }
        if reference_adapter_invocations != 0 {
            return Err(HarError::Invalid {
                kind: "runtime policy",
                message: format!(
                    "NATIVE_REQUIRED violated: reference_adapter_invocations={reference_adapter_invocations} (must be 0)"
                ),
            });
        }
        Ok(())
    }

    /// The fallback contract the runtime must present for this policy.
    /// NATIVE_REQUIRED turns every reference path into a hard rejection.
    pub fn fallback_contract(&self) -> FallbackContract {
        FallbackContract {
            authority_backend: BackendKind::Vulkan,
            on_unknown_capacity: "reject_plan".into(),
            on_stale_generation: "fail_closed".into(),
            on_kernel_unavailable: "reject_plan".into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_is_versioned() {
        assert_eq!(
            RuntimePolicy::NativeRequired.schema(),
            "har.runtime_policy.v1"
        );
    }

    #[test]
    fn native_required_rejects_any_fallback() {
        let policy = RuntimePolicy::NativeRequired;
        assert!(policy.admit(0, 0).is_ok());
        assert!(policy.admit(1, 0).is_err());
        assert!(policy.admit(0, 1).is_err());
        assert!(policy.admit(1, 1).is_err());
    }

    #[test]
    fn rust_only_is_the_default_and_rejects_foreign_execution() {
        let policy = RuntimePolicy::default();
        assert_eq!(policy, RuntimePolicy::NativeRequired);
        assert!(!policy.permits_reference_fallback());
        assert!(policy.admit(1, 0).is_err());
    }

    #[test]
    fn native_required_fallback_contract_is_fail_closed() {
        let contract = RuntimePolicy::NativeRequired.fallback_contract();
        assert_eq!(contract.on_kernel_unavailable, "reject_plan");
        assert_eq!(contract.on_stale_generation, "fail_closed");
        assert_eq!(contract.on_unknown_capacity, "reject_plan");
    }

    #[test]
    fn no_silent_downgrade_from_native_required() {
        // The policy object itself is the guard: a NATIVE_REQUIRED runtime
        // that observes a fallback returns Err rather than a result with a
        // downgraded verdict.
        let policy = RuntimePolicy::NativeRequired;
        let error = policy.admit(1, 0).expect_err("must reject");
        assert!(error.to_string().contains("NATIVE_REQUIRED violated"));
    }
}
