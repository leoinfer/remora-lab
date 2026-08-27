//! Integration invariants for the REMORA metabolism controller.
//!
//! These tests encode the hard design rules:
//! - unknown critical inputs fail closed;
//! - reuse and overlap credit require evidence and are never granted twice;
//! - repeated decisions are deterministic;
//! - reserve mobilization is bounded by physical capacity; and
//! - energy remains UNKNOWN without a GPU-only source.

use har_metabolism::artifact::{ReuseClass, SalvageInput};
use har_metabolism::common::Estimate;
use har_metabolism::controller::{ControllerConfig, MetabolismController};
use har_metabolism::energy::EnergyScope;
use har_metabolism::ledger::{CreditEvidence, LedgerClass};
use har_metabolism::portion::{PortionDecision, PortionInput};
use har_metabolism::reserve::{ReserveDim, ReserveTable};
use har_metabolism::salvage::{Salvage, SalvageCandidate, SalvageDecision};
use har_metabolism::trace::{Trace, TraceMode, TraceRecord};

fn controller() -> MetabolismController {
    MetabolismController::new(ControllerConfig::default())
}

#[test]
fn unknown_salvage_input_fails_closed() {
    let s = Salvage::new();
    let mut candidate = SalvageCandidate {
        input: SalvageInput::with_none(),
        base_seq: 1,
        bucket: 0,
    };
    candidate.input.expected_reuse_cost = Estimate::measured(1000);
    assert_eq!(s.score(&candidate), SalvageDecision::FailClosed);
}

#[test]
fn portion_rejects_unknown_identity() {
    let mut c = controller();
    let input = PortionInput {
        artifact_id: "",
        expected_tokens: 5,
        transfer_cost_bytes: 0,
        compute_cost_ms: 0,
    };
    assert!(c.decide_portion(input, ReuseClass::ExactReusable).is_err());
}

#[test]
fn portion_rejects_unrecoverable_artifact() {
    let mut c = controller();
    let input = PortionInput {
        artifact_id: "x",
        expected_tokens: 5,
        transfer_cost_bytes: 0,
        compute_cost_ms: 0,
    };
    assert!(c.decide_portion(input, ReuseClass::Unrecoverable).is_err());
}

#[test]
fn reuse_credit_requires_prior_reclaim() {
    let mut c = controller();
    let evidence = CreditEvidence {
        witness_identity: "t1".into(),
        redeems_tokens: 10,
        bytes_moved: 0,
        replay_gen: 1,
    };
    assert!(c.grant_reuse_credit(&evidence).is_err());
}

#[test]
fn double_reuse_credit_is_refused() {
    let mut c = controller();
    assert!(c.record_reclaim("t1", 0).is_ok());
    let first = CreditEvidence {
        witness_identity: "t1".into(),
        redeems_tokens: 10,
        bytes_moved: 0,
        replay_gen: 1,
    };
    assert!(c.grant_reuse_credit(&first).is_ok());
    let second = CreditEvidence {
        witness_identity: "t1".into(),
        redeems_tokens: 20,
        bytes_moved: 0,
        replay_gen: 2,
    };
    assert!(c.grant_reuse_credit(&second).is_err());
}

#[test]
fn overlap_credit_requires_measured_bytes() {
    let mut c = controller();
    assert!(c.grant_overlap_credit("inline".into(), 4096).is_ok());
    assert!(c.grant_overlap_credit("assumed".into(), 0).is_err());
}

#[test]
fn mobilized_debt_never_exceeds_physical_headroom() {
    let mut table = ReserveTable::new(&[(ReserveDim::Vram, 100, 10)]);
    assert!(table.commit(ReserveDim::Vram, 60));
    assert!(table.mobilize(ReserveDim::Vram, 40, 0));
    let account = table.account(ReserveDim::Vram);
    assert_eq!(account.committed, 100);
    assert!(account.debt <= 10);
    assert!(!table.mobilize(ReserveDim::Vram, 1, 1));
}

#[test]
fn energy_default_is_unknown() {
    let c = controller();
    assert_eq!(c.energy.scope, EnergyScope::Unknown);
    assert!(c.energy.joules_per_exact_token.is_none());
}

#[test]
fn portion_decision_is_deterministic() {
    let mut a = controller();
    let mut b = controller();
    let input = PortionInput {
        artifact_id: "e1",
        expected_tokens: 100,
        transfer_cost_bytes: 0,
        compute_cost_ms: 0,
    };
    let first = a.decide_portion(input, ReuseClass::ExactReusable).unwrap();
    let second = b.decide_portion(input, ReuseClass::ExactReusable).unwrap();
    assert_eq!(first, second);
    assert!(matches!(first, PortionDecision::Admit { .. }));
}

#[test]
fn ledger_conservation_holds() {
    let mut c = controller();
    c.record_spent(LedgerClass::AuthoritativeUseful, "t1".into(), 100, 10, 1)
        .unwrap();
    let totals = &c.ledger.totals;
    assert_eq!(totals.useful_tokens, 10);
    assert!(totals.reuse_credit_ms <= c.ledger.reuse_credit_given.len() as u64 * 1000);
}

#[test]
fn trace_replays_identically() {
    let _controller = controller();
    let mut trace = Trace::default();
    trace.record(TraceRecord::Advance(3));
    let mut replay_target = controller();
    let replay = trace.replay(&mut replay_target).unwrap();
    assert!(replay.mismatches.is_empty());
    let _ = TraceMode::Replay;
}
