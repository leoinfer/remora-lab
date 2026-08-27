//! Speculative decode (draft + verify) — the "cheaper speculation" layer.
//!
//! [`SpeculativeModel`] wraps a cheap draft model and the target model
//! into one [`BatchStepModel`], so the scheduler and the prefix graph are
//! completely unaware of speculation: the wrapper packs the draft's
//! hidden state and the target's hidden state into one hidden vector
//! (the scheduler treats it opaquely) and advances both along the *true*
//! trajectory, so resumed/graph states stay consistent by construction.
//!
//! Per step (one input token):
//!   1. draft up to `block` tokens with the draft model, stopping early
//!      when a draft's probability < `p_min` — a state-aware horizon;
//!   2. verify the whole draft block in ONE target pass
//!      (`prefill_batch` — the GEMM path; per-token logits);
//!   3. accept the longest matching prefix; the next token is the target's
//!      prediction at the first rejection (or the bonus token after a
//!      full-block acceptance).
//!
//! Cost accounting: the draft is small relative to the target, so per
//! accepted token the bytes moved are
//! `(block × draft_bytes + target_bytes) / acceptance` instead of
//! `target_bytes`.

use crate::adapter::{BatchStepModel, Hidden, Logits, StepOutcome};
use crate::scheduler::argmax;
use std::cell::{Cell, RefCell};

/// Speculation configuration for tiered horizon selection.
///
/// Policy: high-confidence continuation →
/// speculate aggressively (up to `block`); medium → shorten to
/// `med_cap`; uncertain → 1..=`min_cap`; below `p_min` → stop drafting
/// entirely (draft gating).
#[derive(Clone, Copy, Debug)]
pub struct SpecConfig {
    /// Absolute draft cap per verify pass.
    pub block: usize,
    /// Draft probability ≥ this → aggressive tier.
    pub p_high: f32,
    /// Draft probability ≥ this → medium tier (cap `med_cap`).
    pub p_med: f32,
    /// Below this, drafting stops (draft gating).
    pub p_min: f32,
    /// Medium-tier draft cap.
    pub med_cap: usize,
    /// Uncertain-tier draft cap.
    pub min_cap: usize,
}

impl Default for SpecConfig {
    fn default() -> Self {
        Self {
            block: 8,
            p_high: 0.95,
            p_med: 0.85,
            p_min: 0.75,
            med_cap: 4,
            min_cap: 2,
        }
    }
}

/// Tier cap for a draft probability under the tiered horizon.
pub fn tier_cap(cfg: &SpecConfig, p: f32) -> usize {
    if p >= cfg.p_high {
        cfg.block
    } else if p >= cfg.p_med {
        cfg.med_cap
    } else if p >= cfg.p_min {
        cfg.min_cap
    } else {
        0
    }
}

/// Speculation telemetry: rows moved per model and acceptance.
///
/// Speculation's critical metrics: acceptance rate and tokens generated per
/// expensive verification (`acceptance_length`).  Counters are `Cell`s so
/// the wrapper stays `&self`-compatible with [`BatchStepModel`] with no
/// unsafe code.
#[derive(Clone, Debug, Default)]
pub struct SpecTelemetry {
    /// Draft model invocations (one per drafted token).
    draft_rows: Cell<u64>,
    /// Target verify passes (one per step, regardless of block size).
    target_rows: Cell<u64>,
    /// Tokens drafted.
    drafted: Cell<u64>,
    /// Draft tokens accepted by the target.
    accepted: Cell<u64>,
    /// Sequences' steps through the wrapper.
    steps: Cell<u64>,
    /// Accepted draft tokens by position index (0 = first drafted token).
    /// The acceptance curve used to make horizons cliff-aware rather than
    /// fixed-max.
    accepted_by_position: RefCell<Vec<u64>>,
    /// Drafted tokens by position index.
    drafted_by_position: RefCell<Vec<u64>>,
}

/// Max tracked horizon for the per-position curve.
pub const MAX_TRACKED_HORIZON: usize = 16;

/// Elastic-horizon score:
/// `score(d) = d * predicted_acceptance / incremental_cost`.  Positive EV
/// requires score >= 1. This is an executable cost model for live telemetry.
pub fn op01_score(d: usize, p_accept: f64, draft_cost: f64, verify_cost: f64) -> f64 {
    let expected = d as f64 * p_accept;
    let cost = draft_cost * d as f64 + verify_cost;
    expected / cost
}

/// Survival-sum expected acceptance for horizon `d` over a per-position
/// acceptance curve: E[A] = sum_k prod_{i<=k} p_i (the calibrated
/// break-even model's definition).
pub fn expected_accepted(curve: &[f64], d: usize) -> f64 {
    let mut total = 0.0;
    let mut survival = 1.0;
    for k in 1..=d {
        let p = curve.get(k - 1).copied().unwrap_or(0.0);
        survival *= p;
        total += survival;
    }
    total
}

/// Speedup of horizon `d` under the calibrated break-even model (the
/// preregistered `mtp_break_even_sweep` equation):
///
/// ```text
/// T_batch(d) = d*t_draft + t_single*(1 + alpha*E[A](d)) + t_overhead
/// speedup    = (1 + E[A](d)) * t_single / T_batch(d)
/// ```
///
/// The caller supplies all cost parameters; the values used by
/// [`optimal_horizon`] are illustrative defaults.
pub fn horizon_speedup(
    curve: &[f64],
    d: usize,
    t_draft: f64,
    t_single: f64,
    alpha: f64,
    t_overhead: f64,
) -> f64 {
    let ea = expected_accepted(curve, d);
    let t_batch = d as f64 * t_draft + t_single * (1.0 + alpha * ea) + t_overhead;
    (1.0 + ea) * t_single / t_batch
}

/// Value-optimal horizon under the illustrative cost model.
pub fn optimal_horizon(curve: &[f64], max_d: usize) -> usize {
    let t_draft = 4.0;
    let t_single = 32.0;
    let alpha = 0.35;
    let t_overhead = 1.5;
    let mut best_d = 1usize;
    let mut best = f64::NEG_INFINITY;
    for d in 1..=max_d {
        let s = horizon_speedup(curve, d, t_draft, t_single, alpha, t_overhead);
        if s > best {
            best = s;
            best_d = d;
        }
    }
    best_d
}

/// Tier calibration from live acceptance telemetry: the
/// aggressive tier (block) is set at the acceptance cliff — the
/// value-optimal horizon from the supplied curve — and the lower tiers
/// scale down from it.  This is the runtime form of "stop at the first
/// non-positive horizon marginal, calibrated from data instead
/// of assumed curves.
pub fn calibrate_tiers(curve: &[f64], base: &SpecConfig, floor: f64) -> SpecConfig {
    let cliff = cliff_horizon(curve, floor).max(1);
    let mut cfg = *base;
    cfg.block = cliff;
    cfg.med_cap = (cliff / 2).max(1);
    cfg.min_cap = 1;
    cfg
}

/// Cliff-aware horizon: the largest position whose acceptance stays at
/// or above `floor`; the first position below the floor ends the run.
///
/// The horizon is derived from the supplied acceptance curve instead of a
/// fixed block. The caller owns the provenance and calibration of that curve.
pub fn cliff_horizon(curve: &[f64], floor: f64) -> usize {
    let mut best = 0usize;
    for (i, &p) in curve.iter().enumerate() {
        if p >= floor {
            best = i + 1;
        } else {
            break;
        }
    }
    best
}

impl SpecTelemetry {
    pub fn draft_rows(&self) -> u64 {
        self.draft_rows.get()
    }
    pub fn target_rows(&self) -> u64 {
        self.target_rows.get()
    }
    pub fn drafted(&self) -> u64 {
        self.drafted.get()
    }
    pub fn accepted(&self) -> u64 {
        self.accepted.get()
    }
    pub fn steps(&self) -> u64 {
        self.steps.get()
    }
    pub fn accepted_by_position(&self) -> Vec<u64> {
        self.accepted_by_position.borrow().clone()
    }
    pub fn drafted_by_position(&self) -> Vec<u64> {
        self.drafted_by_position.borrow().clone()
    }

    /// Per-position acceptance curve (position 0 = first drafted token).
    pub fn acceptance_curve(&self) -> Vec<f64> {
        let a = self.accepted_by_position.borrow();
        let d = self.drafted_by_position.borrow();
        a.iter()
            .zip(d.iter())
            .map(|(ac, dr)| {
                if *dr == 0 {
                    0.0
                } else {
                    *ac as f64 / *dr as f64
                }
            })
            .collect()
    }

    /// Cliff-aware horizon: the largest position whose acceptance stays
    /// at or above `floor` — see [`cliff_horizon`].
    pub fn cliff_horizon(&self, floor: f64) -> usize {
        cliff_horizon(&self.acceptance_curve(), floor)
    }

    /// Reset all counters (used after an auto-calibration probe so the
    /// reported telemetry covers the calibrated phase only).
    pub fn reset(&self) {
        self.draft_rows.set(0);
        self.target_rows.set(0);
        self.drafted.set(0);
        self.accepted.set(0);
        self.steps.set(0);
        self.accepted_by_position.borrow_mut().clear();
        self.drafted_by_position.borrow_mut().clear();
    }

    /// Accepted tokens (including the bonus/next token) per target pass —
    /// Tokens per expensive verification.
    pub fn acceptance_length(&self) -> f64 {
        let rows = self.target_rows.get();
        if rows == 0 {
            return 0.0;
        }
        (self.accepted.get() as f64 + rows as f64) / rows as f64
    }

    /// Effective weight bytes per accepted token for given per-row costs:
    /// the draft streams every drafted token, the target once per block.
    pub fn bytes_per_accepted(&self, draft_bytes_per_row: f64, target_bytes_per_row: f64) -> f64 {
        let accepted_total = (self.accepted.get() + self.target_rows.get()).max(1) as f64;
        (self.draft_rows.get() as f64 * draft_bytes_per_row
            + self.target_rows.get() as f64 * target_bytes_per_row)
            / accepted_total
    }

    /// Throughput multiplier vs target-only decode at the given row costs.
    pub fn speedup_vs_target_only(
        &self,
        draft_bytes_per_row: f64,
        target_bytes_per_row: f64,
    ) -> f64 {
        if self.bytes_per_accepted(draft_bytes_per_row, target_bytes_per_row) == 0.0 {
            return 0.0;
        }
        target_bytes_per_row / self.bytes_per_accepted(draft_bytes_per_row, target_bytes_per_row)
    }
}

/// Draft + target packed into one BatchStepModel.
///
/// The returned hidden vector is `[target_hidden | draft_hidden |
/// marker]` where `marker` is the target width as an f32 (exact for
/// widths < 2^24 — the reference dense model at 8k context is ~1.6M).
/// The scheduler stores and passes it opaquely; the split is read from
/// the marker each call, so EITHER half (or both) may be variable-width
/// — the dense transformer's hidden grows with its KV history, and
/// self-speculation makes the draft grow identically.  Every returned
/// state is the true state at the position of the next input token, so
/// prefix-graph resume and mid-chunk reuse stay exact
/// (differential-tested).
pub struct SpeculativeModel<D: BatchStepModel, T: BatchStepModel> {
    draft: D,
    target: T,
    cfg: SpecConfig,
    telemetry: SpecTelemetry,
}

impl<D: BatchStepModel, T: BatchStepModel> SpeculativeModel<D, T> {
    pub fn new(draft: D, target: T, cfg: SpecConfig) -> Self {
        Self {
            draft,
            target,
            cfg,
            telemetry: SpecTelemetry::default(),
        }
    }

    pub fn telemetry(&self) -> SpecTelemetry {
        self.telemetry.clone()
    }
    pub fn cfg(&self) -> SpecConfig {
        self.cfg
    }
    /// Hot-swap the speculation policy (auto-calibration path).
    pub fn set_config(&mut self, cfg: SpecConfig) {
        self.cfg = cfg;
    }

    /// Runtime calibration: measure the live acceptance curve,
    /// derive the calibrated tiered horizon on top of `base` (the
    /// user-threshold policy — its p_high/p_med/p_min thresholds are kept,
    /// its caps are replaced by the curve-derived ones), and apply it.
    /// Returns the calibrated config and the observed curve; `None` when
    /// the probe produced too little data (fewer than 2 tracked
    /// positions, or an all-zero curve — e.g. an adversarial draft).  The
    /// probe telemetry is reset so later reports cover the calibrated
    /// phase only.
    pub fn calibrate(&mut self, base: SpecConfig, floor: f64) -> Option<(SpecConfig, Vec<f64>)> {
        let curve = self.telemetry.acceptance_curve();
        if curve.len() < 2 || curve.iter().all(|&p| p <= 0.0) {
            return None;
        }
        let calibrated = calibrate_tiers(&curve, &base, floor);
        self.set_config(calibrated);
        self.telemetry.reset();
        Some((calibrated, curve))
    }

    pub fn draft_rows(&self) -> u64 {
        self.telemetry.draft_rows.get()
    }
    pub fn target_rows(&self) -> u64 {
        self.telemetry.target_rows.get()
    }
    pub fn drafted(&self) -> u64 {
        self.telemetry.drafted.get()
    }
    pub fn accepted(&self) -> u64 {
        self.telemetry.accepted.get()
    }

    fn pack(&self, t: Hidden, d: Hidden) -> Hidden {
        let mut h = Vec::with_capacity(t.len() + d.len() + 1);
        h.extend(t.iter().copied());
        h.extend(d.iter().copied());
        h.push(t.len() as f32);
        h
    }

    fn unpack(&self, h: &Hidden) -> (Hidden, Hidden) {
        let marker = *h.last().expect("packed hidden carries the width marker");
        let t_len = marker as usize;
        assert!(
            t_len < h.len(),
            "corrupt width marker: {marker} in a {}-long hidden",
            h.len()
        );
        (h[..t_len].to_vec(), h[t_len..h.len() - 1].to_vec())
    }

    /// Softmax probability of `token` under `logits`.
    fn probability(logits: &[f32], token: u32) -> f32 {
        let max = logits.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
        let mut sum = 0.0f32;
        for v in logits {
            sum += (v - max).exp();
        }
        (logits[token as usize] - max).exp() / sum
    }

    /// One speculative step for one sequence.  `t_h`/`d_h` are the target
    /// and draft hidden states at the input token's position.
    ///
    /// The returned [`StepOutcome`] is the scheduler contract: the
    /// accepted drafts enter the sequence's stream (each recorded in the
    /// prefix graph with the packed state at its position), then the
    /// token sampled from `logits` (state `next`).
    fn spec_step(&self, t_h: &Hidden, d_h: &Hidden, token: u32) -> StepOutcome {
        let eos = self.target.eos();

        // 1. Draft loop with the tiered horizon: the tier cap is
        //    re-evaluated at every drafted position from the draft's own
        //    confidence (its uncertainty topology), so a confident start
        //    can extend and a wobble shortens the block.
        //
        //    `d_traj[j]` records the draft state after consuming
        //    [token, c1..cj] — the states at every position of the
        //    accepted prefix, which the prefix graph needs for exact
        //    mid-step reuse.  The trajectory is captured for free here
        //    (the loop computes each state exactly once).
        let mut drafts: Vec<u32> = Vec::with_capacity(self.cfg.block);
        let mut d_traj: Vec<Hidden> = Vec::with_capacity(self.cfg.block + 1);
        let mut d = d_h.clone();
        let mut next = token;
        for _ in 0..self.cfg.block {
            let out = self.draft.batch_step(&[(d.clone(), next)]);
            let out = &out[0];
            d_traj.push(out.next.clone());
            let cand = argmax(&out.logits);
            let p = Self::probability(&out.logits, cand);
            let cap = tier_cap(&self.cfg, p);
            if cap == 0 || drafts.len() >= cap {
                break;
            }
            drafts.push(cand);
            d = out.next.clone();
            next = cand;
            if cand == eos {
                break;
            }
        }
        // The loop consumed [token, c1..c_k] (k = drafts.len()) and its
        // iterations recorded d_traj[0..k] (the k-th iteration's state,
        // after the last candidate, is recorded even on a tier break).
        // A normal exit stops one state short: the position after the
        // final draft (the old "state lag" fix — the draft state must
        // match the target's position after the full verify, or
        // acceptance collapses on the next step; caught by the
        // perfect-draft curve test).
        if d_traj.len() == drafts.len() {
            let last = *drafts.last().expect("non-empty drafts");
            let out = self
                .draft
                .batch_step(&[(d_traj.last().expect("trajectory").clone(), last)]);
            d_traj.push(out[0].next.clone());
        }
        self.telemetry
            .draft_rows
            .set(self.telemetry.draft_rows.get() + drafts.len() as u64);
        self.telemetry
            .drafted
            .set(self.telemetry.drafted.get() + drafts.len() as u64);

        // 2. One target verify pass over [token | drafts]; per-token
        //    logits: logits[j] predicts the token after consuming
        //    tokens[0..=j].
        let mut verify_tokens = Vec::with_capacity(1 + drafts.len());
        verify_tokens.push(token);
        verify_tokens.extend(&drafts);
        let (t_states, t_logits) =
            self.target.prefill_batch(&[(t_h.clone(), verify_tokens)])[0].clone();
        self.telemetry
            .target_rows
            .set(self.telemetry.target_rows.get() + 1);
        self.telemetry.steps.set(self.telemetry.steps.get() + 1);

        // 3. Acceptance walk: draft[j] is at position j+1 of the verify.
        let mut accepted = 0usize;
        for (j, &dtok) in drafts.iter().enumerate() {
            if argmax(&t_logits[j]) == dtok {
                accepted += 1;
            } else {
                break;
            }
        }
        self.telemetry
            .accepted
            .set(self.telemetry.accepted.get() + accepted as u64);
        {
            let mut acc = self.telemetry.accepted_by_position.borrow_mut();
            let mut drf = self.telemetry.drafted_by_position.borrow_mut();
            for j in 0..drafts.len() {
                if j >= MAX_TRACKED_HORIZON {
                    break;
                }
                while drf.len() <= j {
                    drf.push(0);
                }
                while acc.len() <= j {
                    acc.push(0);
                }
                drf[j] += 1;
                if j < accepted {
                    acc[j] += 1;
                }
            }
        }

        // `d_traj` holds the draft state after every consumed token of
        // the verify prefix.  On full acceptance the position after the
        // last draft is exactly the position of the bonus token; on
        // rejection the draft state at the accepted prefix is
        // `d_traj[accepted]` — the corrected token arrives at that
        // position next step.  (The old `accepted == 0 → keep d` case
        // was the draft-state lag bug: the state one position behind
        // collapses acceptance — the trajectory fixes it for every
        // acceptance count.)
        let (t_next, next_logits) = if accepted == drafts.len() {
            // Full block accepted: bonus token from the last verify logits.
            (
                t_states.last().expect("verify states").clone(),
                t_logits.last().expect("verify logits").clone(),
            )
        } else {
            // Rejected at `accepted`: next token = target's prediction at
            // the rejection position; state = after the accepted prefix.
            (t_states[accepted].clone(), t_logits[accepted].clone())
        };

        // Pack the scheduler contract: the accepted drafts (with the
        // packed state at each draft's position), the state after the
        // input token, and the next state (at the sampled token).
        let mut outcome = StepOutcome {
            next: self.pack(t_next, d_traj[accepted].clone()),
            logits: next_logits,
            drafts: drafts[..accepted].to_vec(),
            draft_states: Vec::with_capacity(accepted),
            consumed_state: self.pack(t_states[0].clone(), d_traj[0].clone()),
        };
        for j in 1..=accepted {
            outcome
                .draft_states
                .push(self.pack(t_states[j].clone(), d_traj[j].clone()));
        }
        outcome
    }
}

impl<D: BatchStepModel, T: BatchStepModel> BatchStepModel for SpeculativeModel<D, T> {
    fn batch_step(&self, inputs: &[(Hidden, u32)]) -> Vec<StepOutcome> {
        inputs
            .iter()
            .map(|(h, t)| {
                let (t_h, d_h) = self.unpack(h);
                self.spec_step(&t_h, &d_h, *t)
            })
            .collect()
    }

    /// Prompt prefill never speculates — the tokens are known, so there
    /// is nothing to predict.  The target trajectory is exact (per-token
    /// states AND logits), and the draft state tracks the same tokens so
    /// the first decode step drafts from the correct position.  Per-token
    /// states are packed `[t_j | d_j]` like decode states.
    fn prefill_batch(&self, inputs: &[(Hidden, Vec<u32>)]) -> Vec<(Vec<Hidden>, Vec<Logits>)> {
        inputs
            .iter()
            .map(|(h, tokens)| {
                let (t_h, d_h) = self.unpack(h);
                let (t_states, t_logits) =
                    self.target.prefill_batch(&[(t_h, tokens.clone())])[0].clone();
                let mut d = d_h;
                let mut d_states = Vec::with_capacity(tokens.len());
                for &t in tokens {
                    let out = self.draft.batch_step(&[(d, t)]);
                    d = out[0].next.clone();
                    d_states.push(d.clone());
                }
                let states = t_states
                    .iter()
                    .zip(d_states.iter())
                    .map(|(ts, ds)| self.pack(ts.clone(), ds.clone()))
                    .collect();
                (states, t_logits)
            })
            .collect()
    }

    fn initial_hidden(&self) -> Hidden {
        self.pack(self.target.initial_hidden(), self.draft.initial_hidden())
    }

    fn eos(&self) -> u32 {
        self.target.eos()
    }

    fn weight_bytes_per_row(&self) -> u64 {
        // The dominant read is the target's weight set (one verify pass
        // per step); the draft's per-token reads are accounted in
        // telemetry, not the scheduler's bandwidth model.
        self.target.weight_bytes_per_row()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cliff_horizon_matches_synthetic_curve() {
        // A synthetic acceptance curve with a clear cliff:
        // 1.0 x5, 0.571, 0.0 — the selected horizon is 5 at floor 0.9.
        let curve = [1.0, 1.0, 1.0, 1.0, 1.0, 0.571, 0.0];
        assert_eq!(cliff_horizon(&curve, 0.9), 5);
        assert_eq!(cliff_horizon(&curve, 0.5), 6);
        assert_eq!(cliff_horizon(&curve, 0.99), 5, "curve max is 1.0 >= 0.99");
        assert_eq!(cliff_horizon(&curve, 1.01), 0, "floor above the curve");
        assert_eq!(cliff_horizon(&[], 0.5), 0, "empty curve");
        assert_eq!(cliff_horizon(&[1.0, 1.0], 1.0), 2);
    }

    #[test]
    fn perfect_draft_curve_is_flat_full() {
        let target = crate::toy::ToyModel::new(crate::toy::ToyConfig {
            dim: 64,
            vocab: 512,
            layers: 3,
            eos: 499,
            seed: 42,
        });
        let draft = target.clone();
        let spec = SpeculativeModel::new(
            draft,
            target,
            SpecConfig {
                block: 4,
                p_high: 0.0,
                p_med: 0.0,
                p_min: 0.0,
                med_cap: 1,
                min_cap: 1,
            },
        );
        let mut h = spec.initial_hidden();
        for t in 0..5u32 {
            let out = spec.batch_step(&[(h, t)]);
            h = out[0].next.clone();
        }
        let curve = spec.telemetry().acceptance_curve();
        assert!(!curve.is_empty());
        for &p in &curve {
            assert!((p - 1.0).abs() < 1e-9, "perfect draft: {curve:?}");
        }
        assert_eq!(spec.telemetry().cliff_horizon(0.9), curve.len());
    }
}

#[cfg(test)]
mod economics_tests {
    use super::*;

    const SYNTHETIC_CURVE: [f64; 7] = [1.0, 1.0, 1.0, 1.0, 1.0, 0.571, 0.0];

    #[test]
    fn optimal_horizon_sits_at_the_cliff() {
        // With this synthetic acceptance curve and illustrative cost model,
        // the selected horizon includes the last useful position.
        // The final position has zero acceptance and is pure cost, so the
        // selected horizon stops before it.
        let d = optimal_horizon(&SYNTHETIC_CURVE, 8);
        assert_eq!(d, 6, "value-optimal horizon includes the 0.571 position");
        let s5 = horizon_speedup(&SYNTHETIC_CURVE, 5, 4.0, 32.0, 0.35, 1.5);
        let s6 = horizon_speedup(&SYNTHETIC_CURVE, 6, 4.0, 32.0, 0.35, 1.5);
        let s7 = horizon_speedup(&SYNTHETIC_CURVE, 7, 4.0, 32.0, 0.35, 1.5);
        assert!(s5 > 1.0, "cliff horizon beats target-only: {s5}");
        assert!(
            s6 > s7,
            "the longer horizon is worse than the selected horizon"
        );
    }

    #[test]
    fn calibration_places_tiers_at_the_cliff() {
        let base = SpecConfig::default();
        let cfg = calibrate_tiers(&SYNTHETIC_CURVE, &base, 0.9);
        assert_eq!(cfg.block, 5, "aggressive tier at the cliff");
        assert_eq!(cfg.med_cap, 2);
        assert_eq!(cfg.min_cap, 1);
    }

    #[test]
    fn flat_curve_calibrates_to_max() {
        let flat = [0.94f64; 8];
        let cfg = calibrate_tiers(&flat, &SpecConfig::default(), 0.9);
        assert_eq!(cfg.block, 8, "flat high acceptance -> full block");
    }
}
