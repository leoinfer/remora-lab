//! Native MoE path — synthetic (CPU-verifiable) routed-expert model with
//! **expert-major multi-position batching**: positions that route to the same
//! expert are batched
//! into ONE forward per expert, so each routed expert's weights are read
//! once per step instead of once per position — the ÷K bandwidth lever
//! (exact for attention-only models under the stated accounting; here proven on
//! the synthetic MoE with the differential contract).
//!
//! Structure: embedding → router (top-k softmax over `experts`) → routed
//! experts (up/down MLP) + optional shared expert → hidden' → output
//! projection.  All weights are deterministic LCG (same convention as
//! the other backends).

use crate::adapter::{BatchStepModel, Hidden, Logits, StepOutcome};
use crate::q4k::lcg_values;
use std::cell::RefCell;

#[derive(Clone, Debug)]
pub struct MoEConfig {
    pub dim: usize,
    pub experts: usize,
    pub top_k: usize,
    pub shared_expert: bool,
    pub expert_hidden: usize,
    pub vocab: usize,
    pub eos: u32,
    pub seed: u64,
}

impl Default for MoEConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            experts: 16,
            top_k: 2,
            shared_expert: true,
            expert_hidden: 128,
            vocab: 512,
            eos: 499,
            seed: 42,
        }
    }
}

/// Per-step MoE accounting: the ÷K evidence.
#[derive(Clone, Debug, Default)]
pub struct MoETelemetry {
    positions: RefCell<u64>,
    /// Distinct routed experts read this step (each read once).
    expert_reads: RefCell<u64>,
    /// Shared-expert reads (1 per step when enabled).
    shared_reads: RefCell<u64>,
}

impl MoETelemetry {
    pub fn positions(&self) -> u64 {
        *self.positions.borrow()
    }
    pub fn expert_reads(&self) -> u64 {
        *self.expert_reads.borrow()
    }
    pub fn shared_reads(&self) -> u64 {
        *self.shared_reads.borrow()
    }
    /// The ÷K factor achieved over the last steps: naive per-token
    /// expert reads (positions × top_k) vs actual distinct reads.
    pub fn division_factor(&self) -> f64 {
        let p = self.positions();
        let e = self.expert_reads();
        if p == 0 || e == 0 {
            return 1.0;
        }
        (p as f64 * 2.0) / e as f64
    }
}

pub struct MoEModel {
    pub cfg: MoEConfig,
    /// embed: vocab × dim
    pub embed: Vec<f32>,
    /// router: experts × dim
    pub router_w: Vec<f32>,
    /// expert up: experts × (expert_hidden × dim)
    pub expert_up: Vec<Vec<f32>>,
    /// expert down: experts × (dim × expert_hidden)
    pub expert_down: Vec<Vec<f32>>,
    /// shared up/down (when enabled)
    pub shared_up: Vec<f32>,
    pub shared_down: Vec<f32>,
    /// output: vocab × dim
    pub out: Vec<f32>,
    pub telemetry: MoETelemetry,
}

fn matvec(m: &[f32], x: &[f32], out_n: usize, in_n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_n];
    for r in 0..out_n {
        let row = &m[r * in_n..(r + 1) * in_n];
        let mut acc = 0.0f32;
        for (c, xv) in x.iter().enumerate() {
            acc += row[c] * xv;
        }
        y[r] = acc;
    }
    y
}

/// Batched matvec (rows share the weight read — the ÷K core).
fn batched_matvec(m: &[f32], xs: &[Vec<f32>], out_n: usize, in_n: usize) -> Vec<Vec<f32>> {
    let mut ys = vec![vec![0.0f32; out_n]; xs.len()];
    for r in 0..out_n {
        let row = &m[r * in_n..(r + 1) * in_n];
        for (yi, x) in ys.iter_mut().zip(xs) {
            let mut acc = 0.0f32;
            for (c, xv) in x.iter().enumerate() {
                acc += row[c] * xv;
            }
            yi[r] = acc;
        }
    }
    ys
}

impl MoEModel {
    pub fn new(cfg: MoEConfig) -> Self {
        let mut next = 0u64;
        let mut gen = move |count: usize| {
            let v = lcg_values(count, cfg.seed + next);
            next += 1;
            v
        };
        Self {
            embed: gen(cfg.vocab * cfg.dim),
            router_w: gen(cfg.experts * cfg.dim),
            expert_up: (0..cfg.experts)
                .map(|_| gen(cfg.expert_hidden * cfg.dim))
                .collect(),
            expert_down: (0..cfg.experts)
                .map(|_| gen(cfg.dim * cfg.expert_hidden))
                .collect(),
            shared_up: gen(cfg.expert_hidden * cfg.dim),
            shared_down: gen(cfg.dim * cfg.expert_hidden),
            out: gen(cfg.vocab * cfg.dim),
            telemetry: MoETelemetry::default(),
            cfg,
        }
    }

    /// Router logits → top-k (expert, weight) with softmax over the top-k.
    fn route(&self, h: &[f32]) -> Vec<(usize, f32)> {
        let logits = matvec(&self.router_w, h, self.cfg.experts, self.cfg.dim);
        let mut order: Vec<usize> = (0..self.cfg.experts).collect();
        order.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        let top = &order[..self.cfg.top_k];
        let max = logits[top[0]];
        let mut weights = Vec::with_capacity(self.cfg.top_k);
        let mut sum = 0.0f32;
        for &e in top {
            let w = (logits[e] - max).exp();
            weights.push(w);
            sum += w;
        }
        top.iter()
            .zip(weights)
            .map(|(&e, w)| (e, w / sum))
            .collect()
    }

    /// Expert FFN (up → relu → down), batched over the routed positions.
    fn expert_forward(&self, up: &[f32], down: &[f32], xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let d = self.cfg.dim;
        let eh = self.cfg.expert_hidden;
        let mids = batched_matvec(up, xs, eh, d);
        let mut acts = Vec::with_capacity(mids.len());
        for m in &mids {
            acts.push(m.iter().map(|v| v.max(0.0)).collect::<Vec<f32>>());
        }
        batched_matvec(down, &acts, d, eh)
    }

    /// One forward for a batch of (hidden, token) inputs — expert-major.
    pub fn forward_batch(&self, inputs: &[(Hidden, u32)]) -> Vec<(Hidden, Logits)> {
        let d = self.cfg.dim;
        let xs: Vec<Vec<f32>> = inputs
            .iter()
            .map(|(h, t)| {
                let base = *t as usize * d;
                h.iter()
                    .enumerate()
                    .map(|(i, hv)| hv + self.embed[base + i])
                    .collect()
            })
            .collect();

        // Router decisions + expert-major grouping (positions by expert).
        let routes: Vec<Vec<(usize, f32)>> = xs.iter().map(|x| self.route(x)).collect();
        let mut groups: Vec<Vec<usize>> = vec![Vec::new(); self.cfg.experts];
        for (p, route) in routes.iter().enumerate() {
            for (e, _) in route {
                groups[*e].push(p);
            }
        }

        let mut expert_contrib: Vec<Vec<f32>> = vec![vec![0.0f32; d]; xs.len()];
        let mut distinct = 0usize;
        for (e, members) in groups.iter().enumerate() {
            if members.is_empty() {
                continue;
            }
            distinct += 1;
            let batch: Vec<Vec<f32>> = members.iter().map(|&p| xs[p].clone()).collect();
            let outs = self.expert_forward(&self.expert_up[e], &self.expert_down[e], &batch);
            for (k, &p) in members.iter().enumerate() {
                let w = routes[p].iter().find(|(ee, _)| *ee == e).unwrap().1;
                for (i, v) in outs[k].iter().enumerate() {
                    expert_contrib[p][i] += w * v;
                }
            }
        }

        // Shared expert (batched over ALL positions — its read is per step).
        let shared_contrib = if self.cfg.shared_expert {
            let outs = self.expert_forward(&self.shared_up, &self.shared_down, &xs);
            let mut base = vec![vec![0.0f32; d]; xs.len()];
            for (p, o) in outs.iter().enumerate() {
                for (i, v) in o.iter().enumerate() {
                    base[p][i] += v;
                }
            }
            base
        } else {
            vec![vec![0.0f32; d]; xs.len()]
        };

        // Telemetry: distinct expert reads + shared read for this step.
        {
            let mut er = self.telemetry.expert_reads.borrow_mut();
            *er += distinct as u64;
            let mut sr = self.telemetry.shared_reads.borrow_mut();
            *sr += if self.cfg.shared_expert { 1 } else { 0 };
            let mut pos = self.telemetry.positions.borrow_mut();
            *pos += xs.len() as u64;
        }

        xs.into_iter()
            .enumerate()
            .map(|(p, x)| {
                let h_next: Vec<f32> = (0..d)
                    .map(|i| expert_contrib[p][i] + shared_contrib[p][i])
                    .collect();
                let logits = matvec(&self.out, &h_next, self.cfg.vocab, d);
                (x, logits)
            })
            .collect()
    }
}

impl BatchStepModel for MoEModel {
    fn batch_step(&self, inputs: &[(Hidden, u32)]) -> Vec<StepOutcome> {
        self.forward_batch(inputs)
            .into_iter()
            .map(|(h, l)| StepOutcome::plain(h, l))
            .collect()
    }

    fn initial_hidden(&self) -> Hidden {
        vec![0.0f32; self.cfg.dim]
    }

    fn eos(&self) -> u32 {
        self.cfg.eos
    }

    fn weight_bytes_per_row(&self) -> u64 {
        // Per-token worst case: shared + top_k routed experts, fp32.
        let d = self.cfg.dim as u64;
        let eh = self.cfg.expert_hidden as u64;
        let expert = (eh * d + d * eh) * 4;
        let shared = if self.cfg.shared_expert { expert } else { 0 };
        shared + self.cfg.top_k as u64 * expert
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> MoEModel {
        MoEModel::new(MoEConfig {
            dim: 32,
            experts: 8,
            top_k: 2,
            shared_expert: true,
            expert_hidden: 48,
            vocab: 256,
            eos: 255,
            seed: 7,
        })
    }

    #[test]
    fn batched_equals_isolated() {
        // The ÷K correctness contract: one batched expert-major forward
        // must equal the same positions processed one at a time.
        let m = model();
        let h0 = m.initial_hidden();
        let inputs = vec![(h0.clone(), 3u32), (h0.clone(), 7u32), (h0.clone(), 11u32)];
        let batched = m.batch_step(&inputs);
        for (i, (h, t)) in inputs.iter().enumerate() {
            let single = m.batch_step(&[(h.clone(), *t)]);
            assert_eq!(batched[i].next, single[0].next, "hidden row {i}");
            assert_eq!(batched[i].logits, single[0].logits, "logits row {i}");
        }
    }

    #[test]
    fn expert_major_batching_amortizes_reads() {
        // With identical inputs, all positions route identically → 1
        // distinct expert group set; the division factor must be > 1
        // (the ÷K effect) once positions share routed experts.
        let m = model();
        let h0 = m.initial_hidden();
        let inputs = vec![(h0.clone(), 3u32), (h0.clone(), 3u32), (h0.clone(), 3u32)];
        m.batch_step(&inputs);
        let positions = m.telemetry.positions();
        let reads = m.telemetry.expert_reads();
        assert_eq!(positions, 3);
        assert!(
            reads < 3 * 2,
            "expert-major batching must share reads: {reads} < {}",
            3 * 2
        );
        let div = m.telemetry.division_factor();
        assert!(div > 1.0, "÷K factor {div}");
    }

    #[test]
    fn deterministic() {
        let a = model();
        let b = model();
        let h = a.initial_hidden();
        let oa = a.batch_step(&[(h.clone(), 5)]);
        let ob = b.batch_step(&[(h, 5)]);
        assert_eq!(oa[0].next, ob[0].next);
        assert_eq!(oa[0].logits, ob[0].logits);
    }

    #[test]
    fn router_topk_weights_sum_to_one() {
        let m = model();
        let h = vec![0.5f32; m.cfg.dim];
        let route = m.route(&h);
        assert_eq!(route.len(), m.cfg.top_k);
        let sum: f32 = route.iter().map(|(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
