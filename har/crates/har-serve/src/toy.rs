//! Deterministic toy model for scheduler tests and exploratory benches.
//!
//! A tiny recurrent-ish stack: `h = layers(tanh(W_l · (h + E[t])))`, logits
//! `O · h`, greedy argmax sampling.  Everything derives from a fixed-seed
//! LCG so runs are bit-reproducible — the differential tests rely on this.
//!
//! The batched forward is one matrix-matrix multiply per layer, which is
//! exactly the amortization the scheduler exists to enable: `n` sequences
//! share every weight read.

use crate::adapter::{BatchStepModel, Hidden, StepOutcome};

#[derive(Clone, Debug)]
pub struct ToyConfig {
    pub dim: usize,
    pub vocab: usize,
    pub layers: usize,
    pub eos: u32,
    pub seed: u64,
}

impl Default for ToyConfig {
    fn default() -> Self {
        Self {
            dim: 256,
            vocab: 1024,
            layers: 4,
            eos: 999,
            seed: 0x5EED_2026,
        }
    }
}

/// Deterministic LCG in [-1, 1).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = (self.0 >> 40) as u32 as f32 / (1u32 << 24) as f32;
        u * 2.0 - 1.0
    }
}

#[derive(Clone)]
pub struct ToyModel {
    pub cfg: ToyConfig,
    /// Token embedding E (vocab × dim).
    embed: Vec<f32>,
    /// Per-layer weights W_l (dim × dim), row-major (out × in).
    weights: Vec<Vec<f32>>,
    /// Output projection O (vocab × dim).
    out: Vec<f32>,
}

impl ToyModel {
    pub fn new(cfg: ToyConfig) -> Self {
        let mut rng = Lcg(cfg.seed);
        let embed = (0..cfg.vocab * cfg.dim).map(|_| rng.next()).collect();
        let weights = (0..cfg.layers)
            .map(|_| (0..cfg.dim * cfg.dim).map(|_| rng.next()).collect())
            .collect();
        let out = (0..cfg.vocab * cfg.dim).map(|_| rng.next()).collect();
        Self {
            cfg,
            embed,
            weights,
            out,
        }
    }

    /// Row-major matrix-vector: y = M · x (M is out×in row-major).
    #[cfg(test)]
    fn matvec(m: &[f32], x: &[f32], out: usize, inn: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; out];
        for r in 0..out {
            let row = &m[r * inn..(r + 1) * inn];
            let mut acc = 0.0f32;
            for c in 0..inn {
                acc += row[c] * x[c];
            }
            y[r] = acc;
        }
        y
    }

    /// Row-major batched matrix multiply: Y (n×out) = X (n×in) · Mᵀ, i.e.
    /// per row `M · x` — the GEMM form of the GEMV the scheduler batches.
    fn batched_matvec(m: &[f32], xs: &[Vec<f32>], out: usize, inn: usize) -> Vec<Vec<f32>> {
        let n = xs.len();
        let mut ys = vec![vec![0.0f32; out]; n];
        for r in 0..out {
            let row = &m[r * inn..(r + 1) * inn];
            for (yi, x) in ys.iter_mut().zip(xs) {
                let mut acc = 0.0f32;
                for c in 0..inn {
                    acc += row[c] * x[c];
                }
                yi[r] = acc;
            }
        }
        ys
    }

    #[cfg(test)]
    fn forward_one(&self, hidden: &[f32], token: u32) -> (Vec<f32>, Vec<f32>) {
        let d = self.cfg.dim;
        let mut h: Vec<f32> = (0..d)
            .map(|i| hidden[i] + self.embed[token as usize * d + i])
            .collect();
        for w in &self.weights {
            let y = Self::matvec(w, &h, d, d);
            h = y.iter().map(|v| v.tanh()).collect();
        }
        let logits = Self::matvec(&self.out, &h, self.cfg.vocab, d);
        (h, logits)
    }
}

impl BatchStepModel for ToyModel {
    fn batch_step(&self, inputs: &[(Hidden, u32)]) -> Vec<StepOutcome> {
        let d = self.cfg.dim;
        // Residual-add the token embeddings, then one GEMM per layer across
        // the whole batch, then the output projection GEMM.
        let mut hs: Vec<Vec<f32>> = inputs
            .iter()
            .map(|(h, t)| {
                (0..d)
                    .map(|i| h[i] + self.embed[*t as usize * d + i])
                    .collect()
            })
            .collect();
        for w in &self.weights {
            hs = Self::batched_matvec(w, &hs, d, d);
            for h in hs.iter_mut() {
                for v in h.iter_mut() {
                    *v = v.tanh();
                }
            }
        }
        let logits = Self::batched_matvec(&self.out, &hs, self.cfg.vocab, d);
        hs.into_iter()
            .zip(logits)
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
        // Every layer's weight matrix plus the output projection, fp32.
        let d = self.cfg.dim;
        ((self.cfg.layers * d * d + d * self.cfg.vocab) * 4) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batched_forward_matches_single_forward() {
        let m = ToyModel::new(ToyConfig::default());
        let h0 = m.initial_hidden();
        let inputs = vec![(h0.clone(), 3u32), (h0.clone(), 7u32)];
        let batched = m.batch_step(&inputs);
        for (i, t) in [3u32, 7].iter().enumerate() {
            let (h1, l1) = m.forward_one(&h0, *t);
            assert_eq!(batched[i].next, h1, "hidden row {i} differs");
            assert_eq!(batched[i].logits, l1, "logits row {i} differ");
        }
    }

    #[test]
    fn deterministic_across_instances() {
        let a = ToyModel::new(ToyConfig::default());
        let b = ToyModel::new(ToyConfig::default());
        let h = a.initial_hidden();
        let ha = a.batch_step(&[(h.clone(), 42)])[0].next.clone();
        let hb = b.batch_step(&[(h, 42)])[0].next.clone();
        assert_eq!(ha, hb);
    }
}
