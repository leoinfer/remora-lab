//! Native dense transformer reference (CPU) — the real-model-shaped
//! backend behind `--model dense`.
//!
//! Unlike the single-layer `toy`/`Q4K`/`Q4_0` models, this is a complete
//! causal decoder stack: token embeddings, per-layer GQA attention with
//! RoPE + RMSNorm, and a SiLU gated MLP, ending in a vocab projection.
//! It is the reference implementation of the forward the Vulkan kernels
//! will eventually accelerate, and the differential oracle for the
//! quantized layer kernels.
//!
//! The recurrent hidden state is the whole model state: `[x | kv]` where
//! `kv` holds every layer's K/V rows for every consumed position (the
//! serving layer's KV abstraction — the same packing the scheduler's
//! prefix graph stores per position, so prefix resume carries exact KV
//! history).  Position for RoPE is implicit in the KV length.
//!
//! Weights are deterministic LCG (small init) — no training data, no
//! caller-supplied model needed for the contract tests.

use crate::adapter::{BatchStepModel, Hidden, StepOutcome};

/// Geometry of a dense decoder model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DenseConfig {
    /// Model width (hidden dim).
    pub dim: usize,
    /// Transformer layers.
    pub n_layers: usize,
    /// Query heads (GQA: q_heads / kv_heads query heads share one KV head).
    pub n_heads: usize,
    /// KV heads.
    pub n_kv_heads: usize,
    /// Per-head width.
    pub head_dim: usize,
    pub vocab: usize,
    pub eos: u32,
    pub seed: u64,
}

impl Default for DenseConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            n_layers: 3,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 16,
            vocab: 512,
            eos: 499,
            seed: 42,
        }
    }
}

/// F32 rows per (K or V) position per layer.
fn kv_span(cfg: &DenseConfig) -> usize {
    cfg.n_kv_heads * cfg.head_dim
}

/// KV length (positions consumed) implied by a hidden state.
pub fn dense_kv_len(cfg: &DenseConfig, hidden: &[f32]) -> usize {
    let span = kv_span(cfg);
    let per_layer = 2 * span;
    assert!(
        hidden.len() >= cfg.dim && (hidden.len() - cfg.dim) % (per_layer * cfg.n_layers) == 0,
        "hidden must be [x | kv]: len {} dim {}",
        hidden.len(),
        cfg.dim
    );
    (hidden.len() - cfg.dim) / (per_layer * cfg.n_layers)
}

#[derive(Clone)]
pub struct DenseModel {
    pub cfg: DenseConfig,
    /// vocab × dim token embedding.
    embed: Vec<f32>,
    /// Per layer: [attn_norm (dim), wq, wk, wv, wo (dim×dim each),
    /// ffn_norm (dim), w_gate, w_up, w_down (dim×dim each)].
    layers: Vec<LayerWeights>,
    /// vocab × dim output projection.
    out: Vec<f32>,
}

#[derive(Clone)]
struct LayerWeights {
    attn_norm: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    ffn_norm: Vec<f32>,
    w_gate: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
}

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

fn rmsnorm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(gamma).map(|(v, g)| v * inv * g).collect()
}

fn rope(q: &[f32], pos: usize) -> Vec<f32> {
    let d = q.len();
    let mut out = q.to_vec();
    for i in (0..d - 1).step_by(2) {
        let theta = pos as f32 / 10_000f32.powf(2.0 * i as f32 / d as f32);
        let (c, s) = (theta.cos(), theta.sin());
        let (a, b) = (out[i], out[i + 1]);
        out[i] = a * c - b * s;
        out[i + 1] = a * s + b * c;
    }
    out
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

impl DenseModel {
    pub fn new(cfg: DenseConfig) -> Self {
        let scale = 0.1f32;
        let mat = |rows: usize, cols: usize, rng: &mut dyn FnMut() -> f32| {
            (0..rows * cols)
                .map(|_| rng() * scale)
                .collect::<Vec<f32>>()
        };
        let mut state = cfg.seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 40) as u32 as f32 / (1u32 << 24) as f32 - 0.5
        };
        let embed = (0..cfg.vocab * cfg.dim).map(|_| next() * scale).collect();
        let kv_in = cfg.n_kv_heads * cfg.head_dim;
        let layers = (0..cfg.n_layers)
            .map(|_| LayerWeights {
                attn_norm: (0..cfg.dim).map(|_| 1.0 + next() * 0.05).collect(),
                wq: mat(cfg.dim, cfg.dim, &mut next),
                wk: mat(kv_in, cfg.dim, &mut next),
                wv: mat(kv_in, cfg.dim, &mut next),
                wo: mat(cfg.dim, cfg.dim, &mut next),
                ffn_norm: (0..cfg.dim).map(|_| 1.0 + next() * 0.05).collect(),
                w_gate: mat(cfg.dim, cfg.dim, &mut next),
                w_up: mat(cfg.dim, cfg.dim, &mut next),
                w_down: mat(cfg.dim, cfg.dim, &mut next),
            })
            .collect();
        let out = (0..cfg.vocab * cfg.dim).map(|_| next() * scale).collect();
        Self {
            cfg,
            embed,
            layers,
            out,
        }
    }

    /// One forward step: consumes `token` at position `kv_len` (implicit
    /// in the hidden state) and returns the next hidden + logits.
    fn forward_one(&self, hidden: &[f32], token: u32) -> (Hidden, Vec<f32>) {
        let cfg = &self.cfg;
        let span = kv_span(cfg);
        let pos = dense_kv_len(cfg, hidden); // cached positions are 0..pos
        let d = cfg.dim;
        let q_per_kv = cfg.n_heads / cfg.n_kv_heads;
        let inv_scale = 1.0 / (cfg.head_dim as f32).sqrt();

        let mut x: Vec<f32> = (0..d)
            .map(|i| hidden[i] + self.embed[token as usize * d + i])
            .collect();

        let mut kv_new: Vec<Vec<f32>> = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let w = &self.layers[l];
            // --- Attention ---
            let xn = rmsnorm(&x, &w.attn_norm, 1e-5);
            let q_all = matvec(&w.wq, &xn, d, d);
            let k_all = matvec(&w.wk, &xn, span, d);
            let v_all = matvec(&w.wv, &xn, span, d);

            // RoPE the query heads and KV heads at the current position.
            let q_rope: Vec<Vec<f32>> = (0..cfg.n_heads)
                .map(|h| rope(&q_all[h * cfg.head_dim..(h + 1) * cfg.head_dim], pos))
                .collect();
            let k_rope: Vec<Vec<f32>> = (0..cfg.n_kv_heads)
                .map(|h| rope(&k_all[h * cfg.head_dim..(h + 1) * cfg.head_dim], pos))
                .collect();

            // Causal attention over cached positions 0..pos.  Past K rows
            // are stored ALREADY-ROPED at their own positions, so the
            // score is roped_q · stored_roped_k directly.
            let mut ctx = vec![0.0f32; d];
            for qh in 0..cfg.n_heads {
                let kvh = qh / q_per_kv;
                let q = &q_rope[qh];
                let mut scores = Vec::with_capacity(pos);
                let mut max_s = f32::NEG_INFINITY;
                for p in 0..pos {
                    let k_row = &hidden
                        [k_offset(cfg, l, p, pos, span)..k_offset(cfg, l, p, pos, span) + span];
                    let mut s = 0.0f32;
                    for i in 0..cfg.head_dim {
                        s += q[i] * k_row[kvh * cfg.head_dim + i];
                    }
                    scores.push(s * inv_scale);
                    max_s = max_s.max(scores[p]);
                }
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max_s).exp();
                    sum += *s;
                }
                let mut acc = vec![0.0f32; cfg.head_dim];
                for p in 0..pos {
                    let w_p = scores[p] / sum;
                    let v_row = &hidden
                        [v_offset(cfg, l, p, pos, span)..v_offset(cfg, l, p, pos, span) + span];
                    for i in 0..cfg.head_dim {
                        acc[i] += w_p * v_row[kvh * cfg.head_dim + i];
                    }
                }
                ctx[qh * cfg.head_dim..(qh + 1) * cfg.head_dim].copy_from_slice(&acc);
            }
            let o = matvec(&w.wo, &ctx, d, d);
            for i in 0..d {
                x[i] += o[i];
            }

            // Store the new K (roped) and V (raw) rows at position `pos`.
            let mut kv = Vec::with_capacity(2 * span);
            for row in &k_rope {
                kv.extend_from_slice(row);
            }
            kv.extend_from_slice(&v_all);
            kv_new.push(kv);

            // --- FFN ---
            let xn = rmsnorm(&x, &w.ffn_norm, 1e-5);
            let gate = matvec(&w.w_gate, &xn, d, d);
            let up = matvec(&w.w_up, &xn, d, d);
            let mut gated = vec![0.0f32; d];
            for i in 0..d {
                gated[i] = silu(gate[i]) * up[i];
            }
            let down = matvec(&w.w_down, &gated, d, d);
            for i in 0..d {
                x[i] += down[i];
            }
        }

        let logits = matvec(&self.out, &x, cfg.vocab, d);

        // New hidden: [x | per-layer K rows (0..=pos) then V rows (0..=pos)].
        let kv_len = pos + 1;
        let mut h = Vec::with_capacity(d + cfg.n_layers * 2 * span * kv_len);
        h.extend_from_slice(&x);
        for (l, kv) in kv_new.iter().enumerate() {
            let k_start = k_offset(cfg, l, 0, pos, span);
            h.extend_from_slice(&hidden[k_start..k_start + span * pos]);
            h.extend_from_slice(&kv[..span]); // new K row at pos
            let v_start = v_offset(cfg, l, 0, pos, span);
            h.extend_from_slice(&hidden[v_start..v_start + span * pos]);
            h.extend_from_slice(&kv[span..]); // new V row at pos
        }
        (h, logits)
    }
}

/// Offset of layer `l`'s K rows for position `p` in a hidden state whose
/// cache holds `kv_len` positions.  Layout: `[x | per layer: K(0..kv_len)
/// | V(0..kv_len)]`.
fn k_offset(cfg: &DenseConfig, l: usize, p: usize, kv_len: usize, span: usize) -> usize {
    cfg.dim + l * 2 * span * kv_len + p * span
}

fn v_offset(cfg: &DenseConfig, l: usize, p: usize, kv_len: usize, span: usize) -> usize {
    cfg.dim + (l * 2 + 1) * span * kv_len + p * span
}

impl BatchStepModel for DenseModel {
    fn batch_step(&self, inputs: &[(Hidden, u32)]) -> Vec<StepOutcome> {
        inputs
            .iter()
            .map(|(h, t)| {
                assert!((*t as usize) < self.cfg.vocab, "token {t} outside vocab");
                let (h2, logits) = self.forward_one(h, *t);
                StepOutcome::plain(h2, logits)
            })
            .collect()
    }

    fn initial_hidden(&self) -> Hidden {
        vec![0.0f32; self.cfg.dim]
    }

    fn eos(&self) -> u32 {
        self.cfg.eos
    }

    fn weight_bytes_per_row(&self) -> u64 {
        // Honest accounting: total weight bytes (fp32) / vocab rows.
        let cfg = &self.cfg;
        let kv_in = cfg.n_kv_heads * cfg.head_dim;
        let per_layer = (cfg.dim * cfg.dim) * 7 + kv_in * cfg.dim * 2 + cfg.dim * 2;
        let total = (cfg.vocab * cfg.dim + cfg.n_layers * per_layer + cfg.vocab * cfg.dim) * 4;
        (total / cfg.vocab.max(1)) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> DenseModel {
        DenseModel::new(DenseConfig::default())
    }

    #[test]
    fn hidden_grows_by_kv_span_per_token() {
        let m = model();
        let cfg = m.cfg;
        let span = kv_span(&cfg);
        let h0 = m.initial_hidden();
        assert_eq!(dense_kv_len(&cfg, &h0), 0);
        let (h1, _) = m.forward_one(&h0, 5);
        assert_eq!(dense_kv_len(&cfg, &h1), 1);
        assert_eq!(h1.len(), cfg.dim + 2 * cfg.n_layers * span);
        let (h2, _) = m.forward_one(&h1, 7);
        assert_eq!(dense_kv_len(&cfg, &h2), 2);
        assert_eq!(h2.len(), cfg.dim + 2 * cfg.n_layers * span * 2);
    }

    #[test]
    fn deterministic_and_batched_equals_isolated() {
        let m = model();
        let h = m.initial_hidden();
        let batched = m.batch_step(&[(h.clone(), 3), (h.clone(), 9)]);
        let a = m.batch_step(&[(h.clone(), 3)]);
        let b = m.batch_step(&[(h, 9)]);
        assert_eq!(batched[0].next, a[0].next, "batched row 0");
        assert_eq!(batched[0].logits, a[0].logits, "logits row 0");
        assert_eq!(batched[1].next, b[0].next, "batched row 1");
        assert_eq!(batched[1].logits, b[0].logits, "logits row 1");
    }

    #[test]
    fn prefill_fold_matches_stepwise() {
        let m = model();
        let h = m.initial_hidden();
        let (states, logits) = m.prefill_batch(&[(h.clone(), vec![5, 6, 7])])[0].clone();
        let mut hh = h;
        for (i, t) in [5u32, 6, 7].iter().enumerate() {
            let out = m.batch_step(&[(hh, *t)]);
            hh = out[0].next.clone();
            assert_eq!(states[i], hh, "state {i}");
            assert_eq!(logits[i], out[0].logits, "logits {i}");
        }
    }

    #[test]
    fn rope_makes_absolute_position_part_of_the_state() {
        // The same tokens at different absolute positions are different
        // states (RoPE is position-indexed).  This is why the scheduler's
        // prefix graph must restore the state stored AT the matched
        // position — the resume differential test (serve_dense) checks
        // that property end-to-end.
        let m = model();
        let h0 = m.initial_hidden();
        let (h_ab, _) = m.forward_one(&h0, 5);
        let (h_ab, _) = m.forward_one(&h_ab, 7);
        let (h_9, _) = m.forward_one(&h0, 9);
        let (h_9_5, _) = m.forward_one(&h_9, 5);
        let (h_9_5_7, _) = m.forward_one(&h_9_5, 7);
        assert_ne!(h_ab, h_9_5_7, "different absolute positions differ");
        // And the same prefix at the same positions is exactly identical
        // (determinism — the resume contract).
        let (h_ab2, _) = m.forward_one(&h0, 5);
        let (h_ab2, _) = m.forward_one(&h_ab2, 7);
        assert_eq!(h_ab, h_ab2, "same prefix at same positions is identical");
    }
}
