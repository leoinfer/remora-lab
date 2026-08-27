//! Canonical Q4_K reference (CPU) — the differential oracle for the
//! Vulkan `q4k_gemv` shader adapter.
//!
//! The dequant rules here mirror `shaders/q4k_gemv.comp` exactly
//! (validated against synthetic GGUF payloads and the small local numeric
//! tests:
//!
//! ```text
//! x = d * scale[group] * q - dmin * min[group]
//! ```
//!
//! with a 144-byte super-block: fp16 `d` @0, fp16 `dmin` @2, 12 scale/min
//! bytes @4, 128 q bytes @16 (256 values, 2 per byte, value `i` and
//! `i+32` sharing a byte).

use crate::adapter::{BatchStepModel, Hidden, StepOutcome};

pub const Q4K_BLOCK_BYTES: usize = 144;
pub const Q4K_BLOCK_VALUES: usize = 256;

/// IEEE binary16 → f32 (bit-exact).
pub fn f16_to_f32(b0: u8, b1: u8) -> f32 {
    let bits = u16::from_le_bytes([b0, b1]);
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let value = if exp == 0 {
        if frac == 0 {
            0.0
        } else {
            // subnormal
            let m = frac as f32 / 1024.0;
            m * 2f32.powi(-14)
        }
    } else if exp == 31 {
        if frac == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        let m = 1.0 + frac as f32 / 1024.0;
        m * 2f32.powi(exp as i32 - 15)
    };
    if sign == 1 {
        -value
    } else {
        value
    }
}

fn byte(block: &[u8; Q4K_BLOCK_BYTES], offset: usize) -> u8 {
    block[offset]
}

fn half_at(block: &[u8; Q4K_BLOCK_BYTES], offset: usize) -> f32 {
    f16_to_f32(byte(block, offset), byte(block, offset + 1))
}

fn scale_at(block: &[u8; Q4K_BLOCK_BYTES], index: usize) -> u32 {
    if index < 4 {
        byte(block, 4 + index) as u32 & 63
    } else {
        (byte(block, 4 + index + 4) as u32 & 15) | ((byte(block, 4 + index - 4) as u32 >> 6) << 4)
    }
}

fn min_at(block: &[u8; Q4K_BLOCK_BYTES], index: usize) -> u32 {
    if index < 4 {
        byte(block, 4 + index + 4) as u32 & 63
    } else {
        (byte(block, 4 + index + 4) as u32 >> 4) | ((byte(block, 4 + index) as u32 >> 6) << 4)
    }
}

/// Dequantize one of the 256 values in a Q4_K super-block.
pub fn q4k_dequant(block: &[u8; Q4K_BLOCK_BYTES], index: usize) -> f32 {
    let group = index / 32;
    let lane_in_64 = index & 63;
    let q_byte = byte(block, 16 + (index / 64) * 32 + (lane_in_64 & 31));
    let q = if lane_in_64 < 32 {
        q_byte & 15
    } else {
        q_byte >> 4
    };
    let d = half_at(block, 0);
    let dm = half_at(block, 2);
    d * scale_at(block, group) as f32 * q as f32 - dm * min_at(block, group) as f32
}

/// One row of a Q4_K weight matrix: `blocks` super-blocks (each 256
/// values) laid out consecutively.  Returns `values` (blocks × 256).
pub fn q4k_row_values(blocks: &[u8]) -> Vec<f32> {
    assert_eq!(blocks.len() % Q4K_BLOCK_BYTES, 0);
    let n = blocks.len() / Q4K_BLOCK_BYTES;
    let mut out = Vec::with_capacity(n * Q4K_BLOCK_VALUES);
    for b in 0..n {
        let block: [u8; Q4K_BLOCK_BYTES] = blocks[b * Q4K_BLOCK_BYTES..(b + 1) * Q4K_BLOCK_BYTES]
            .try_into()
            .expect("block slice");
        for i in 0..Q4K_BLOCK_VALUES {
            out.push(q4k_dequant(&block, i));
        }
    }
    out
}

/// Deterministic LCG in [-0.5, 0.5) for the embedding table.
pub fn lcg_values(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = (state >> 40) as u32 as f32 / (1u32 << 24) as f32;
        out.push(u - 0.5);
    }
    out
}

/// A real-quantized dense model: `rows` Q4_K rows (`dim/256`
/// super-blocks each) form the weight matrix; `vocab == rows`; logits =
/// W·(h + E[t]); next hidden is the same input vector.  Deterministic
/// and token-dependent, so the scheduler's differential contract applies
/// to a *real* quantization format, not just the toy.  `dim` is the
/// row width from the source tensor (256 for the fixture; 5120 for a
/// real 27B-class embedding).
#[derive(Clone)]
pub struct Q4KModel {
    pub rows: usize,
    /// Values per row (hidden width).
    pub dim: usize,
    /// rows × (dim/256) super-blocks.
    pub weights: Vec<u8>,
    /// vocab × dim embedding, f32.
    pub embed: Vec<f32>,
    pub eos: u32,
}

impl Q4KModel {
    /// One 256-value super-block per row (the fixture layout).
    pub fn from_blocks(blocks: &[u8], vocab: usize, eos: u32, seed: u64) -> Self {
        Self::from_blocks_dim(blocks, vocab, Q4K_BLOCK_VALUES, eos, seed)
    }

    /// Build with an explicit row width (multiple super-blocks per row).
    pub fn from_blocks_dim(blocks: &[u8], vocab: usize, dim: usize, eos: u32, seed: u64) -> Self {
        assert_eq!(dim % Q4K_BLOCK_VALUES, 0, "dim must be super-block-aligned");
        let blocks_per_row = dim / Q4K_BLOCK_VALUES;
        assert_eq!(
            blocks.len() % (blocks_per_row * Q4K_BLOCK_BYTES),
            0,
            "block-aligned weights"
        );
        let rows = blocks.len() / (blocks_per_row * Q4K_BLOCK_BYTES);
        assert_eq!(rows, vocab, "one row per vocab entry");
        Self {
            rows,
            dim,
            weights: blocks.to_vec(),
            embed: lcg_values(vocab * dim, seed),
            eos,
        }
    }

    /// Row-major matvec: out[r] = sum_c W[r][c] * x[c] with on-the-fly
    /// dequant (mirrors the shader's arithmetic order: per-row accumulate
    /// over blocks, lanes, then f32 sum).
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.dim, "activations must match the row width");
        let bpr = self.dim / Q4K_BLOCK_VALUES;
        let mut out = vec![0.0f32; self.rows];
        for (r, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for b in 0..bpr {
                let block: [u8; Q4K_BLOCK_BYTES] = self.weights
                    [(r * bpr + b) * Q4K_BLOCK_BYTES..(r * bpr + b + 1) * Q4K_BLOCK_BYTES]
                    .try_into()
                    .expect("block slice");
                for i in 0..Q4K_BLOCK_VALUES {
                    acc += q4k_dequant(&block, i) * x[b * Q4K_BLOCK_VALUES + i];
                }
            }
            *o = acc;
        }
        out
    }
}

impl BatchStepModel for Q4KModel {
    fn batch_step(&self, inputs: &[(Hidden, u32)]) -> Vec<StepOutcome> {
        let d = self.dim;
        inputs
            .iter()
            .map(|(h, t)| {
                let base = *t as usize * d;
                let x: Vec<f32> = h
                    .iter()
                    .enumerate()
                    .map(|(i, hv)| hv + self.embed[base + i])
                    .collect();
                let logits = self.matvec(&x);
                StepOutcome::plain(x, logits)
            })
            .collect()
    }

    fn initial_hidden(&self) -> Hidden {
        vec![0.0f32; self.dim]
    }

    fn eos(&self) -> u32 {
        self.eos
    }

    fn weight_bytes_per_row(&self) -> u64 {
        (self.rows * (self.dim / Q4K_BLOCK_VALUES) * Q4K_BLOCK_BYTES) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small deterministic Q4_K block and activation vector used by the
    /// publication tests. No model payload is required.
    fn fixture() -> ([u8; Q4K_BLOCK_BYTES], Vec<f32>, f32) {
        let mut block = [0u8; Q4K_BLOCK_BYTES];
        block[0] = 0x00;
        block[1] = 0x3c; // d = 1.0; all scales and codes remain zero.
        (block, lcg_values(Q4K_BLOCK_VALUES, 17), 0.0)
    }

    #[test]
    fn f16_conversion_is_exact() {
        assert_eq!(f16_to_f32(0x00, 0x3c), 1.0); // 0x3c00 = 1.0
        assert_eq!(f16_to_f32(0x00, 0x00), 0.0);
        assert_eq!(f16_to_f32(0x00, 0x80), -0.0);
        assert_eq!(f16_to_f32(0x00, 0xbc), -1.0); // 0xbc00 = -1.0
        assert_eq!(f16_to_f32(0x01, 0x3c), 1.0009766); // 0x3c01
        assert_eq!(f16_to_f32(0x01, 0x00), 5.9604645e-8); // bits 0x0001 = smallest subnormal 2^-24
        assert_eq!(f16_to_f32(0x00, 0x03), 4.5776367e-5); // bits 0x0300 = subnormal 0.75*2^-14
    }

    #[test]
    fn dequant_matches_synthetic_gguf_fixture() {
        let (block, input, reference) = fixture();
        let mut dot = 0.0f32;
        for (i, xv) in input.iter().enumerate() {
            dot += q4k_dequant(&block, i) * xv;
        }
        let abs_err = (dot - reference).abs();
        assert!(
            abs_err < 1e-6,
            "CPU Q4_K reference off by {abs_err:.3e} vs synthetic reference {reference}"
        );
    }

    #[test]
    fn model_batched_matches_single() {
        let blocks = vec![0u8; Q4K_BLOCK_BYTES * 8];
        let m = Q4KModel::from_blocks(&blocks, 8, 7, 99);
        let h = m.initial_hidden();
        let batched = m.batch_step(&[(h.clone(), 1), (h.clone(), 2)]);
        let single = m.batch_step(&[(h, 1)]);
        assert_eq!(batched[0].next, single[0].next);
        assert_eq!(batched[0].logits, single[0].logits);
    }
}
