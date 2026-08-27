//! Canonical Q4_0 reference (CPU) — the differential oracle for the
//! batched `q4_0_batched_gemv.comp` shader.
//!
//! Q4_0 super-block: 32 values, 18 bytes = fp16 scale `d` (2 bytes) + 16
//! packed 4-bit codes (format layout: code `j` is byte `j/2`, low nibble
//! for even `j`).  Dequant is signed-nibble:
//!
//! ```text
//! x = (nibble - 8) * d        // nibble 0..15 → -8..7
//! ```
//!
//! This is a compact Q4_0 format used by the synthetic serving path and by
//! caller-supplied GGUF inputs.

use crate::adapter::{BatchStepModel, Hidden, StepOutcome};
use crate::q4k::{f16_to_f32, lcg_values};

pub const Q40_BLOCK_BYTES: usize = 18;
pub const Q40_BLOCK_VALUES: usize = 32;
/// Values per row (256) / values per block (32).
pub const Q40_BLOCKS_PER_ROW: usize = 8;

/// Dequantize one of the 32 values in a Q4_0 block.
pub fn q40_dequant(block: &[u8; Q40_BLOCK_BYTES], index: usize) -> f32 {
    let q_byte = block[2 + index / 2];
    let nibble = if index % 2 == 0 {
        q_byte & 15
    } else {
        q_byte >> 4
    };
    let d = f16_to_f32(block[0], block[1]);
    d * (nibble as i32 - 8) as f32
}

/// A quantized dense model in Q4_0: `rows`
/// rows × `dim` values (`dim/32` blocks of 32), embedding, greedy
/// semantics — mirrors `Q4KModel` exactly so both formats share the
/// scheduler contract and the differential harness.  `dim` is the real
/// model's hidden width, not a hardcoded 256.
#[derive(Clone)]
pub struct Q40Model {
    pub rows: usize,
    /// Values per row (hidden width).
    pub dim: usize,
    /// rows × `dim/32` Q4_0 blocks.
    pub weights: Vec<u8>,
    /// vocab × dim embedding, f32.
    pub embed: Vec<f32>,
    pub eos: u32,
}

impl Q40Model {
    /// Build from row-aligned blocks, 256 values per row (8 blocks/row) —
    /// the synthetic-format default.
    pub fn from_blocks(blocks: &[u8], vocab: usize, eos: u32, seed: u64) -> Self {
        Self::from_blocks_dim(blocks, vocab, 256, eos, seed)
    }

    /// Build from row-aligned blocks with an explicit row width.
    pub fn from_blocks_dim(blocks: &[u8], vocab: usize, dim: usize, eos: u32, seed: u64) -> Self {
        assert_eq!(dim % Q40_BLOCK_VALUES, 0, "dim must be block-aligned");
        let blocks_per_row = dim / Q40_BLOCK_VALUES;
        assert_eq!(blocks.len() % (blocks_per_row * Q40_BLOCK_BYTES), 0);
        let rows = blocks.len() / (blocks_per_row * Q40_BLOCK_BYTES);
        assert_eq!(rows, vocab, "one row per vocab entry");
        Self {
            rows,
            dim,
            weights: blocks.to_vec(),
            embed: lcg_values(vocab * dim, seed),
            eos,
        }
    }

    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.dim, "activations must match the row width");
        let bpr = self.dim / Q40_BLOCK_VALUES;
        let mut out = vec![0.0f32; self.rows];
        for (r, o) in out.iter_mut().enumerate() {
            let row = &self.weights[r * bpr * Q40_BLOCK_BYTES..(r + 1) * bpr * Q40_BLOCK_BYTES];
            let mut acc = 0.0f32;
            for (abs, xv) in x.iter().enumerate() {
                let block_off = (abs / Q40_BLOCK_VALUES) * Q40_BLOCK_BYTES;
                let block: [u8; Q40_BLOCK_BYTES] = row[block_off..block_off + Q40_BLOCK_BYTES]
                    .try_into()
                    .expect("block");
                acc += q40_dequant(&block, abs % Q40_BLOCK_VALUES) * xv;
            }
            *o = acc;
        }
        out
    }
}

impl BatchStepModel for Q40Model {
    fn batch_step(&self, inputs: &[(Hidden, u32)]) -> Vec<StepOutcome> {
        inputs
            .iter()
            .map(|(h, t)| {
                assert!(
                    (*t as usize) < self.rows,
                    "token {t} outside vocab {}",
                    self.rows
                );
                let base = *t as usize * self.dim;
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
        (self.rows * (self.dim / Q40_BLOCK_VALUES) * Q40_BLOCK_BYTES) as u64
    }
}

/// Deterministic synthetic Q4_0 weights: row `r`, value `v` gets a fixed
/// pattern so the reference can be checked exactly by hand.
pub fn synthetic_blocks(rows: usize, seed: u64) -> Vec<u8> {
    let mut rng_state = seed;
    let mut next = move || {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (rng_state >> 33) as u8
    };
    let mut out = Vec::with_capacity(rows * Q40_BLOCKS_PER_ROW * Q40_BLOCK_BYTES);
    for _ in 0..rows * Q40_BLOCKS_PER_ROW {
        // d = 1.0 (0x3c00), then 16 random code bytes.
        out.push(0x00);
        out.push(0x3c);
        for _ in 0..16 {
            out.push(next());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dequant_signed_nibble_exact() {
        // d = 1.0; byte 2 = 0x08 → value 0 (low 0x8) = 0, value 1 (high
        // 0x0) = -8; byte 3 = 0x7f → value 2 (low 0xf) = 7, value 3
        // (high 0x7) = -1; byte 4 = 0x00 → value 4 (low) = -8.
        let mut block = [0u8; Q40_BLOCK_BYTES];
        block[0] = 0x00;
        block[1] = 0x3c; // 1.0
        block[2] = 0x08;
        block[3] = 0x7f;
        block[4] = 0x00;
        assert_eq!(q40_dequant(&block, 0), 0.0);
        assert_eq!(q40_dequant(&block, 1), -8.0);
        assert_eq!(q40_dequant(&block, 2), 7.0);
        assert_eq!(q40_dequant(&block, 3), -1.0);
        assert_eq!(q40_dequant(&block, 4), -8.0);
    }

    #[test]
    fn synthetic_matvec_exact_ones() {
        let blocks = synthetic_blocks(4, 7);
        let m = Q40Model::from_blocks(&blocks, 4, 63, 99);
        // All-ones activations: row r = sum of dequantized values; with
        // d=1.0 the dequant sum is 256×(-8) + ... — verify against a
        // direct accumulation instead of a hand constant.
        let x = vec![1.0f32; 256];
        let out = m.matvec(&x);
        assert_eq!(out.len(), 4);
        for (r, o) in out.iter().enumerate() {
            let row = &m.weights[r * 144..(r + 1) * 144];
            let mut expect = 0.0f32;
            for abs in 0..256 {
                let off = (abs / 32) * 18;
                let b: [u8; 18] = row[off..off + 18].try_into().unwrap();
                expect += q40_dequant(&b, abs % 32);
            }
            assert_eq!(*o, expect, "row {r}");
        }
    }

    #[test]
    fn model_batched_matches_single() {
        let m = Q40Model::from_blocks(&synthetic_blocks(8, 3), 8, 63, 5);
        let h = m.initial_hidden();
        let batched = m.batch_step(&[(h.clone(), 0), (h.clone(), 4)]);
        let single = m.batch_step(&[(h, 0)]);
        assert_eq!(batched[0].next, single[0].next);
        assert_eq!(batched[0].logits, single[0].logits);
    }
}
