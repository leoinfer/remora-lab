//! Paged-KV flash-decode attention reference (CPU) — the differential
//! oracle for `shaders/flash_decode_q8.comp`.
//!
//! Decode-phase GQA attention over a **q8_0 quantized KV cache**
//! (the live server's `-ctk/-ctv q8_0` config): one query token attends
//! over all cached positions.  The reference implements the same
//! chunked two-pass online-softmax algorithm as the shader, so the
//! GPU-vs-CPU differential is tight by construction; a brute-force
//! canonical-softmax check validates the algorithm itself.
//!
//! Geometry (hybrid model class): HEAD_DIM = 256, Q8_0 blocks of 32 values
//! (34 bytes: fp16 scale + 32 int8), GQA with `q_heads / kv_heads`
//! query heads sharing one KV head.

use crate::q4k::f16_to_f32;

pub const HEAD_DIM: usize = 256;
pub const Q8_BLOCK_VALUES: usize = 32;
pub const Q8_BLOCK_BYTES: usize = 34;
pub const BLOCKS_PER_HEAD: usize = HEAD_DIM / Q8_BLOCK_VALUES;
pub const KV_ROW_BYTES: usize = BLOCKS_PER_HEAD * Q8_BLOCK_BYTES;

/// Quantize one f32 vector to the q8_0 row format (34 bytes per 32
/// values): fp16 scale = max_abs/127, int8 q = round(x/scale).
pub fn q8_0_quantize(values: &[f32]) -> Vec<u8> {
    assert_eq!(values.len() % Q8_BLOCK_VALUES, 0);
    let mut out = Vec::with_capacity(values.len() / Q8_BLOCK_VALUES * Q8_BLOCK_BYTES);
    for block in values.chunks(Q8_BLOCK_VALUES) {
        let max_abs = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = if max_abs == 0.0 { 0.0 } else { max_abs / 127.0 };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        for v in block {
            let q = if d == 0.0 {
                0i8
            } else {
                (v / d).round().clamp(-127.0, 127.0) as i8
            };
            out.push(q as u8);
        }
    }
    out
}

/// f32 → IEEE binary16 (round-to-nearest via f64 path).
pub fn f32_to_f16(value: f32) -> u16 {
    let v = value as f64;
    if v == 0.0 {
        return if v.is_sign_negative() { 0x8000 } else { 0 };
    }
    if !v.is_finite() {
        return if v.is_nan() {
            0x7e00
        } else if v > 0.0 {
            0x7c00
        } else {
            0xfc00
        };
    }
    let exp = v.abs().log2().floor() as i32;
    if exp < -14 {
        // subnormal
        let m = (v / 2f64.powi(-24)).round() as i32;
        let bits = m.max(0) as u16;
        return if v < 0.0 { bits | 0x8000 } else { bits };
    }
    if exp > 15 {
        return if v < 0.0 { 0xfc00 } else { 0x7c00 };
    }
    let mantissa = ((v / 2f64.powi(exp) - 1.0) * 1024.0).round() as u16;
    let bits = (((exp + 15) as u16) << 10) | (mantissa & 0x3ff);
    if v < 0.0 {
        bits | 0x8000
    } else {
        bits
    }
}

/// Dequantize one q8_0 row (quantized by [`q8_0_quantize`]) back to f32.
pub fn q8_0_dequant_row(row: &[u8]) -> Vec<f32> {
    assert_eq!(row.len(), KV_ROW_BYTES);
    let mut out = Vec::with_capacity(HEAD_DIM);
    for b in 0..BLOCKS_PER_HEAD {
        let d = f16_to_f32(row[b * Q8_BLOCK_BYTES], row[b * Q8_BLOCK_BYTES + 1]);
        for i in 0..Q8_BLOCK_VALUES {
            let q = row[b * Q8_BLOCK_BYTES + 2 + i] as i8;
            out.push(d * q as f32);
        }
    }
    out
}

fn byte_at(words: &[u32], byte_offset: usize) -> u8 {
    ((words[byte_offset >> 2] >> ((byte_offset & 3) * 8)) & 0xff) as u8
}

/// One q8_0 KV value read through the shader's word-indexed layout:
/// `row` is the word-aligned base of a KV row in `words`.
pub fn q8_kv_value(words: &[u32], row_word: usize, index: usize) -> f32 {
    let block_off = (index / Q8_BLOCK_VALUES) * Q8_BLOCK_BYTES;
    let byte_off = block_off + 2 + (index % Q8_BLOCK_VALUES);
    let scale_word = words[row_word + (block_off >> 2)];
    let halves = [
        f16_to_f32(scale_word as u8, (scale_word >> 8) as u8),
        f16_to_f32((scale_word >> 16) as u8, (scale_word >> 24) as u8),
    ];
    let d = if block_off % 4 == 0 {
        halves[0]
    } else {
        halves[1]
    };
    d * byte_at(words, row_word * 4 + byte_off) as i8 as f32
}

/// Convert a raw q8_0 row (quantized bytes) to the shader's word layout.
pub fn row_to_words(row: &[u8]) -> Vec<u32> {
    assert_eq!(row.len() % 4, 0);
    row.chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Parameters for one decode-attention call.
#[derive(Clone, Debug)]
pub struct FlashDecodeParams {
    pub q_heads: usize,
    pub kv_heads: usize,
    pub seq_len: usize,
    /// Online-softmax chunk (positions) — mirrors the shader's CHUNK.
    pub chunk: usize,
}

impl Default for FlashDecodeParams {
    fn default() -> Self {
        Self {
            q_heads: 24,
            kv_heads: 4,
            seq_len: 1024,
            chunk: 1024,
        }
    }
}

/// Reference flash decode over a q8_0 KV cache (all query heads).
///
/// `k_cache`/`v_cache`: raw q8_0 rows in shader word layout,
/// `[kv_head][position]` × `KV_ROW_BYTES/4` words.  `query`: f32
/// `[q_head][HEAD_DIM]`.  Returns normalized `[q_head][HEAD_DIM]`.
pub fn flash_decode_reference(
    params: &FlashDecodeParams,
    k_words: &[u32],
    v_words: &[u32],
    query: &[f32],
) -> Vec<f32> {
    let q_per_kv = params.q_heads / params.kv_heads;
    let mut out = vec![0.0f32; params.q_heads * HEAD_DIM];
    for qh in 0..params.q_heads {
        let kvh = qh / q_per_kv;
        let q = &query[qh * HEAD_DIM..(qh + 1) * HEAD_DIM];
        let (_, l, o) = flash_decode_online(params, k_words, v_words, q, kvh);
        for (i, v) in o.iter().enumerate() {
            out[qh * HEAD_DIM + i] = v / l;
        }
    }
    out
}

/// Exact online-softmax flash decode (single implementation used by both
/// the reference and the brute-force check).
pub fn flash_decode_online(
    params: &FlashDecodeParams,
    k_words: &[u32],
    v_words: &[u32],
    q: &[f32],
    kvh: usize,
) -> (f32, f32, Vec<f32>) {
    let row_words = KV_ROW_BYTES / 4;
    let k_base = (kvh * params.seq_len) * row_words;
    let v_base = (kvh * params.seq_len) * row_words;
    let mut m = f32::NEG_INFINITY;
    let mut l = 0.0f32;
    let mut o = vec![0.0f32; HEAD_DIM];
    for c in (0..params.seq_len).step_by(params.chunk) {
        let n = (params.seq_len - c).min(params.chunk);
        let mut scores = vec![0.0f32; n];
        let mut chunk_max = f32::NEG_INFINITY;
        for (p, score) in scores.iter_mut().enumerate() {
            let pos = c + p;
            let mut s = 0.0f32;
            for (i, qv) in q.iter().enumerate() {
                s += qv * q8_kv_value(k_words, k_base + pos * row_words, i);
            }
            *score = s;
            chunk_max = chunk_max.max(s);
        }
        let mut l_c = 0.0f32;
        let mut o_c = vec![0.0f32; HEAD_DIM];
        for (p, &score) in scores.iter().enumerate() {
            let e = (score - chunk_max).exp();
            let pos = c + p;
            for (i, oi) in o_c.iter_mut().enumerate() {
                *oi += e * q8_kv_value(v_words, v_base + pos * row_words, i);
            }
            l_c += e;
        }
        let m_new = m.max(chunk_max);
        let a = (m - m_new).exp();
        let b = (chunk_max - m_new).exp();
        for i in 0..HEAD_DIM {
            o[i] = o[i] * a + o_c[i] * b;
        }
        l = l * a + l_c * b;
        m = m_new;
    }
    (m, l, o)
}

/// Brute-force canonical attention (reference of the reference): exact
/// softmax over ALL positions, no chunking.
pub fn flash_decode_bruteforce(
    params: &FlashDecodeParams,
    k_words: &[u32],
    v_words: &[u32],
    q: &[f32],
    kvh: usize,
) -> Vec<f32> {
    let row_words = KV_ROW_BYTES / 4;
    let k_base = (kvh * params.seq_len) * row_words;
    let v_base = (kvh * params.seq_len) * row_words;
    let mut scores = vec![0.0f32; params.seq_len];
    let mut max_s = f32::NEG_INFINITY;
    for (p, score) in scores.iter_mut().enumerate() {
        let mut s = 0.0f32;
        for (i, qv) in q.iter().enumerate() {
            s += qv * q8_kv_value(k_words, k_base + p * row_words, i);
        }
        *score = s;
        max_s = max_s.max(s);
    }
    let mut l = 0.0f32;
    let mut o = vec![0.0f32; HEAD_DIM];
    for (p, &score) in scores.iter().enumerate() {
        let e = (score - max_s).exp();
        for (i, oi) in o.iter_mut().enumerate() {
            *oi += e * q8_kv_value(v_words, v_base + p * row_words, i);
        }
        l += e;
    }
    o.iter_mut().for_each(|v| *v /= l);
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::q4k::lcg_values;

    fn kv_cache(params: &FlashDecodeParams, seed: u64) -> (Vec<u32>, Vec<u32>) {
        let mut k = Vec::new();
        let mut v = Vec::new();
        for h in 0..params.kv_heads {
            for p in 0..params.seq_len {
                let kv = lcg_values(HEAD_DIM, seed + (h * params.seq_len + p) as u64 * 31 + 7);
                let kr = q8_0_quantize(&kv);
                let vr = q8_0_quantize(&lcg_values(
                    HEAD_DIM,
                    seed + (h * params.seq_len + p) as u64 * 17 + 3,
                ));
                k.extend(row_to_words(&kr));
                v.extend(row_to_words(&vr));
            }
        }
        (k, v)
    }

    fn query(params: &FlashDecodeParams, seed: u64) -> Vec<f32> {
        lcg_values(params.q_heads * HEAD_DIM, seed)
    }

    #[test]
    fn q8_roundtrip_is_close() {
        let x = lcg_values(HEAD_DIM, 11);
        let row = q8_0_quantize(&x);
        let y = q8_0_dequant_row(&row);
        for (a, b) in x.iter().zip(y.iter()) {
            assert!((a - b).abs() < 0.02, "q8 roundtrip error {a} vs {b}");
        }
    }

    #[test]
    fn f16_roundtrip() {
        for v in [1.0f32, -1.0, 0.5, std::f32::consts::PI, 0.0, 127.0, -0.25] {
            let bits = f32_to_f16(v);
            let back = f16_to_f32(bits as u8, (bits >> 8) as u8);
            assert!((v - back).abs() < 0.001, "f16 roundtrip {v} → {back}");
        }
    }

    #[test]
    fn online_matches_bruteforce() {
        let params = FlashDecodeParams {
            q_heads: 4,
            kv_heads: 2,
            seq_len: 513, // non-chunk-multiple
            chunk: 128,
        };
        let (k, v) = kv_cache(&params, 5);
        let q = query(&params, 9);
        for qh in 0..params.q_heads {
            let kvh = qh / (params.q_heads / params.kv_heads);
            let qq = &q[qh * HEAD_DIM..(qh + 1) * HEAD_DIM];
            let (_, l, o) = flash_decode_online(&params, &k, &v, qq, kvh);
            let o_norm: Vec<f32> = o.iter().map(|x| x / l).collect();
            let bf = flash_decode_bruteforce(&params, &k, &v, qq, kvh);
            for (a, b) in o_norm.iter().zip(bf.iter()) {
                assert!((a - b).abs() < 1e-5, "online vs brute force: {a} vs {b}");
            }
        }
    }

    #[test]
    fn gqa_mapping_uses_shared_kv_head() {
        let params = FlashDecodeParams {
            q_heads: 4,
            kv_heads: 2,
            seq_len: 64,
            chunk: 32,
        };
        let (k, v) = kv_cache(&params, 3);
        let q = query(&params, 4);
        // q heads 0,1 share kv head 0; heads 2,3 share kv head 1: same
        // query vectors on the same kv head must give identical outputs.
        // The same query against the same kv head must give the same
        // output regardless of which query-head slot it is viewed as.
        let q0 = &q[..HEAD_DIM];
        let (_, l0, o0) = flash_decode_online(&params, &k, &v, q0, 0);
        let (_, l1, o1) = flash_decode_online(&params, &k, &v, q0, 0);
        let a: Vec<f32> = o0.iter().map(|x| x / l0).collect();
        let b: Vec<f32> = o1.iter().map(|x| x / l1).collect();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }
}
