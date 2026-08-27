//! Block quantization codecs: symmetric, group size 32.
//!
//! Wire layout per 32-element group (little-endian):
//!   Q8: [f16 scale][32 x int8]                     = 34 B
//!   Q6: [f16 scale][192-bit LE packed 6-bit codes] = 26 B
//!   Q4: [f16 scale][16 B nibbles: byte j = q(2j) | q(2j+1)<<4] = 18 B
//!   Q3: [f16 scale][96-bit LE packed 3-bit codes]  = 14 B
//!   F16: raw binary16                              = 64 B
//!
//! Codes are signed symmetric with clamping to [-max_code, +max_code]:
//! max_code = 127 (Q8), 31 (Q6), 7 (Q4), 3 (Q3). Scale d = amax / max_code;
//! amax == 0 encodes as all-zero group with d == 0.

use crate::f16::{f16_to_f32, f32_to_f16};
use crate::Fmt;

pub const GROUP: usize = crate::GROUP;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupStats {
    pub max_abs_err: f32,
    pub sum_sq_err: f64,
    pub sum_sq_ref: f64,
}

impl GroupStats {
    pub fn rel_rms(&self) -> f32 {
        if self.sum_sq_ref == 0.0 {
            return 0.0;
        }
        (self.sum_sq_err / self.sum_sq_ref).sqrt() as f32
    }
}

fn max_code(fmt: Fmt) -> i32 {
    match fmt {
        Fmt::F16 => unreachable!("F16 is not block-quantized"),
        Fmt::Q8 => 127,
        Fmt::Q6 => 31,
        Fmt::Q4 => 7,
        Fmt::Q3 => 3,
    }
}

fn quantize_group(src: &[f32], fmt: Fmt, dst: &mut Vec<u8>) -> (f32, f32) {
    // returns (scale_f32, max_abs_err for this group)
    debug_assert_eq!(src.len(), GROUP);
    let mc = max_code(fmt);
    let mut amax = 0.0f32;
    for &v in src {
        let a = v.abs();
        if a > amax {
            amax = a;
        }
    }
    let d = if amax == 0.0 { 0.0 } else { amax / mc as f32 };
    let inv = if d == 0.0 { 0.0 } else { 1.0 / d };
    dst.extend_from_slice(&f32_to_f16(d).to_le_bytes());

    let mut codes = [0i32; GROUP];
    let mut max_err = 0.0f32;
    for (i, &v) in src.iter().enumerate() {
        let q = ((v * inv).round() as i32).clamp(-mc, mc);
        codes[i] = q;
        let dq = q as f32 * d;
        let e = (v - dq).abs();
        if e > max_err {
            max_err = e;
        }
    }

    match fmt {
        Fmt::F16 => unreachable!(),
        Fmt::Q8 => {
            for &q in &codes {
                dst.push(q as i8 as u8);
            }
        }
        Fmt::Q6 => pack_bits(dst, &codes, 6),
        Fmt::Q4 => {
            for j in 0..16 {
                let lo = codes[2 * j] as u8 & 0x0f;
                let hi = (codes[2 * j + 1] as u8 & 0x0f) << 4;
                dst.push(lo | hi);
            }
        }
        Fmt::Q3 => pack_bits(dst, &codes, 3),
    }
    (d, max_err)
}

fn pack_bits(dst: &mut Vec<u8>, codes: &[i32; GROUP], bits: u32) {
    let total_bits = (GROUP as u32) * bits; // 192 or 96
    let n_bytes = (total_bits / 8) as usize;
    let start = dst.len();
    dst.resize(start + n_bytes, 0);
    for (i, &q) in codes.iter().enumerate() {
        let uq = (q as u32) & ((1u32 << bits) - 1); // two's complement in-field
        let bitpos = (i as u32) * bits;
        let byte_i = start + (bitpos >> 3) as usize;
        let off = bitpos & 7;
        dst[byte_i] |= ((uq << off) & 0xff) as u8;
        if off + bits > 8 {
            dst[byte_i + 1] |= (uq >> (8 - off)) as u8;
            if off + bits > 16 {
                dst[byte_i + 2] |= (uq >> (16 - off)) as u8;
            }
        }
    }
}

fn unpack_bits(src: &[u8], out: &mut [i32; GROUP], bits: u32) {
    for (i, value) in out.iter_mut().enumerate().take(GROUP) {
        let bitpos = (i as u32) * bits;
        let byte_i = (bitpos >> 3) as usize;
        let off = bitpos & 7;
        let mut acc = (src[byte_i] as u32) >> off;
        if off + bits > 8 {
            acc |= (src[byte_i + 1] as u32) << (8 - off);
            if off + bits > 16 {
                acc |= (src[byte_i + 2] as u32) << (16 - off);
            }
        }
        let field = acc & ((1u32 << bits) - 1);
        // sign extend from `bits` width
        let sign_bit = 1u32 << (bits - 1);
        let v = if field & sign_bit != 0 {
            (field | !((1u32 << bits) - 1)) as i32
        } else {
            field as i32
        };
        *value = v;
    }
}

/// Encode `src` (length multiple of GROUP) into R4KV wire bytes.
/// Returns (bytes, worst-case group max_abs_err).
pub fn encode(src: &[f32], fmt: Fmt) -> (Vec<u8>, f32) {
    assert!(fmt != Fmt::F16 || src.len() % 2 == 0);
    match fmt {
        Fmt::F16 => {
            let mut out = Vec::with_capacity(src.len() * 2);
            let mut max_err = 0.0f32;
            for chunk in src.chunks(GROUP) {
                for &v in chunk {
                    let h = f32_to_f16(v);
                    let back = f16_to_f32(h);
                    let e = (v - back).abs();
                    if e > max_err {
                        max_err = e;
                    }
                    out.extend_from_slice(&h.to_le_bytes());
                }
            }
            (out, max_err)
        }
        _ => {
            assert!(src.len() % GROUP == 0, "block formats need len %% 32 == 0");
            let mut out = Vec::with_capacity((src.len() / GROUP) * fmt.group_bytes());
            let mut worst = 0.0f32;
            for g in src.chunks(GROUP) {
                let (_, err) = quantize_group(g, fmt, &mut out);
                if err > worst {
                    worst = err;
                }
            }
            (out, worst)
        }
    }
}

/// Decode R4KV wire bytes back to f32.
pub fn decode(bytes: &[u8], fmt: Fmt, n_elems: usize) -> Vec<f32> {
    assert_eq!(
        bytes.len(),
        fmt.encoded_len(n_elems),
        "wire length mismatch"
    );
    let mut out = vec![0f32; n_elems];
    match fmt {
        Fmt::F16 => {
            for (i, o) in out.iter_mut().enumerate() {
                let b = u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
                *o = f16_to_f32(b);
            }
        }
        _ => {
            let gb = fmt.group_bytes();
            let mc = max_code(fmt);
            let mut codes = [0i32; GROUP];
            for (g, ob) in out.chunks_mut(GROUP).enumerate() {
                let base = g * gb;
                let d = f16_to_f32(u16::from_le_bytes([bytes[base], bytes[base + 1]]));
                let payload = &bytes[base + 2..base + gb];
                match fmt {
                    Fmt::Q8 => {
                        for (j, c) in codes.iter_mut().enumerate() {
                            *c = payload[j] as i8 as i32;
                        }
                    }
                    Fmt::Q6 => unpack_bits(payload, &mut codes, 6),
                    Fmt::Q4 => {
                        for j in 0..16 {
                            let b = payload[j];
                            let lo = (b & 0x0f) as i32;
                            let hi = ((b >> 4) & 0x0f) as i32;
                            codes[2 * j] = if lo >= 8 { lo - 16 } else { lo };
                            codes[2 * j + 1] = if hi >= 8 { hi - 16 } else { hi };
                        }
                        // note: nibble fields are two's complement 4-bit; mask keeps it correct
                        for c in codes.iter_mut().take(32) {
                            *c = (*c as i8 as i32).clamp(-mc, mc);
                        }
                    }
                    Fmt::Q3 => unpack_bits(payload, &mut codes, 3),
                    Fmt::F16 => unreachable!(),
                }
                for (o, &c) in ob.iter_mut().zip(codes.iter()) {
                    *o = c as f32 * d;
                }
            }
        }
    }
    out
}

/// Convenience: encode+decode and report error statistics vs reference.
pub fn roundtrip_stats(src: &[f32], fmt: Fmt) -> (Vec<f32>, GroupStats) {
    let (enc, _) = encode(src, fmt);
    let dec = decode(&enc, fmt, src.len());
    let mut st = GroupStats {
        max_abs_err: 0.0,
        sum_sq_err: 0.0,
        sum_sq_ref: 0.0,
    };
    for (r, d) in src.iter().zip(dec.iter()) {
        let e = (r - d) as f64;
        if e.abs() > st.max_abs_err as f64 {
            st.max_abs_err = e.abs() as f32;
        }
        st.sum_sq_err += e * e;
        st.sum_sq_ref += (*r as f64) * (*r as f64);
    }
    (dec, st)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded(i: usize) -> f32 {
        // deterministic pseudo-random in [-4, 4]
        let x = (i.wrapping_mul(2654435761).wrapping_add(1013904223)) as u32;
        ((x >> 8) as f32 / (1 << 24) as f32) * 8.0 - 4.0
    }

    #[test]
    fn exact_wire_sizes() {
        let data: Vec<f32> = (0..1024).map(seeded).collect();
        for fmt in [Fmt::F16, Fmt::Q8, Fmt::Q6, Fmt::Q4, Fmt::Q3] {
            let (enc, _) = encode(&data, fmt);
            assert_eq!(enc.len(), fmt.encoded_len(1024), "{fmt:?}");
        }
        // row of 1024 elems:
        assert_eq!(Fmt::Q8.encoded_len(1024), 1088); // Q8_0-compatible row bytes
        assert_eq!(Fmt::Q4.encoded_len(1024), 576);
    }

    #[test]
    fn monotonic_error_ordering() {
        let data: Vec<f32> = (0..4096).map(seeded).collect();
        let (_, s_q8) = roundtrip_stats(&data, Fmt::Q8);
        let (_, s_q6) = roundtrip_stats(&data, Fmt::Q6);
        let (_, s_q4) = roundtrip_stats(&data, Fmt::Q4);
        let (_, s_q3) = roundtrip_stats(&data, Fmt::Q3);
        assert!(s_q8.rel_rms() < s_q6.rel_rms());
        assert!(s_q6.rel_rms() < s_q4.rel_rms());
        assert!(s_q4.rel_rms() < s_q3.rel_rms());
        // theoretical worst rel err bounds (sym uniform-ish): sanity ceilings
        assert!(s_q8.rel_rms() < 0.02, "q8 rel rms {}", s_q8.rel_rms());
        assert!(s_q4.rel_rms() < 0.12, "q4 rel rms {}", s_q4.rel_rms());
    }

    #[test]
    fn zero_block_encodes_zero_with_zero_scale() {
        let z = [0f32; 32];
        let (enc, err) = encode(&z, Fmt::Q4);
        assert_eq!(err, 0.0);
        let dec = decode(&enc, Fmt::Q4, 32);
        assert!(dec.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn extreme_values_clamp_not_saturate_wrongly() {
        let mut v = vec![0.0f32; 32];
        v[0] = -9.0;
        v[1] = 9.0;
        let (enc, _) = encode(&v, Fmt::Q3);
        let dec = decode(&enc, Fmt::Q3, 32);
        // with amax=9, d=3, codes clamp to +-3 => reconstructed +-9 exactly at extremes
        assert_eq!(dec[0], -9.0);
        assert_eq!(dec[1], 9.0);
        assert_eq!(dec[2], 0.0);
    }

    #[test]
    fn bit_packing_roundtrip_all_codes_q6_q3() {
        for fmt_bits in [(Fmt::Q6, 6u32), (Fmt::Q3, 3u32)] {
            let (fmt, bits) = fmt_bits;
            let mc = (1i32 << (bits - 1)) - 1;
            let codes: Vec<i32> = (-mc..=mc).cycle().take(32).collect();
            let vals: Vec<f32> = codes.iter().map(|&c| c as f32).collect();
            let (enc, _) = encode(&vals, fmt);
            let dec = decode(&enc, fmt, 32);
            for (v, d) in vals.iter().zip(dec.iter()) {
                assert_eq!(*v, *d, "{fmt:?}");
            }
        }
    }

    #[test]
    fn f16_roundtrip_error_bounded() {
        let data: Vec<f32> = (0..4096).map(seeded).collect();
        let (_, st) = roundtrip_stats(&data, Fmt::F16);
        assert!(st.max_abs_err <= 0.005, "f16 max err {}", st.max_abs_err);
    }
}
