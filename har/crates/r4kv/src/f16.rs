//! IEEE 754 binary16 <-> binary32 conversion, round-to-nearest-even.
//! Dependency-free so the reference implementation is trivially portable.

pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;

    let out = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            // subnormal half: value = frac * 2^-24; renormalize
            let t = 31 - frac.leading_zeros(); // highest set bit idx
            let exp32 = 127 + t as i32 - 24;
            let mantissa = (frac << (23 - t)) & 0x007f_ffff;
            sign | (((exp32 as u32) & 0xff) << 23) | mantissa
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13) // inf / nan
    } else {
        sign | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(out)
}

pub fn f32_to_f16(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let aexp = ((x >> 23) & 0xff) as i32;
    let frac = x & 0x007f_ffff;

    if aexp == 0xff {
        // inf/nan preserve
        return sign | 0x7c00 | if frac != 0 { 0x0200 } else { 0 };
    }

    let e = aexp - 127;
    let mant = frac | 0x0080_0000; // implicit bit

    if e > 15 {
        return sign | 0x7c00; // overflow -> inf
    }
    if e >= -14 {
        // normal: drop low 13 mantissa bits with RTNE
        let exp_h = (e + 15) as u32;
        let keep = mant >> 13;
        let rem = mant & 0x1fff;
        let halfway = 0x1000;
        let mut r = keep;
        if rem > halfway || (rem == halfway && (keep & 1) == 1) {
            r += 1;
        }
        if r >= 0x800 {
            // mantissa carried into exponent
            let eo = exp_h + 1;
            if eo >= 0x1f {
                return sign | 0x7c00;
            }
            return sign | ((eo as u16) << 10);
        }
        return sign | ((exp_h as u16) << 10) | ((r & 0x3ff) as u16);
    }
    // subnormal: representable unit is 2^-24; value = mant * 2^(e-23)
    if aexp == 0 && frac == 0 {
        return sign; // true zero (incl. -0.0)
    }
    let sh = (-(1 + e)) as i64; // e <= -15 => sh >= 14
    if sh >= 31 {
        // far below half of one unit: cannot round up
        return sign;
    }
    let m64 = mant as u64;
    let whole = (m64 >> sh) as u32;
    let rem = m64 & ((1u64 << sh) - 1);
    let halfway = 1u64 << (sh - 1);
    let mut kept = whole;
    if rem > halfway || (rem == halfway && (whole & 1) == 1) {
        kept += 1;
    }
    if kept >= 0x400 {
        // rounded up to minimum normal
        return sign | (1 << 10);
    }
    sign | (kept as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_values() {
        assert_eq!(f32_to_f16(0.0), 0);
        assert_eq!(f16_to_f32(0), 0.0);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f32_to_f16(2.0), 0x4000);
        assert_eq!(f32_to_f16(0.5), 0x3800);
        assert_eq!(f32_to_f16(-1.5), 0xbe00);
    }

    #[test]
    fn rtne_halfway() {
        // 1.0 + 2^-11 is exactly halfway between 1.0 and 1+2^-10 in f16
        let halfway = 1.0 + f32::from_bits(((127 - 11) as u32) << 23);
        assert_eq!(f32_to_f16(halfway), 0x3c00); // ties to even
        let just_above = 1.0 + f32::from_bits((((127 - 11) + 1) as u32) << 23);
        assert_eq!(f32_to_f16(just_above), 0x3c01);
    }

    #[test]
    fn roundtrip_small_set() {
        for i in -20..=20 {
            let v = i as f32 * 0.25;
            let h = f32_to_f16(v);
            assert!((f16_to_f32(h) - v).abs() < 1e-6);
        }
    }

    #[test]
    fn subnormal_roundtrip() {
        // 6 * 2^-24 = exactly 6 half-subnormal units
        let tiny = 6.0f32 * 2f32.powi(-24);
        assert_eq!(f32_to_f16(tiny), 6);
        assert_eq!(f16_to_f32(6), tiny);
        // just below 0.5 unit rounds to zero; just above rounds to 1 unit
        let half = 0.5f32 * 2f32.powi(-24);
        assert_eq!(f32_to_f16(half - 2f32.powi(-30)), 0);
        assert_eq!(f32_to_f16(half), 0); // ties to even
        assert_eq!(f32_to_f16(half + 2f32.powi(-30)), 1);
    }

    #[test]
    fn known_scale_values() {
        // scales produced by quantization must round-trip exactly
        for &d in &[3.0f32, 0.57, 1.0 / 31.0, 9.0] {
            let h = f32_to_f16(d);
            let back = f16_to_f32(h);
            assert!((back - d).abs() <= d * 5e-4 + 1e-9, "{d} -> {back}");
        }
    }
}
