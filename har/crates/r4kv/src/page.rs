//! Exact prefix-bound KV page container (T1 lukewarm tier).
//!
//! A page is the unit of VRAM<->RAM movement. It stores packed R4KV codes for
//! a contiguous token range of one or more attention layers, together with
//! enough identity to FAIL CLOSED on any mismatch (see HAR_PREFIX_KV_SPEC.md
//! identity contract; this header binds into it via `prefix_digest`).

use crate::crc32c::crc32c;
use crate::Fmt;

pub const MAGIC: u32 = 0x5234_4B56; // "R4KV"
pub const VERSION: u16 = 1;
pub const HEADER_LEN: usize = 96;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageHeader {
    pub prefix_digest: u64,
    pub token_start: u32,
    pub token_count: u32,
    pub pos_start: u32,
    pub layer_lo: u16,
    pub layer_hi: u16,
    pub k_fmt: Fmt,
    pub v_fmt: Fmt,
    pub epoch: u32,
    pub generation: u32,
    pub payload_len: u64,
    pub sketch_offset: u32, // 0 => absent
    pub flags: u16,         // bit0 has_sketch, bit1 rotated_k
}

/// Identity expected by the live stream when restoring a page.
#[derive(Clone)]
pub struct RestoreExpectation<'a> {
    pub prefix_digest: u64,
    pub token_start: u32,
    pub pos_start: u32,
    /// inclusive layer range needed
    pub layer_lo: u16,
    pub layer_hi: u16,
    /// accepted formats (None = any)
    pub k_fmt: Option<Fmt>,
    pub v_fmt: Option<Fmt>,
    /// optional strictness on epoch/generation
    pub generation: Option<&'a dyn Fn(u32) -> bool>,
}

impl std::fmt::Debug for RestoreExpectation<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RestoreExpectation")
            .field("prefix_digest", &self.prefix_digest)
            .field("token_start", &self.token_start)
            .field("pos_start", &self.pos_start)
            .field("layer_range", &(self.layer_lo, self.layer_hi))
            .field("k_fmt", &self.k_fmt)
            .field("v_fmt", &self.v_fmt)
            .field(
                "generation_predicate",
                &self.generation.map(|_| "<predicate>"),
            )
            .finish()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum RestoreError {
    BadMagic,
    BadVersion,
    HeaderChecksumMismatch,
    PayloadChecksumMismatch,
    LengthMismatch { declared: u64, actual: usize },
    PrefixMismatch { want: u64, got: u64 },
    TokenRangeMismatch { want_start: u32, got_start: u32 },
    PosMismatch { want: u32, got: u32 },
    LayerRangeInsufficient { want: (u16, u16), got: (u16, u16) },
    FormatRefused { field: &'static str },
    GenerationRejected,
}

fn put_u16(b: &mut [u8], off: usize, v: u16) {
    b[off..off + 2].copy_from_slice(&v.to_le_bytes());
}
fn put_u32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}
fn put_u64(b: &mut [u8], off: usize, v: u64) {
    b[off..off + 8].copy_from_slice(&v.to_le_bytes());
}
fn get_u16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}
fn get_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}
fn get_u64(b: &[u8], off: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(a)
}

impl PageHeader {
    /// Serialize the fixed 96-byte little-endian header with trailing CRC32C.
    pub fn to_bytes(&self) -> [u8; HEADER_LEN] {
        let mut b = [0u8; HEADER_LEN];
        put_u32(&mut b, 0, MAGIC);
        put_u16(&mut b, 4, VERSION);
        put_u16(&mut b, 6, self.flags);
        put_u64(&mut b, 8, self.prefix_digest);
        put_u32(&mut b, 16, self.token_start);
        put_u32(&mut b, 20, self.token_count);
        put_u32(&mut b, 24, self.pos_start);
        put_u16(&mut b, 28, self.layer_lo);
        put_u16(&mut b, 30, self.layer_hi);
        b[32] = self.k_fmt as u8;
        b[33] = self.v_fmt as u8;
        put_u32(&mut b, 34, self.epoch);
        put_u32(&mut b, 38, self.generation);
        put_u64(&mut b, 42, self.payload_len);
        put_u32(&mut b, 50, self.sketch_offset);
        // 54..92 reserved zeros
        let crc = crc32c(&b[0..92]);
        put_u32(&mut b, 92, crc);
        b
    }

    pub fn from_bytes(b: &[u8; HEADER_LEN]) -> Result<PageHeader, RestoreError> {
        if get_u32(b, 0) != MAGIC {
            return Err(RestoreError::BadMagic);
        }
        if get_u16(b, 4) != VERSION {
            return Err(RestoreError::BadVersion);
        }
        let stored_crc = get_u32(b, 92);
        if crc32c(&b[0..92]) != stored_crc {
            return Err(RestoreError::HeaderChecksumMismatch);
        }
        Ok(PageHeader {
            flags: get_u16(b, 6),
            prefix_digest: get_u64(b, 8),
            token_start: get_u32(b, 16),
            token_count: get_u32(b, 20),
            pos_start: get_u32(b, 24),
            layer_lo: get_u16(b, 28),
            layer_hi: get_u16(b, 30),
            k_fmt: Fmt::from_u8(b[32]).ok_or(RestoreError::FormatRefused { field: "k_fmt" })?,
            v_fmt: Fmt::from_u8(b[33]).ok_or(RestoreError::FormatRefused { field: "v_fmt" })?,
            epoch: get_u32(b, 34),
            generation: get_u32(b, 38),
            payload_len: get_u64(b, 42),
            sketch_offset: get_u32(b, 50),
        })
    }

    /// Fail-closed legality gate. Every mismatch is an error; nothing is
    /// silently coerced. Callers pass the payload *body* (CRC already
    /// verified by `unpack_page`).
    pub fn check_restore(
        &self,
        exp: &RestoreExpectation<'_>,
        payload_body: &[u8],
    ) -> Result<(), RestoreError> {
        if payload_body.len() as u64 != self.payload_len {
            return Err(RestoreError::LengthMismatch {
                declared: self.payload_len,
                actual: payload_body.len(),
            });
        }
        if self.prefix_digest != exp.prefix_digest {
            return Err(RestoreError::PrefixMismatch {
                want: exp.prefix_digest,
                got: self.prefix_digest,
            });
        }
        if self.token_start != exp.token_start {
            return Err(RestoreError::TokenRangeMismatch {
                want_start: exp.token_start,
                got_start: self.token_start,
            });
        }
        if self.pos_start != exp.pos_start {
            return Err(RestoreError::PosMismatch {
                want: exp.pos_start,
                got: self.pos_start,
            });
        }
        let got = (self.layer_lo, self.layer_hi);
        let want = (exp.layer_lo, exp.layer_hi);
        if got.0 > want.0 || got.1 < want.1 {
            return Err(RestoreError::LayerRangeInsufficient { want, got });
        }
        if let Some(kf) = exp.k_fmt {
            if kf != self.k_fmt {
                return Err(RestoreError::FormatRefused { field: "k_fmt" });
            }
        }
        if let Some(vf) = exp.v_fmt {
            if vf != self.v_fmt {
                return Err(RestoreError::FormatRefused { field: "v_fmt" });
            }
        }
        if let Some(g) = exp.generation {
            if !g(self.generation) {
                return Err(RestoreError::GenerationRejected);
            }
        }
        Ok(())
    }
}

/// Pack a full page: header + payload (+ trailing payload CRC32C).
pub fn pack_page(h: &PageHeader, payload_body: &[u8]) -> Vec<u8> {
    let crc = crc32c(payload_body);
    let mut payload = Vec::with_capacity(payload_body.len() + 4);
    payload.extend_from_slice(payload_body);
    payload.extend_from_slice(&crc.to_le_bytes());
    debug_assert_eq!(payload.len() as u64, h.payload_len + 4);
    let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
    out.extend_from_slice(&h.to_bytes());
    out.extend_from_slice(&payload);
    out
}

/// Unpack: parse + verify header and payload checksums.
pub fn unpack_page(buf: &[u8]) -> Result<(PageHeader, Vec<u8>), RestoreError> {
    if buf.len() < HEADER_LEN + 4 {
        return Err(RestoreError::LengthMismatch {
            declared: 0,
            actual: buf.len(),
        });
    }
    let mut hb = [0u8; HEADER_LEN];
    hb.copy_from_slice(&buf[..HEADER_LEN]);
    let h = PageHeader::from_bytes(&hb)?;
    let body_end = buf.len() - 4;
    let body = &buf[HEADER_LEN..body_end];
    if body.len() as u64 != h.payload_len {
        return Err(RestoreError::LengthMismatch {
            declared: h.payload_len,
            actual: body.len(),
        });
    }
    let stored = u32::from_le_bytes([
        buf[body_end],
        buf[body_end + 1],
        buf[body_end + 2],
        buf[body_end + 3],
    ]);
    if crc32c(body) != stored {
        return Err(RestoreError::PayloadChecksumMismatch);
    }
    Ok((h, body.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_header() -> PageHeader {
        PageHeader {
            prefix_digest: 0xDEADBEEF12345678,
            token_start: 4096,
            token_count: 512,
            pos_start: 4096,
            layer_lo: 3,
            layer_hi: 63,
            k_fmt: Fmt::Q6,
            v_fmt: Fmt::Q4,
            epoch: 7,
            generation: 42,
            payload_len: 1024,
            sketch_offset: 96 + 1024,
            flags: 0,
        }
    }

    #[test]
    fn header_roundtrip_and_checksum_gate() {
        let h = sample_header();
        let bytes = h.to_bytes();
        assert_eq!(bytes.len(), HEADER_LEN);
        let back = PageHeader::from_bytes(&bytes).unwrap();
        assert_eq!(back, h);

        let mut corrupted = bytes;
        corrupted[20] ^= 0x01; // flip token_start bit
        assert_eq!(
            PageHeader::from_bytes(&corrupted),
            Err(RestoreError::HeaderChecksumMismatch)
        );
    }

    #[test]
    fn page_roundtrip_with_payload_crc() {
        let mut h = sample_header();
        let body: Vec<u8> = (0..1024).map(|i| (i * 31 % 251) as u8).collect();
        h.payload_len = body.len() as u64;
        let page = pack_page(&h, &body);
        let (h2, body2) = unpack_page(&page).unwrap();
        assert_eq!(h2, h);
        assert_eq!(body2, body);

        let mut corrupted = page.clone();
        let n = corrupted.len();
        corrupted[n - 10] ^= 0xff;
        assert!(matches!(
            unpack_page(&corrupted),
            Err(RestoreError::PayloadChecksumMismatch)
        ));
    }

    #[test]
    fn restore_gate_fails_closed_on_every_axis() {
        let h = sample_header();
        let payload = vec![0u8; 1024];
        let ok = RestoreExpectation {
            prefix_digest: h.prefix_digest,
            token_start: h.token_start,
            pos_start: h.pos_start,
            layer_lo: 3,
            layer_hi: 63,
            k_fmt: None,
            v_fmt: None,
            generation: None,
        };
        assert_eq!(h.check_restore(&ok, &payload), Ok(()));

        // wrong prefix
        let e = RestoreExpectation {
            prefix_digest: 1,
            ..ok
        };
        assert!(matches!(
            h.check_restore(&e, &payload),
            Err(RestoreError::PrefixMismatch { .. })
        ));

        // wrong token range
        let e = RestoreExpectation {
            token_start: 999,
            ..ok
        };
        assert!(matches!(
            h.check_restore(&e, &payload),
            Err(RestoreError::TokenRangeMismatch { .. })
        ));

        // insufficient layer coverage
        let e = RestoreExpectation { layer_hi: 99, ..ok };
        assert!(matches!(
            h.check_restore(&e, &payload),
            Err(RestoreError::LayerRangeInsufficient { .. })
        ));

        // refused format
        let e = RestoreExpectation {
            v_fmt: Some(Fmt::F16),
            ..ok
        };
        assert!(matches!(
            h.check_restore(&e, &payload),
            Err(RestoreError::FormatRefused { field: "v_fmt" })
        ));

        // rejected generation predicate
        let e = RestoreExpectation {
            generation: Some(&|g: u32| g == 41),
            ..ok
        };
        assert!(matches!(
            h.check_restore(&e, &payload),
            Err(RestoreError::GenerationRejected)
        ));
    }
}
