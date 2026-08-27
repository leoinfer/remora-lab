//! R4KV — HAR custom compressed KV cache format.
//!
//! CPU reference codec. Authority order: this crate's tests > prose documents.
//! Design contracts are summarized in the public R4KV format brief; the byte
//! layouts below are normative for this candidate.

pub mod capture;
pub mod crc32c;
pub mod dma;
pub mod f16;
pub mod geometry;
pub mod page;
pub mod profiles;
pub mod quant;
pub mod tier;

/// Elements per KV row for the target model family (4 kv-heads x 256 dims).
pub const ROW_ELEMS: usize = 1024;
/// Quantization group size shared by all R4KV block formats.
pub const GROUP: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Fmt {
    F16 = 0,
    Q8 = 1,
    Q6 = 2,
    Q4 = 3,
    Q3 = 4,
}

impl Fmt {
    pub fn from_u8(v: u8) -> Option<Fmt> {
        Some(match v {
            0 => Fmt::F16,
            1 => Fmt::Q8,
            2 => Fmt::Q6,
            3 => Fmt::Q4,
            4 => Fmt::Q3,
            _ => return None,
        })
    }

    /// Bytes per 32-element group as encoded on wire (scale included).
    pub fn group_bytes(self) -> usize {
        match self {
            Fmt::F16 => 64,
            Fmt::Q8 => 34,
            Fmt::Q6 => 26,
            Fmt::Q4 => 18,
            Fmt::Q3 => 14,
        }
    }

    /// Exact bytes per element (wire cost).
    pub fn bytes_per_elem(self) -> f64 {
        self.group_bytes() as f64 / GROUP as f64
    }

    /// Encoded size in bytes for `n` elements (n must be a multiple of GROUP
    /// for block formats; F16 accepts any even count).
    pub fn encoded_len(self, n: usize) -> usize {
        match self {
            Fmt::F16 => n * 2,
            _ => (n / GROUP) * self.group_bytes(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_byte_sizes_match_spec() {
        assert_eq!(Fmt::F16.group_bytes(), 64);
        assert_eq!(Fmt::Q8.group_bytes(), 34);
        assert_eq!(Fmt::Q6.group_bytes(), 26);
        assert_eq!(Fmt::Q4.group_bytes(), 18);
        assert_eq!(Fmt::Q3.group_bytes(), 14);
    }

    #[test]
    fn fmt_ids_are_stable() {
        // Wire format ids must never be renumbered; pages persist them.
        assert_eq!(Fmt::F16 as u8, 0);
        assert_eq!(Fmt::Q8 as u8, 1);
        assert_eq!(Fmt::Q6 as u8, 2);
        assert_eq!(Fmt::Q4 as u8, 3);
        assert_eq!(Fmt::Q3 as u8, 4);
        assert_eq!(Fmt::from_u8(5), None);
    }
}
