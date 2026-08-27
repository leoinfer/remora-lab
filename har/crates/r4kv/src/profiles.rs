//! Named K/V precision profiles and their exact byte economics.
//! Code is authority; prose tables in the research notes are derived from
//! here.

use crate::{Fmt, ROW_ELEMS};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Profile {
    /// Reference oracle: f16 K + f16 V.
    RefA,
    /// Current standard: q8_0-equivalent symmetric blocks.
    BaseQ8,
    /// Frontier candidate 1: K q6 / V q4.
    PK6V4,
    /// Frontier candidate 2: K q4 / V q4.
    PK4V4,
    /// Aggressive edge: K q4 / V q3.
    PK4V3,
}

impl Profile {
    pub fn k_fmt(self) -> Fmt {
        match self {
            Profile::RefA => Fmt::F16,
            Profile::BaseQ8 => Fmt::Q8,
            Profile::PK6V4 => Fmt::Q6,
            Profile::PK4V4 | Profile::PK4V3 => Fmt::Q4,
        }
    }

    pub fn v_fmt(self) -> Fmt {
        match self {
            Profile::RefA => Fmt::F16,
            Profile::BaseQ8 => Fmt::Q8,
            Profile::PK6V4 | Profile::PK4V4 => Fmt::Q4,
            Profile::PK4V3 => Fmt::Q3,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Profile::RefA => "REF-A_f16f16",
            Profile::BaseQ8 => "BASE-Q8",
            Profile::PK6V4 => "P-K6V4",
            Profile::PK4V4 => "P-K4V4",
            Profile::PK4V3 => "P-K4V3",
        }
    }

    /// Exact KV bytes per token for `n_kv_layers` attention layers
    /// (+ `n_mtp_layers` optional extra dense-attn draft block).
    pub fn bytes_per_token(self, n_kv_layers: usize, n_mtp_layers: usize) -> f64 {
        let per_layer =
            ROW_ELEMS as f64 * (self.k_fmt().bytes_per_elem() + self.v_fmt().bytes_per_elem());
        per_layer * (n_kv_layers as f64 + n_mtp_layers as f64)
    }

    /// Total KV bytes for a context of `tokens` tokens.
    pub fn context_bytes(self, tokens: usize, n_kv_layers: usize, n_mtp_layers: usize) -> f64 {
        self.bytes_per_token(n_kv_layers, n_mtp_layers) * tokens as f64
    }
}

/// Default geometry constants for the public accounting examples.
pub const N_KV_LAYERS_MAIN: usize = 16;
pub const N_MTP_KV_LAYERS: usize = 1;

pub const ALL_PROFILES: [Profile; 5] = [
    Profile::RefA,
    Profile::BaseQ8,
    Profile::PK6V4,
    Profile::PK4V4,
    Profile::PK4V3,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_q8_matches_documented_geometry() {
        // 16 layers with 1024 elements per K/V row gives the following
        // deterministic byte counts for the public accounting example.
        let p = Profile::BaseQ8;
        assert!((p.bytes_per_token(16, 0) - 34816.0).abs() < 1e-9);
        assert!((p.bytes_per_token(16, 1) - 36992.0).abs() < 1e-9);
        // f16 reference: 65,536 main / 69,632 with MTP (17 attention layers x 4096 B)
        assert!((Profile::RefA.bytes_per_token(16, 0) - 65536.0).abs() < 1e-9);
        assert!((Profile::RefA.bytes_per_token(16, 1) - 69632.0).abs() < 1e-9);
    }

    #[test]
    fn frontier_profiles_cut_bytes_monotonically() {
        let bpt = |p: Profile| p.bytes_per_token(N_KV_LAYERS_MAIN, N_MTP_KV_LAYERS);
        assert!(bpt(Profile::BaseQ8) > bpt(Profile::PK6V4));
        assert!(bpt(Profile::PK6V4) > bpt(Profile::PK4V4));
        assert!(bpt(Profile::PK4V4) > bpt(Profile::PK4V3));
    }

    #[test]
    fn k4v4_saves_about_half_vs_q8() {
        let ratio = Profile::PK4V4.bytes_per_token(16, 1) / Profile::BaseQ8.bytes_per_token(16, 1);
        assert!((ratio - 0.5263).abs() < 0.01, "ratio {ratio}");
    }
}
