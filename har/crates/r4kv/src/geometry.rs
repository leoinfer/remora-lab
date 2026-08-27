//! Context-level byte tables for the target model geometry.
//! Constants from the reviewed KV geometry record.

use crate::profiles::{Profile, N_KV_LAYERS_MAIN, N_MTP_KV_LAYERS};

pub const NATIVE_CONTEXT_LENGTH: usize = 262144;

/// Reference context sizes used across the KV benchmark matrix (directive §30).
pub const CTX_SWEEP: [usize; 7] = [4096, 8192, 16384, 32768, 65536, 131072, 262144];

#[derive(Clone, Debug)]
pub struct CtxRow {
    pub ctx_tokens: usize,
    pub total_bytes: f64,
    pub mib: f64,
}

/// Table of context -> bytes for a profile.
pub fn context_table(p: Profile) -> Vec<CtxRow> {
    CTX_SWEEP
        .iter()
        .map(|&c| {
            let b = p.context_bytes(c, N_KV_LAYERS_MAIN, N_MTP_KV_LAYERS);
            CtxRow {
                ctx_tokens: c,
                total_bytes: b,
                mib: b / (1024.0 * 1024.0),
            }
        })
        .collect()
}

/// Largest plain-decode context feasible inside a VRAM budget (bytes),
/// ignoring weights/buffers — the pure upper bound.
pub fn max_ctx_within_budget(p: Profile, vram_budget_bytes: f64) -> usize {
    let bpt = p.bytes_per_token(N_KV_LAYERS_MAIN, N_MTP_KV_LAYERS);
    ((vram_budget_bytes / bpt).floor()) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_262k_f16_is_16gib_main_cache() {
        // main decoder cache only (16 layers): 16 GiB exactly
        let b = Profile::RefA.context_bytes(262144, N_KV_LAYERS_MAIN, 0);
        assert!((b - 17179869184.0).abs() < 1.0);
    }

    #[test]
    fn q8_capture_reconciliation() {
        // exact capture: 263 tokens, 16 layers, q8 rows 1088+1088
        let per_tok = Profile::BaseQ8.bytes_per_token(16, 0);
        assert_eq!(263.0 * per_tok, 9156608.0);
    }

    #[test]
    fn feasible_context_doubles_as_bytes_halve() {
        let budget = 2_147_483_648.0; // 2 GiB
        let q8 = max_ctx_within_budget(Profile::BaseQ8, budget);
        let k4v4 = max_ctx_within_budget(Profile::PK4V4, budget);
        // PK4V4 bpt = 19,584; BaseQ8 bpt = 36,992 -> ratio ~1.889
        assert!((k4v4 as f64 / q8 as f64 - 1.89).abs() < 0.05);
    }
}
