//! Prefix reuse / eviction telemetry — the observation layer for
//! residency tuning (design: `research/2026-08-15-prefix-statistics-design.md`).
//!
//! Pure counters, no allocation in the hot path, no behavior change:
//! telemetry only *observes* admission and eviction.  Its consumers are
//! policy decisions (page granularity, eviction policy, pin relaxation)
//! and honest reporting — never performance claims (BENCHMARK_POLICY).

/// Depth buckets for the hit-depth histogram.
pub const BUCKETS: [usize; 5] = [0, 8, 32, 128, usize::MAX];

/// Bucket index for a matched depth (0 → cold, 1 → 1..=8, 2 → 9..=32,
/// 3 → 33..=128, 4 → 129+).
pub fn depth_bucket(depth: usize) -> usize {
    match depth {
        0 => 0,
        1..=8 => 1,
        9..=32 => 2,
        33..=128 => 3,
        _ => 4,
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PrefixTelemetry {
    /// Sequences admitted.
    pub admissions: u64,
    /// Admission whose matched node had resident state.
    pub cache_hits: u64,
    /// Admission whose matched node's state was evicted (cold fallback).
    pub cache_misses: u64,
    /// Matched depth == prompt length (first token sampled free).
    pub full_prompt_hits: u64,
    /// Sum of matched depths across hits.
    pub hit_depth_total: u64,
    /// Hit-depth histogram (bucket per `depth_bucket`).
    pub hit_depth_buckets: [u64; 5],
    /// Rows of model work saved by prefix reuse (sum of matched depths).
    pub reuse_rows_saved: u64,
    /// Sum of eviction ages (steps since last_access at eviction).
    pub eviction_age_total: u64,
    /// States evicted from the residency pool.
    pub evictions: u64,
    /// Sweep iterations that could not evict because everything was pinned.
    pub pinned_evictions_skipped: u64,
    /// Steps where the sweep ended still over budget (pool too small or
    /// fully pinned).
    pub over_budget_steps: u64,
}

impl PrefixTelemetry {
    pub fn hit_rate(&self) -> f64 {
        if self.admissions == 0 {
            return 0.0;
        }
        self.cache_hits as f64 / self.admissions as f64
    }

    pub fn mean_reuse_depth(&self) -> f64 {
        if self.cache_hits == 0 {
            return 0.0;
        }
        self.hit_depth_total as f64 / self.cache_hits as f64
    }

    pub fn mean_eviction_age(&self) -> f64 {
        if self.evictions == 0 {
            return 0.0;
        }
        self.eviction_age_total as f64 / self.evictions as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_boundaries() {
        assert_eq!(depth_bucket(0), 0);
        assert_eq!(depth_bucket(1), 1);
        assert_eq!(depth_bucket(8), 1);
        assert_eq!(depth_bucket(9), 2);
        assert_eq!(depth_bucket(32), 2);
        assert_eq!(depth_bucket(33), 3);
        assert_eq!(depth_bucket(128), 3);
        assert_eq!(depth_bucket(129), 4);
    }

    #[test]
    fn derived_metrics_are_sane() {
        let t = PrefixTelemetry {
            admissions: 2,
            cache_hits: 1,
            hit_depth_total: 40,
            evictions: 4,
            eviction_age_total: 12,
            ..Default::default()
        };
        assert_eq!(t.hit_rate(), 0.5);
        assert_eq!(t.mean_reuse_depth(), 40.0);
        assert_eq!(t.mean_eviction_age(), 3.0);
        assert_eq!(PrefixTelemetry::default().hit_rate(), 0.0);
    }
}
