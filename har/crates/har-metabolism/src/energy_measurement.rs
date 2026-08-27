//! Physical energy measurement sources.
//!
//! REMORA-08 defines `EnergyScope::Unknown` as the default until a real
//! energy counter exists.  This module provides the first real measurement
//! path: integrating GPU package power from Linux hwmon sysfs over elapsed
//! compute time to estimate joules.
//!
//! The measurement is an ESTIMATE (not a hardware energy counter) because
//! some AMD devices expose package power but not cumulative energy. The
//! integration is:
//!
//! ```text
//! joules = ∫ power(t) dt ≈ Σ power_i * Δt_i
//! ```
//!
//! Under `EnergyScope::Estimated` the fields are valid but annotated as
//! estimated; under `EnergyScope::Unknown` they remain `None`.

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::energy::EnergyScope;

/// A sample of GPU package power at a point in time.
///
/// Uses `u64` nanoseconds since Unix epoch for serializable timestamps;
/// `Instant` is not serializable so we use `SystemTime`-based epochs

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PowerSample {
    /// Milliwatts read from the hardware sensor.
    pub milliwatts: u64,
    /// Nanoseconds since UNIX_EPOCH.
    pub timestamp_ns: u64,
}

impl PowerSample {
    pub fn new(milliwatts: u64) -> Self {
        Self {
            milliwatts,
            timestamp_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
        }
    }
}

/// Integrate a sequence of power samples to estimate consumed energy.
///
/// Returns `(estimated_joules, sample_count)`.  The integration is a
/// simple trapezoidal rule over consecutive samples; if only one sample
/// is provided the energy is reported as `0` and the caller must decide
/// whether to attribute it to a window or skip it.
pub fn integrate_samples(samples: &[PowerSample]) -> (f64, u64) {
    if samples.len() < 2 {
        return (0.0, samples.len() as u64);
    }
    let mut joules = 0.0f64;
    let mut count = 0u64;
    for i in 1..samples.len() {
        let dt_ns = samples[i]
            .timestamp_ns
            .saturating_sub(samples[i - 1].timestamp_ns);
        let dt = dt_ns as f64 / 1_000_000_000.0;
        let p_avg = (samples[i - 1].milliwatts + samples[i].milliwatts) as f64 / 2.0;
        joules += p_avg * dt / 1000.0; // mW * s -> J
        count += 1;
    }
    (joules, count)
}

/// Reads GPU package power from the Linux hwmon sysfs interface.
///
/// The production runtime does not spawn a vendor utility. A missing or
/// inaccessible sensor is represented as `None`; callers must not interpret
/// that as zero power.
///
/// Returns `Some(milliwatts)` on success, `None` if no sensor is
/// reachable.  Callers should treat `None` as "not measurable right now"
/// rather than "zero power".
pub fn read_gpu_power_milliwatts() -> Option<u64> {
    read_hwmon_power()
}

fn read_hwmon_power() -> Option<u64> {
    let base = "/sys/class/hwmon";
    let mut fallback = None;
    if let Ok(entries) = std::fs::read_dir(base) {
        for entry in entries.flatten() {
            let path = entry.path();
            let power_file = path.join("power1_average");
            if power_file.exists() {
                if let Ok(bytes) = std::fs::read_to_string(power_file) {
                    if let Ok(mw) = bytes.trim().parse::<u64>() {
                        // hwmon reports in microwatts
                        let value = mw / 1000;
                        let name = std::fs::read_to_string(path.join("name"))
                            .unwrap_or_default()
                            .to_ascii_lowercase();
                        if name.contains("amdgpu") || name.contains("gpu") {
                            return Some(value);
                        }
                        fallback.get_or_insert(value);
                    }
                }
            }
        }
    }
    fallback
}

/// Tracks power samples over time and computes running energy estimates.
#[derive(Debug, Default)]
pub struct EnergyTracker {
    samples: Vec<PowerSample>,
    max_samples: usize,
    total_joules: f64,
    sample_count: u64,
}

impl EnergyTracker {
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: Vec::with_capacity(max_samples),
            max_samples,
            total_joules: 0.0,
            sample_count: 0,
        }
    }

    /// Add a new sample (either manually or via `read_gpu_power_milliwatts`).
    pub fn add_sample(&mut self, sample: PowerSample) {
        // Always keep the last `max_samples` for recent integration.
        if self.samples.len() >= self.max_samples {
            self.samples.remove(0);
        }
        self.samples.push(sample);
        // Recompute integrated energy from all samples.
        let (joules, count) = integrate_samples(&self.samples);
        self.total_joules = joules;
        self.sample_count = count;
    }

    /// Returns the estimated energy in joules from all integrated samples.
    pub fn estimated_joules(&self) -> f64 {
        self.total_joules
    }

    /// Returns the number of integration intervals completed.
    pub fn interval_count(&self) -> u64 {
        self.sample_count
    }

    /// Returns the number of samples currently tracked.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

/// Estimates joules per exact token from the tracker and token count.
///
/// Returns `None` if there are no integration intervals or no tokens.
pub fn joules_per_exact_token(estimated_joules: f64, exact_tokens: u64) -> Option<f64> {
    if exact_tokens == 0 || estimated_joules == 0.0 {
        return None;
    }
    Some(estimated_joules / exact_tokens as f64)
}

/// Builds an `EnergyLabel` with `EnergyScope::Estimated` from a tracker
/// and token count.  This is the first real energy path for REMORA — it
/// uses power integration rather than a hardware energy counter, so the
/// scope is explicitly `Estimated` rather than `GpuOnly`.
pub fn build_estimated_energy_label(
    tracker: &EnergyTracker,
    exact_tokens: u64,
) -> crate::energy::EnergyLabel {
    let joules = tracker.estimated_joules();
    let jpt = joules_per_exact_token(joules, exact_tokens);

    // We can't compute energy_delay_product without throughput data,
    // so leave it as None until EDP measurement is implemented.
    crate::energy::EnergyLabel {
        scope: EnergyScope::Estimated,
        joules_per_exact_token: jpt,
        joules_per_accepted_token: None,
        energy_delay_product: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrate_two_samples() {
        let s1 = PowerSample {
            milliwatts: 1000,
            timestamp_ns: 1_000_000_000,
        };
        let s2 = PowerSample {
            milliwatts: 2000,
            timestamp_ns: 2_000_000_000,
        };
        let (joules, count) = integrate_samples(&[s1, s2]);
        assert!((joules - 1.5).abs() < 0.01);
        assert_eq!(count, 1);
    }

    #[test]
    fn integrate_three_samples_linear() {
        let s1 = PowerSample {
            milliwatts: 1000,
            timestamp_ns: 1_000_000_000,
        };
        let s2 = PowerSample {
            milliwatts: 2000,
            timestamp_ns: 1_500_000_000,
        };
        let s3 = PowerSample {
            milliwatts: 3000,
            timestamp_ns: 2_000_000_000,
        };
        let (joules, count) = integrate_samples(&[s1, s2, s3]);
        assert!((joules - 2.0).abs() < 0.01);
        assert_eq!(count, 2);
    }

    #[test]
    fn joules_per_token() {
        let jpt = joules_per_exact_token(10.0, 5).unwrap();
        assert!((jpt - 2.0).abs() < 0.001);
    }

    #[test]
    fn joules_per_token_zero_tokens() {
        assert!(joules_per_exact_token(10.0, 0).is_none());
    }

    #[test]
    fn tracker_accumulates() {
        let mut tracker = EnergyTracker::new(10);
        tracker.add_sample(PowerSample {
            milliwatts: 1000,
            timestamp_ns: 1_000_000_000,
        });
        tracker.add_sample(PowerSample {
            milliwatts: 2000,
            timestamp_ns: 2_000_000_000,
        });
        assert!((tracker.estimated_joules() - 1.5).abs() < 0.01);
        assert_eq!(tracker.interval_count(), 1);
        assert_eq!(tracker.sample_count(), 2);
    }
}
