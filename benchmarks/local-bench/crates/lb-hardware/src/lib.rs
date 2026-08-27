//! # lb-hardware
//!
//! Hardware phenotype profiles for a synthetic simulation. Values in the
//! bundled example are deliberately illustrative rather than host telemetry.
//! A profile is plain JSON so any host can be added without code changes.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub name: String,
    pub cpu: String,
    pub cpu_threads: usize,
    pub dram_gbps: f64,
    pub dram_gbps_dual_channel: f64,
    pub vram_gbps: f64,
    pub vram_bytes: u64,
    pub nvme_gbps: f64,
    pub pcie_gbps: f64,
    pub ram_bytes: u64,
    /// per-byte decode cost (µs/KB) per quant class; key "default" required.
    /// The simulator uses the streaming value unless the layer set fits the
    /// caller's cache-budget assumptions.
    pub per_byte_decode_us_per_kb: BTreeMap<String, f64>,
    /// Kernel-selection regions: one path wins below `small_payload`, another
    /// above `large_payload`; the values are supplied by the caller.
    pub kernel_region_small_payload_bytes: u64,
    pub kernel_region_large_payload_bytes: u64,
    pub kernel_spawn_overhead_ms: f64,
    /// Host graph-node work per token (binding-cost class).
    pub graph_nodes_per_token: f64,
    pub graph_fusible_fraction: f64,
    /// calibrated efficiency/scale factors; `lb calibrate` fits these.
    pub eff: Efficiency,
    /// MTP acceptance per drafted position (caller-supplied calibration).
    pub mtp_acceptance: f64,
    /// relative cost of one MTP draft step vs one target layer pass.
    pub mtp_draft_step_frac: f64,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Efficiency {
    pub dram: f64,
    pub vram: f64,
    pub compute: f64,
    pub prefill: f64,
}

impl HardwareProfile {
    pub fn load(path: &Path) -> Result<Self, String> {
        let data = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&data).map_err(|e| e.to_string())
    }

    pub fn per_byte_us_per_kb(&self, quant: &str) -> f64 {
        self.per_byte_decode_us_per_kb
            .get(quant)
            .or_else(|| self.per_byte_decode_us_per_kb.get("default"))
            .copied()
            .unwrap_or(1.0)
    }
}

/// An illustrative profile for exercising the CLI without a hardware probe.
pub const EXAMPLE_PROFILE: &str = r#"{
  "name": "example-host",
  "cpu": "example-cpu",
  "cpu_threads": 8,
  "dram_gbps": 32.0,
  "dram_gbps_dual_channel": 64.0,
  "vram_gbps": 400.0,
  "vram_bytes": 16000000000,
  "nvme_gbps": 3.0,
  "pcie_gbps": 32.0,
  "ram_bytes": 64000000000,
  "per_byte_decode_us_per_kb": {
    "default": 1.0,
    "Q5_K": 1.0,
    "Q6_K": 1.0,
    "Q8_0": 0.9,
    "F16": 1.0,
    "F32": 1.2
  },
  "kernel_region_small_payload_bytes": 4000000,
  "kernel_region_large_payload_bytes": 64000000,
  "kernel_spawn_overhead_ms": 10.0,
  "graph_nodes_per_token": 3000.0,
  "graph_fusible_fraction": 0.4,
  "eff": { "dram": 0.96, "vram": 0.65, "compute": 0.85, "prefill": 0.9 },
  "mtp_acceptance": 0.62,
  "mtp_draft_step_frac": 0.05,
  "notes": [
    "illustrative DRAM bandwidth; replace with a reviewed host profile",
    "illustrative VRAM streaming bandwidth; replace with a reviewed host profile",
    "illustrative NVMe and PCIe values; not a host measurement",
    "illustrative kernel startup class for simulation only",
    "illustrative graph-node cost; validate before using for a claim"
  ]
}"#;
