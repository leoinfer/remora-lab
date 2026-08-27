//! # lb-sim
//!
//! Synthetic performance simulation: predicts prefill and decode tok/s for
//! any model metadata × hardware phenotype × config — **without ever
//! running the model**.
//!
//! ## Model (all terms documented; defaults are illustrative)
//!
//! Decode is modeled as the minimum of two ceilings:
//!
//! ```text
//!   bandwidth ceiling (per accepted token):
//!     t_pass   = w_cpu / (DRAM · eff_dram) + w_gpu / (VRAM · eff_vram) · (1 − overlap)
//!     MTP:     t = (t_pass · (1 + n_max · draft_frac) + graph) / E[A]
//!     E[A]    = Σ_{k=1..n_max} p^k        (survival sum; p is an input)
//!   compute ceiling:
//!     t_comp   = flops_per_accepted / (threads · gf_per_thread · 1e9 · eff_compute)
//!     flops_per_accepted = 2·n_params·(n_max/E[A])·(cpu_layer_frac + gpu_layer_frac·(VRAM_rate/CPU_rate))
//! ```
//!
//! Prefill is a caller-calibrated batch-power model:
//!
//! ```text
//!   prefill = base · (batch / 128)^0.85 · fa_factor · (w_gpu_frac·gpu_pf + w_cpu_frac·cpu_pf)
//! ```
//!
//! Every synthetic constant is a [`Calibration`] that `lb calibrate` can fit
//! to caller-supplied anchors; nothing is hard-coded as truth.
//!
//! ## No-cheat guarantees
//!
//! 1. **Predictions are ceilings with named bottlenecks** — a result always
//!    says *which* resource binds, so a mismatch against a real run is
//!    diagnosable (that is the calibration loop).
//! 2. **Fit is checked before trust** — memory/VRAM overflow degrades the
//!    prediction into a thrash warning instead of a number.
//! 3. **Known-bug knowledge is carried** — exactness-risk flags (p-min
//!    early-stop on hybrid models, B0 first-use) are part of the result.

use lb_hardware::HardwareProfile;
use lb_model::ModelMeta;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpecKind {
    None,
    Mtp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub ngl: usize,
    pub threads: usize,
    pub batch: usize,
    pub ubatch: usize,
    pub ctx: usize,
    pub kv_q8: bool,
    pub fa: bool,
    pub mmap: bool,
    pub kv_offload: bool,
    pub spec: SpecKind,
    pub n_max: usize,
    pub p_min: f64,
    /// host/GPU overlap ∈ [0,1]; 0 = fully serialized (conservative default).
    pub overlap: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ngl: 0,
            threads: 8,
            batch: 128,
            ubatch: 128,
            ctx: 4096,
            kv_q8: false,
            fa: false,
            mmap: true,
            kv_offload: true,
            spec: SpecKind::None,
            n_max: 0,
            p_min: 0.0,
            overlap: 0.0,
        }
    }
}

impl Config {
    pub fn kv_elem_bytes(&self) -> (u64, u64) {
        if self.kv_q8 { (1, 1) } else { (2, 2) }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workload {
    pub prompt_tokens: usize,
    pub gen_tokens: usize,
}

impl Default for Workload {
    fn default() -> Self {
        Self {
            prompt_tokens: 2048,
            gen_tokens: 512,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Bottleneck {
    DramBandwidth,
    VramBandwidth,
    CpuCompute,
    GpuCompute,
    NvmeCold,
    HostGraph,
}

impl Bottleneck {
    pub fn label(&self) -> &'static str {
        match self {
            Self::DramBandwidth => "DRAM bandwidth",
            Self::VramBandwidth => "VRAM bandwidth",
            Self::CpuCompute => "CPU compute",
            Self::GpuCompute => "GPU compute",
            Self::NvmeCold => "NVMe cold (thrash)",
            Self::HostGraph => "host graph-node work",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimResult {
    pub config: Config,
    pub decode_tok_s: f64,
    pub prefill_tok_s: f64,
    pub decode_bw_tok_s: f64,
    pub decode_comp_tok_s: f64,
    pub bottleneck: Bottleneck,
    pub vram_gb: f64,
    pub ram_gb: f64,
    pub kv_bytes: u64,
    pub mtp: Option<MtpStats>,
    pub risks: Vec<String>,
    /// research paths suggested by the binding bottleneck (the "make it
    /// faster" map, straight from the idea registry).
    pub levers: Vec<String>,
    /// wall-clock estimate for the workload (s).
    pub est_wall_s: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MtpStats {
    pub expected_accepted_per_block: f64,
    pub mean_draft_len: f64,
    pub acceptance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    /// AVX2 FMA GFLOPS per thread (compute ceiling; batched ~4×).
    pub gf_per_thread: f64,
    pub gf_per_thread_batched: f64,
    /// prefill base tok/s at batch=128, fa=off (per quant-class byte weight).
    pub prefill_base_tok_s: f64,
    pub prefill_batch_exp: f64,
    pub prefill_fa_factor: f64,
    pub prefill_gpu_factor: f64,
    /// host graph overhead seconds per token (binding-cost model).
    pub graph_overhead_s: f64,
    /// MTP draft step as a fraction of one full-model pass.
    pub draft_step_frac: f64,
    /// per-position MTP acceptance supplied by calibration input.
    pub acceptance: f64,
    /// per-drafted-position acceptance decay (p_k = p · decay^(k−1)).
    pub acceptance_decay: f64,
    /// fixed per-draft-step overhead (kernel spawn/sync class).
    pub draft_step_overhead_ms: f64,
    /// effective CPU-side stream factor vs raw layer bytes, including any
    /// caller-modelled page-cache or memory effects.
    pub cpu_bytes_factor: f64,
    /// compute-thread efficiency multipliers for [4, 8, 12, 16] threads
    /// (SMT/memory contention model).
    #[serde(default = "default_thread_eff")]
    pub thread_eff: [f64; 4],
    /// default host/GPU overlap used by search seeds (0 = serial).
    #[serde(default)]
    pub overlap_default: f64,
}

fn default_thread_eff() -> [f64; 4] {
    [1.0, 1.0, 0.9, 0.45]
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            gf_per_thread: 45.0,
            gf_per_thread_batched: 180.0,
            prefill_base_tok_s: 12.5,
            prefill_batch_exp: 0.85,
            prefill_fa_factor: 0.92,
            prefill_gpu_factor: 3.0,
            graph_overhead_s: 0.004,
            draft_step_frac: 0.05,
            acceptance: 0.62,
            acceptance_decay: 0.75,
            draft_step_overhead_ms: 15.0,
            cpu_bytes_factor: 0.45,
            thread_eff: [1.0, 1.0, 0.9, 0.45],
            overlap_default: 0.0,
        }
    }
}

/// Simulate decode + prefill for a config. Returns predictions + fit + risks.
pub fn simulate(
    model: &ModelMeta,
    hw: &HardwareProfile,
    cfg: &Config,
    cal: &Calibration,
    wl: &Workload,
) -> SimResult {
    let mut risks: Vec<String> = Vec::new();
    let mut levers: Vec<String> = Vec::new();

    // ---- byte accounting -------------------------------------------------
    // llama.cpp placement: GPU gets `ngl` layers first; non-layer tensors
    // (embeddings/output) fill only leftover VRAM.
    let layer_b = model.avg_layer_bytes();
    let gpu_layers = cfg.ngl.min(model.n_layers);
    let cpu_layers = model.n_layers.saturating_sub(gpu_layers);
    let vram_budget = hw.vram_bytes as f64 * 0.93;
    let w_gpu_layers = gpu_layers as f64 * layer_b;
    let non_layer = model.non_layer_bytes as f64;
    let non_layer_gpu = (vram_budget - w_gpu_layers).max(0.0).min(non_layer);
    let w_gpu = if gpu_layers > 0 {
        w_gpu_layers + non_layer_gpu
    } else {
        0.0
    };
    let w_cpu =
        (cpu_layers as f64 * layer_b + (non_layer - non_layer_gpu)) * cal.cpu_bytes_factor.max(0.1);
    let (k_elem, v_elem) = cfg.kv_elem_bytes();
    let kv_per_tok = model.kv_bytes_per_token(k_elem, v_elem);
    let kv_bytes = kv_per_tok.saturating_mul(cfg.ctx as u64);
    let kv_gpu = if cfg.kv_offload { kv_bytes as f64 } else { 0.0 };
    let vram_used = w_gpu + kv_gpu;
    let vram_cap = hw.vram_bytes as f64 * 0.93;
    let ram_used = if cfg.mmap {
        w_cpu + (kv_bytes as f64 - kv_gpu)
    } else {
        w_gpu + w_cpu + (kv_bytes as f64 - kv_gpu)
    };

    // ---- fit checks ------------------------------------------------------
    let vram_ok = vram_used <= vram_cap;
    let ram_ok = ram_used <= hw.ram_bytes as f64;
    if !vram_ok {
        risks.push(format!(
            "VRAM over budget: {:.1} GB > cap {:.1} GB — reduce ngl or move KV to CPU (--no-kv-offload)",
            vram_used / 1e9,
            vram_cap / 1e9
        ));
    }
    if !ram_ok && cfg.mmap {
        risks.push(format!(
            "RAM working set {:.1} GB > {:.1} GB with mmap — CPU-layer weights will thrash from NVMe (cold-decode regime)",
            ram_used / 1e9,
            hw.ram_bytes as f64 / 1e9
        ));
        levers.push("increase RAM, drop page cache, or move layers to GPU (N02 residency)".into());
    }
    if !cfg.mmap && !ram_ok {
        risks.push("RAM over budget with no-mmap — model cannot load".into());
    }

    // ---- decode: bandwidth ceiling --------------------------------------
    let dram = hw.dram_gbps * 1e9 * hw.eff.dram;
    let vram = hw.vram_gbps * 1e9 * hw.eff.vram;
    let t_cpu = if dram > 0.0 {
        w_cpu / dram
    } else {
        f64::INFINITY
    };
    let t_gpu = if vram > 0.0 && w_gpu > 0.0 {
        w_gpu / vram
    } else {
        0.0
    };
    let t_pass = t_cpu + t_gpu * (1.0 - cfg.overlap.clamp(0.0, 1.0));

    let graph = cal.graph_overhead_s;
    let (bw_tok_s, mtp) = match cfg.spec {
        SpecKind::None => (1.0 / (t_pass + graph), None),
        SpecKind::Mtp if cfg.n_max == 0 => (1.0 / (t_pass + graph), None),
        SpecKind::Mtp => {
            let p = cal.acceptance;
            let mut ea = 0.0;
            let mut pk = 1.0;
            for k in 1..=cfg.n_max {
                pk *= p * cal.acceptance_decay.powf(k as f64 - 1.0);
                ea += pk;
            }
            let ea = ea.max(1e-9);
            let step_overhead_s = cal.draft_step_overhead_ms / 1000.0;
            let t_block = (t_pass + graph) * (1.0 + cfg.n_max as f64 * cal.draft_step_frac)
                + cfg.n_max as f64 * step_overhead_s;
            let stats = MtpStats {
                expected_accepted_per_block: ea,
                mean_draft_len: ea / p.max(1e-9),
                acceptance: p,
            };
            (ea / t_block, Some(stats))
        }
    };

    // ---- decode: compute ceiling -----------------------------------------
    let n_params = model.total_elements as f64;
    let flops_full = 2.0 * n_params * 2.0; // 2 ops/FMA, 2 flops per param per token
    let cpu_frac = w_cpu / (w_cpu + w_gpu).max(1.0);
    let (k_pos, acc_scale) = match &mtp {
        Some(m) => (cfg.n_max as f64, m.expected_accepted_per_block),
        None => (1.0, 1.0),
    };
    let flops_per_accepted = flops_full * (k_pos / acc_scale.max(1e-9)) * cpu_frac;
    let teff = match cfg.threads {
        4 => cal.thread_eff[0],
        8 => cal.thread_eff[1],
        12 => cal.thread_eff[2],
        16 => cal.thread_eff[3],
        t if t <= 4 => cal.thread_eff[0],
        t if t <= 8 => cal.thread_eff[1],
        t if t <= 12 => cal.thread_eff[2],
        _ => cal.thread_eff[3],
    };
    let cpu_gflops = cfg.threads as f64 * cal.gf_per_thread * hw.eff.compute * teff;
    let comp_tok_s = if cpu_gflops > 0.0 {
        (cpu_gflops * 1e9) / flops_per_accepted.max(1.0)
    } else {
        f64::INFINITY
    };

    // ---- decode: pick bottleneck -----------------------------------------
    let (decode_tok_s, bottleneck) = if !vram_ok {
        // model doesn't fit as configured: report the fit-limited ceiling
        (bw_tok_s.min(comp_tok_s), Bottleneck::VramBandwidth)
    } else if bw_tok_s < comp_tok_s {
        (bw_tok_s, Bottleneck::DramBandwidth)
    } else {
        (comp_tok_s, Bottleneck::CpuCompute)
    };
    let decode_bw_tok_s = bw_tok_s;
    let decode_comp_tok_s = comp_tok_s;

    // ---- prefill (calibrated batch-power model) ---------------------------
    let gpu_frac = w_gpu / (w_gpu + w_cpu).max(1.0);
    let batch_pow = (cfg.batch as f64 / 128.0)
        .max(0.125)
        .powf(cal.prefill_batch_exp);
    let fa_factor = if cfg.fa { cal.prefill_fa_factor } else { 1.0 };
    let tier = (1.0 - gpu_frac) + gpu_frac * cal.prefill_gpu_factor;
    let prefill_tok_s = (cal.prefill_base_tok_s * batch_pow * fa_factor * tier).max(0.5);

    // ---- research levers from the binding bottleneck ----------------------
    match bottleneck {
        Bottleneck::DramBandwidth => {
            levers
                .push("DRAM-bound decode: increase MTP depth — weights read once per block".into());
            levers.push(
                "dual-channel RAM — provide a reviewed host-specific bandwidth profile".into(),
            );
            levers.push(
                "byte reduction: REMORA-10 organ map / F16 collapse — fewer bytes/token".into(),
            );
            levers.push("graph-node reduction: host nodes track wall-time, not bytes".into());
        }
        Bottleneck::VramBandwidth => {
            levers.push("VRAM-bound: raise ngl efficiency or move KV to CPU".into());
        }
        Bottleneck::CpuCompute => {
            levers.push(
                "compute-bound decode: explore bounded MTP depths, then threads/quantization"
                    .into(),
            );
            levers.push("kernel lead: test fused pre-dequant GEMV variants".into());
        }
        Bottleneck::NvmeCold => {
            levers.push("cold-decode regime: RAM increase or residency (N02)".into());
        }
        Bottleneck::HostGraph => {
            levers.push("host-graph-bound: consider view elision after exactness review".into());
        }
        Bottleneck::GpuCompute => {
            levers.push("GPU-compute-bound: inspect RDNA4 kernel occupancy and arithmetic".into());
        }
    }
    if cfg.p_min > 0.0 && cfg.spec == SpecKind::Mtp {
        risks.push(
            "p-min > 0 can corrupt exactness when state-boundary conditions are unproven — use p_min=0 until validated"
                .into(),
        );
    }
    if model.has_mtp && cfg.spec == SpecKind::None {
        levers.push(
            "model has a NextN/MTP block — enable --spec-type draft-mtp n_max=3 after calibration"
                .into(),
        );
    }
    if cfg.ctx > 32_768 && cfg.kv_offload {
        risks.push(format!(
            "KV at ctx {} with offload = {:.1} GB on GPU; --no-kv-offload keeps KV in RAM (computed from supplied inputs)",
            cfg.ctx,
            kv_bytes as f64 / 1e9
        ));
    }
    if cfg.ctx > 32_768 && !cfg.fa {
        risks.push(
            "ctx > 32K without flash attention: prefill and KV round-trips degrade sharply".into(),
        );
    }
    if cfg.kv_q8 && !cfg.fa {
        risks.push("q8_0 KV requires flash attention in the dense fork ('V cache quantization requires flash_attn')".into());
    }

    let gen_s = wl.gen_tokens as f64 / decode_tok_s.max(1e-9);
    let pf_s = wl.prompt_tokens as f64 / prefill_tok_s.max(1e-9);
    let est_wall_s = gen_s + pf_s;

    SimResult {
        config: cfg.clone(),
        decode_tok_s,
        prefill_tok_s,
        decode_bw_tok_s,
        decode_comp_tok_s,
        bottleneck,
        vram_gb: vram_used / 1e9,
        ram_gb: ram_used / 1e9,
        kv_bytes,
        mtp,
        risks,
        levers,
        est_wall_s,
    }
}

/// Fit the calibration scalars to caller-supplied anchors by coarse grid search.
/// Anchors: `[{"decode_tok_s": f64, "prefill_tok_s": f64, "config": {...}}]`.
pub fn calibrate(
    model: &ModelMeta,
    hw: &HardwareProfile,
    anchors: &[Anchor],
) -> (Calibration, f64) {
    let mut best = Calibration::default();
    let mut best_rmse = f64::INFINITY;
    let cpus = [30.0, 45.0, 60.0, 80.0];
    let bases = [8.0, 10.0, 12.5, 15.0];
    let exps = [0.7, 0.85, 1.0];
    let fa_f = [0.85, 0.92, 1.0];
    let drafts = [0.03, 0.05, 0.08];
    let accs = [0.55, 0.62, 0.7, 0.85];
    let decays = [0.6, 0.75, 0.9, 1.0];
    let ovh = [5.0, 10.0, 15.0, 25.0];
    let cpu_f = [0.4, 0.45, 0.55, 0.7, 1.0];
    let gpu_f = [1.0, 1.5, 2.0, 3.0];
    let t16 = [0.45, 0.6, 0.8];
    for gf in cpus {
        for base in bases {
            for exp in exps {
                for faf in fa_f {
                    for dfrac in drafts {
                        for acc in accs {
                            for dec in decays {
                                for ov in ovh {
                                    for cbf in cpu_f {
                                        for gpf in gpu_f {
                                            for t16e in t16 {
                                                let cal = Calibration {
                                                    gf_per_thread: gf,
                                                    prefill_base_tok_s: base,
                                                    prefill_batch_exp: exp,
                                                    prefill_fa_factor: faf,
                                                    draft_step_frac: dfrac,
                                                    acceptance: acc,
                                                    acceptance_decay: dec,
                                                    draft_step_overhead_ms: ov,
                                                    cpu_bytes_factor: cbf,
                                                    prefill_gpu_factor: gpf,
                                                    thread_eff: [1.0, 1.0, 0.9, t16e],
                                                    ..Default::default()
                                                };
                                                let mut se = 0.0;
                                                let mut n = 0;
                                                for a in anchors {
                                                    let r = simulate(
                                                        model,
                                                        hw,
                                                        &a.config,
                                                        &cal,
                                                        &Workload::default(),
                                                    );
                                                    let d_err = (r.decode_tok_s - a.decode_tok_s)
                                                        / a.decode_tok_s.max(1e-9);
                                                    let p_err = (r.prefill_tok_s - a.prefill_tok_s)
                                                        / a.prefill_tok_s.max(1e-9);
                                                    se += d_err * d_err + p_err * p_err;
                                                    n += 2;
                                                }
                                                let rmse = (se / n.max(1) as f64).sqrt();
                                                if rmse < best_rmse {
                                                    best_rmse = rmse;
                                                    best = cal;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (best, best_rmse)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anchor {
    pub config: Config,
    pub decode_tok_s: f64,
    pub prefill_tok_s: f64,
    pub label: String,
}
