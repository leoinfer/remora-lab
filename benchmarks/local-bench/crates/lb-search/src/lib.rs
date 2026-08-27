//! # lb-search
//!
//! Beam search over the configuration space, scored on a workload
//! (prompt tokens + generated tokens). Every candidate is fully simulated
//! (fit checks + exactness-risk flags included), so the ranked table is a
//! **research map**: the top configs, their binding bottlenecks, and the
//! levers that would make them faster.

use lb_hardware::HardwareProfile;
use lb_model::ModelMeta;
use lb_sim::{Calibration, Config, SpecKind, Workload, simulate};
use serde::{Deserialize, Serialize};

/// Research-lever transforms applied before searching. Each lever answers
/// "how fast WOULD it be if this research path landed?" — the search then
/// finds the config that exploits it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Levers {
    /// dual-channel RAM: swap DRAM bandwidth to the dual-channel value.
    pub dual_channel: bool,
    /// Overlap ∈ [0,1] between CPU and GPU layer work.
    pub overlap: Option<f64>,
    /// REMORA-10 organ-map byte cut: scale factor for weight bytes (e.g.
    /// 0.75 = F16 collapse; 0.59 = Q4_K conversion).
    pub bytes_scale: Option<f64>,
    /// Graph-node reduction: scale factor for graph overhead (0.5 =
    /// half the nodes).
    pub graph_cut: Option<f64>,
}

impl Levers {
    pub fn label(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if self.dual_channel {
            parts.push("dual-channel".into());
        }
        if let Some(o) = self.overlap {
            parts.push(format!("overlap{o:.2}"));
        }
        if let Some(b) = self.bytes_scale {
            parts.push(format!("bytes{b:.2}"));
        }
        if let Some(g) = self.graph_cut {
            parts.push(format!("graph{g:.2}"));
        }
        if parts.is_empty() {
            "baseline".into()
        } else {
            parts.join("+")
        }
    }
}

/// Scale a model's byte accounting by a factor (organ-map quant conversion).
pub fn scale_model(model: &ModelMeta, bytes_scale: f64) -> ModelMeta {
    let mut m = model.clone();
    m.total_tensor_bytes = (m.total_tensor_bytes as f64 * bytes_scale) as u64;
    m.non_layer_bytes = (m.non_layer_bytes as f64 * bytes_scale) as u64;
    m.layer_bytes = (m.layer_bytes as f64 * bytes_scale) as u64;
    m
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub rank: usize,
    pub config: Config,
    pub decode_tok_s: f64,
    pub prefill_tok_s: f64,
    pub est_wall_s: f64,
    pub bottleneck: String,
    pub vram_gb: f64,
    pub ram_gb: f64,
    pub risks: Vec<String>,
    pub levers: Vec<String>,
}

/// Beam search. `beam` keeps the top-N candidates per round; each round
/// varies one dimension from the current frontier.
#[allow(clippy::too_many_arguments)]
pub fn search(
    model: &ModelMeta,
    hw: &HardwareProfile,
    cal: &Calibration,
    wl: &Workload,
    ctx: usize,
    kv_q8: bool,
    fa: bool,
    beam: usize,
    max_rounds: usize,
    levers: &Levers,
) -> Vec<SearchHit> {
    let mut hw = hw.clone();
    let mut cal = cal.clone();
    let mut model = model.clone();
    if levers.dual_channel {
        hw.dram_gbps = hw.dram_gbps_dual_channel;
    }
    if let Some(o) = levers.overlap {
        cal.overlap_default = o.clamp(0.0, 1.0);
    }
    if let Some(b) = levers.bytes_scale {
        model = scale_model(&model, b.clamp(0.3, 1.0));
    }
    if let Some(g) = levers.graph_cut {
        cal.graph_overhead_s *= g.clamp(0.1, 1.0);
    }
    let mut frontier: Vec<(Config, f64)> = Vec::new();
    let seed = Config {
        ctx,
        kv_q8,
        fa,
        spec: SpecKind::Mtp,
        n_max: 3,
        p_min: 0.0,
        overlap: cal.overlap_default,
        ..Default::default()
    };
    frontier.push((seed.clone(), score(&model, &hw, &cal, wl, &seed)));

    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    seen.insert(cfg_key(&seed));

    for _ in 0..max_rounds {
        let mut next: Vec<(Config, f64)> = Vec::new();
        for (cfg, _) in &frontier {
            for c in neighbors(&model, &hw, cfg, ctx, kv_q8, fa) {
                let key = cfg_key(&c);
                if !seen.insert(key.clone()) {
                    continue;
                }
                let s = score(&model, &hw, &cal, wl, &c);
                if s.is_finite() {
                    next.push((c, s));
                }
            }
        }
        next.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        next.truncate(beam);
        if next.is_empty() {
            break;
        }
        frontier = next;
    }

    frontier.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    frontier
        .into_iter()
        .enumerate()
        .map(|(i, (cfg, _))| {
            let r = simulate(&model, &hw, &cfg, &cal, wl);
            SearchHit {
                rank: i + 1,
                config: cfg,
                decode_tok_s: r.decode_tok_s,
                prefill_tok_s: r.prefill_tok_s,
                est_wall_s: r.est_wall_s,
                bottleneck: r.bottleneck.label().to_string(),
                vram_gb: r.vram_gb,
                ram_gb: r.ram_gb,
                risks: r.risks.clone(),
                levers: r.levers.clone(),
            }
        })
        .collect()
}

/// Lower wall-time is better; ties broken toward fewer risks implicitly by
/// the fit checks already penalizing (infinite) bad fits.
fn score(
    model: &ModelMeta,
    hw: &HardwareProfile,
    cal: &Calibration,
    wl: &Workload,
    cfg: &Config,
) -> f64 {
    let r = simulate(model, hw, cfg, cal, wl);
    if r.risks
        .iter()
        .any(|x| x.contains("over budget") || x.contains("cannot load"))
    {
        return f64::INFINITY;
    }
    r.est_wall_s
}

/// One-dimension neighbors of a config (respecting fork constraints).
fn neighbors(
    model: &ModelMeta,
    hw: &HardwareProfile,
    cfg: &Config,
    ctx: usize,
    _kv_q8: bool,
    _fa: bool,
) -> Vec<Config> {
    let mut out = Vec::new();
    let mut push = |c: Config| {
        // fork constraint: q8_0 KV requires flash attention
        if c.kv_q8 && !c.fa {
            return;
        }
        // p_min must be 0 (state-boundary bug on hybrid models)
        if c.p_min > 0.0 {
            return;
        }
        // MTP needs the model to have a NextN block
        if c.spec == SpecKind::Mtp && !model.has_mtp {
            return;
        }
        if c.ngl > model.n_layers {
            return;
        }
        out.push(c);
    };

    // ngl ladder: fit-driven — max useful ngl ≈ vram_cap / layer_b
    let layer_b = model.avg_layer_bytes();
    let cap_layers = ((hw.vram_bytes as f64 * 0.93) / layer_b.max(1.0)) as usize;
    let mut nl = vec![0, 8, 16];
    for l in [20, 24, 26, 30, 32, 36, 38, 40, 44, 48, 52, 56, 60, 64] {
        if l <= cap_layers {
            nl.push(l);
        }
    }
    nl.push(model.n_layers.min(cap_layers.max(1)));
    nl.dedup();
    for l in nl {
        if l != cfg.ngl {
            let mut c = cfg.clone();
            c.ngl = l;
            push(c);
        }
    }

    // threads
    for t in [4, 8, 12, 16] {
        if t != cfg.threads && t <= hw.cpu_threads * 2 {
            let mut c = cfg.clone();
            c.threads = t;
            push(c);
        }
    }
    // batch/ubatch
    for b in [64, 128, 256, 512, 1024, 2048] {
        if b != cfg.batch {
            let mut c = cfg.clone();
            c.batch = b;
            c.ubatch = b;
            push(c);
        }
    }
    // MTP depth — capped at 3 for this bounded search. Multi-block models
    // can extend this later after a caller supplies suitable calibration.
    for n in [0usize, 1, 2, 3] {
        if n != cfg.n_max {
            let mut c = cfg.clone();
            c.n_max = n;
            c.spec = if n == 0 {
                SpecKind::None
            } else {
                SpecKind::Mtp
            };
            push(c);
        }
    }
    // KV/FA cross-product (constraints enforced in push)
    for (kq, f) in [(false, false), (false, true), (true, true)] {
        if (kq, f) != (cfg.kv_q8, cfg.fa) {
            let mut c = cfg.clone();
            c.kv_q8 = kq;
            c.fa = f;
            push(c);
        }
    }
    // KV offload flip
    for ko in [true, false] {
        if ko != cfg.kv_offload {
            let mut c = cfg.clone();
            c.kv_offload = ko;
            push(c);
        }
    }
    // ctx ladder (keep requested ctx, but also probe 512/4096/32k/128k/192k)
    for c in [512usize, 4096, 32768, 131072, 196608] {
        if c != ctx && cfg.ctx == ctx {
            let mut cc = cfg.clone();
            cc.ctx = c;
            push(cc);
        }
    }
    out
}

fn cfg_key(c: &Config) -> String {
    format!(
        "{}-{}-{}-{}-{}-{}-{}-{}-{}",
        c.ngl,
        c.threads,
        c.batch,
        c.ctx,
        c.kv_q8,
        c.fa,
        c.kv_offload,
        c.spec == SpecKind::Mtp,
        c.n_max
    )
}
