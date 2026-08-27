//! Native server configuration — familiar model-serving flags mapped onto the
//! serving runtime (scheduler + residency + speculation + kernels).
//!
//! The `har-server` binary is the native Rust serving entry
//! point: every runtime knob is a `--flag`. Today the model backends are
//! the CPU-verifiable ones (toy, synthetic Q4_K, synthetic Q4_0 — the
//! GGUF-compatible format); the native GGUF tensor backend and the
//! Vulkan kernels plug in behind the same `BatchStepModel` seam without
//! changing the flag surface.

use crate::adapter::BatchStepModel;
use crate::gguf::load_rows;
use crate::moe::{MoEConfig, MoEModel};
use crate::q40::{synthetic_blocks, Q40Model};
use crate::q4k::Q4KModel;
use crate::speculation::{SpecConfig, SpeculativeModel};
use crate::tokenizer::Tokenizer;
use crate::{
    SequenceId, ServeConfig, ServeError, ServeScheduler, SpecTelemetry, StepReport, ToyConfig,
    ToyModel,
};
use std::path::PathBuf;

/// The full server configuration surface (flag → field).
#[derive(Clone, Debug)]
pub struct ServerConfig {
    // model
    pub model: String,
    pub backend: BackendKind,
    pub tensor: String,
    pub rows: usize,
    // serving
    pub max_batch: usize,
    pub prefill_chunk: usize,
    pub page_size: usize,
    pub kv_type: String,
    pub target_context: usize,
    // residency
    pub cache_bytes: Option<u64>,
    pub live_state_bytes: Option<u64>,
    pub pin_live: bool,
    // speculation
    pub spec_type: SpecType,
    pub spec_draft_n_max: usize,
    pub spec_draft_p_min: f32,
    // HTTP serving (`--port` switches from CLI demo to server mode)
    pub host: String,
    pub port: Option<u16>,
    // run
    pub prompt_ids: Vec<u32>,
    pub prompt_text: Option<String>,
    pub max_new: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BackendKind {
    Toy,
    Q4KSynthetic,
    Q40Synthetic,
    /// Full dense transformer reference (GQA + RoPE + RMSNorm + MLP) —
    /// the real-model-shaped CPU backend (`--model dense`).
    Dense,
    /// Native GGUF file (`--model model.gguf [--tensor ...] [--rows n]`);
    /// tensor/rows are read live from `ServerConfig` so flag order does
    /// not matter.
    Gguf {
        path: String,
    },
    /// Synthetic routed-expert model with expert-major batching.
    MoE,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpecType {
    None,
    Draft,
    /// Telemetry-driven tier calibration: the first request is served with a
    /// wide, ungated probe horizon; the observed
    /// acceptance curve then derives the value-optimal block and the
    /// lower tiers (`calibrate_tiers`), which apply from the second
    /// request on.
    Auto,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            model: "toy".into(),
            backend: BackendKind::Toy,
            tensor: "token_embd.weight".into(),
            rows: 128,
            max_batch: 4,
            prefill_chunk: 1,
            page_size: 8,
            kv_type: "q8_0".into(),
            target_context: 8192,
            cache_bytes: None,
            live_state_bytes: None,
            pin_live: true,
            spec_type: SpecType::None,
            spec_draft_n_max: 3,
            spec_draft_p_min: 0.75,
            host: "127.0.0.1".into(),
            port: None,
            prompt_ids: vec![1, 2, 3],
            prompt_text: None,
            max_new: 16,
        }
    }
}

/// Parse `--flag value` / `--flag=value` arguments.
/// Unknown flags are ignored with a warning (forward compat).
pub fn parse_args(args: &[String]) -> Result<ServerConfig, String> {
    let mut cfg = ServerConfig::default();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) => (f.to_string(), Some(v.to_string())),
            None => (arg.clone(), None),
        };
        let next = |i: &mut usize| -> Result<String, String> {
            if let Some(v) = &inline {
                Ok(v.clone())
            } else {
                *i += 1;
                args.get(*i)
                    .cloned()
                    .ok_or_else(|| format!("flag {flag} needs a value"))
            }
        };
        match flag.as_str() {
            "--model" | "-m" => {
                cfg.model = next(&mut i)?;
                let p = PathBuf::from(&cfg.model);
                if cfg.model == "toy" {
                    cfg.backend = BackendKind::Toy;
                } else if cfg.model == "dense" {
                    cfg.backend = BackendKind::Dense;
                } else if cfg.model == "moe" {
                    cfg.backend = BackendKind::MoE;
                } else if p.extension().is_some_and(|e| e == "gguf") && p.exists() {
                    cfg.backend = BackendKind::Gguf {
                        path: cfg.model.clone(),
                    };
                } else if cfg.model.ends_with("q4k") || cfg.model.contains("fixture") {
                    cfg.backend = BackendKind::Q4KSynthetic;
                } else if cfg.model == "q40" || cfg.model.contains("synthetic") {
                    cfg.backend = BackendKind::Q40Synthetic;
                } else {
                    cfg.backend = BackendKind::Q4KSynthetic;
                }
            }
            "--tensor" => cfg.tensor = next(&mut i)?,
            "--rows" => cfg.rows = next(&mut i)?.parse().map_err(|_| "rows")?,
            "--batch" | "-b" => cfg.max_batch = next(&mut i)?.parse().map_err(|_| "batch")?,
            "--chunked-prefill" => {
                cfg.prefill_chunk = next(&mut i)?.parse().map_err(|_| "chunked-prefill")?
            }
            "--page-size" => cfg.page_size = next(&mut i)?.parse().map_err(|_| "page-size")?,
            "--kv-dtype" | "-ctk" => cfg.kv_type = next(&mut i)?,
            "--ctx" | "-c" => cfg.target_context = next(&mut i)?.parse().map_err(|_| "ctx")?,
            "--cache-bytes" => {
                cfg.cache_bytes = Some(next(&mut i)?.parse().map_err(|_| "cache-bytes")?)
            }
            "--live-state-bytes" => {
                cfg.live_state_bytes = Some(next(&mut i)?.parse().map_err(|_| "live-state-bytes")?)
            }
            "--pin-live" => cfg.pin_live = next(&mut i)? != "0",
            "--spec-type" => {
                cfg.spec_type = match next(&mut i)?.as_str() {
                    "none" | "off" => SpecType::None,
                    "draft" | "mtp" | "dspark" | "respark" => SpecType::Draft,
                    "auto" | "calibrated" => SpecType::Auto,
                    other => return Err(format!("unknown --spec-type {other}")),
                }
            }
            "--spec-draft-n-max" => {
                cfg.spec_draft_n_max = next(&mut i)?.parse().map_err(|_| "spec-draft-n-max")?
            }
            "--spec-draft-p-min" => {
                cfg.spec_draft_p_min = next(&mut i)?.parse().map_err(|_| "spec-draft-p-min")?
            }
            "--host" => cfg.host = next(&mut i)?,
            "--port" => {
                cfg.port = Some(next(&mut i)?.parse().map_err(|_| "port")?);
            }
            "--prompt" => cfg.prompt_text = Some(next(&mut i)?),
            "--prompt-ids" => {
                cfg.prompt_ids = next(&mut i)?
                    .split(',')
                    .map(|v| v.trim().parse().map_err(|_| "prompt-ids"))
                    .collect::<Result<Vec<_>, _>>()?
            }
            "--max-new" | "-n" => cfg.max_new = next(&mut i)?.parse().map_err(|_| "max-new")?,
            _ => {
                if !flag.starts_with("--") {
                    return Err(format!("unexpected positional argument {flag}"));
                }
                eprintln!("har-server: ignoring unknown flag {flag}");
                if inline.is_none() {
                    i += 1; // skip the value
                }
                i += 1; // skip the flag
                continue;
            }
        }
        i += 1;
    }
    Ok(cfg)
}

impl ServerConfig {
    pub fn serve_config(&self) -> ServeConfig {
        ServeConfig {
            max_batch: self.max_batch,
            page_size: self.page_size,
            kv_type: self.kv_type.clone(),
            max_cache_bytes: self.cache_bytes,
            max_live_state_bytes: self.live_state_bytes,
            pin_live_nodes: self.pin_live,
            prefill_chunk: self.prefill_chunk,
            ..Default::default()
        }
    }

    pub fn spec_config(&self) -> Option<SpecConfig> {
        match self.spec_type {
            SpecType::None => None,
            SpecType::Draft => Some(self.threshold_config()),
            // Auto probe config: ungated (p_high=0) so every drafted
            // position reaches the absolute cap — the raw acceptance
            // curve is what the calibration needs.
            SpecType::Auto => Some(SpecConfig {
                block: self.spec_draft_n_max,
                p_high: 0.0,
                p_med: 0.0,
                p_min: 0.0,
                med_cap: self.spec_draft_n_max,
                min_cap: self.spec_draft_n_max,
            }),
        }
    }

    /// The user-threshold speculation config: the design p_high/p_med/
    /// p_min thresholds plus the user's n-max and p-min flags.  This is
    /// the threshold base for auto-calibration (its caps are replaced by
    /// the curve-derived ones; its thresholds are kept).
    fn threshold_config(&self) -> SpecConfig {
        SpecConfig {
            block: self.spec_draft_n_max,
            p_min: self.spec_draft_p_min,
            ..Default::default()
        }
    }

    /// Vocabulary size of the configured backend (the draft must share
    /// the target's tokenizer).
    pub fn vocab(&self) -> usize {
        match &self.backend {
            BackendKind::Toy => 1024,
            BackendKind::Dense => 512,
            BackendKind::Q4KSynthetic | BackendKind::Q40Synthetic => 128,
            BackendKind::Gguf { .. } => self.rows,
            BackendKind::MoE => 512,
        }
    }

    /// The toy target backend (also the self-speculation draft under
    /// `--spec-type auto`).
    fn build_toy(&self) -> ToyModel {
        ToyModel::new(ToyConfig {
            dim: 256,
            vocab: 1024,
            layers: 4,
            eos: 1023,
            seed: 42,
        })
    }

    /// Build the model backend for this config.
    pub fn build_model(&self) -> Result<Box<dyn BatchStepModel>, String> {
        let model: Box<dyn BatchStepModel> = match &self.backend {
            BackendKind::Toy => Box::new(self.build_toy()),
            BackendKind::Dense => Box::new(crate::dense::DenseModel::new(
                crate::dense::DenseConfig::default(),
            )),
            BackendKind::Q4KSynthetic => {
                let blocks = vec![0u8; 144 * 128];
                Box::new(Q4KModel::from_blocks(&blocks, 128, 127, 0xC0FFEE))
            }
            BackendKind::Q40Synthetic => {
                let blocks = synthetic_blocks(128, 0x5EED);
                Box::new(Q40Model::from_blocks(&blocks, 128, 127, 0x5EED))
            }
            BackendKind::Gguf { path } => {
                let loaded = load_rows(path, &self.tensor, self.rows, 0x5EED)?;
                Box::new(loaded.backend)
            }
            BackendKind::MoE => Box::new(MoEModel::new(MoEConfig {
                dim: 64,
                experts: 16,
                top_k: 2,
                shared_expert: true,
                expert_hidden: 128,
                vocab: 512,
                eos: 499,
                seed: 42,
            })),
        };
        Ok(model)
    }

    /// The draft for speculation: a smaller instance of the same backend
    /// family — with the TARGET's vocabulary (shared tokenizer;
    /// mismatched vocabs are out-of-bounds on embed).
    fn build_draft(&self) -> Box<dyn BatchStepModel> {
        let vocab = self.vocab();
        Box::new(ToyModel::new(ToyConfig {
            dim: 64,
            vocab,
            layers: 2,
            eos: vocab as u32 - 1,
            seed: 7,
        }))
    }

    /// Wire speculation on top of the model when configured and build the
    /// persistent serving runtime.  `--spec-type auto` builds the probe
    /// (ungated) config; [`BuiltScheduler::calibrate_auto`] then hot-swaps
    /// the calibrated policy from the observed acceptance curve. Under
    /// auto with the toy backend the draft is a copy of the target
    /// (self-speculation — the perfect-draft ideal), so the probe measures
    /// a real acceptance curve instead of an adversarial ~0 one.
    pub fn build_scheduler(&self) -> Result<BuiltScheduler, String> {
        match self.spec_type {
            SpecType::None => Ok(BuiltScheduler::Plain(ServeScheduler::new(
                self.build_model()?,
                self.serve_config(),
            ))),
            SpecType::Draft | SpecType::Auto => {
                // Self-speculation under auto for the toy and dense
                // backends: the draft is a copy of the target (the
                // perfect-draft ideal — the in-model MTP head is the real
                // deployment form), so the probe measures a real
                // acceptance curve instead of an adversarial ~0 one.
                let draft: Box<dyn BatchStepModel> = if self.spec_type == SpecType::Auto
                    && (self.backend == BackendKind::Toy || self.backend == BackendKind::Dense)
                {
                    match self.backend {
                        BackendKind::Toy => Box::new(self.build_toy()),
                        BackendKind::Dense => Box::new(crate::dense::DenseModel::new(
                            crate::dense::DenseConfig::default(),
                        )),
                        _ => unreachable!("checked above"),
                    }
                } else {
                    self.build_draft()
                };
                let base = self.threshold_config();
                let probe = self.spec_config().expect("spec");
                let spec = SpeculativeModel::new(
                    draft,
                    self.build_model()?,
                    if self.spec_type == SpecType::Auto {
                        probe
                    } else {
                        base
                    },
                );
                Ok(BuiltScheduler::Spec {
                    serve: ServeScheduler::new(spec, self.serve_config()),
                    auto: self.spec_type == SpecType::Auto,
                    base,
                })
            }
        }
    }
}

/// The built serving runtime: plain, or speculation-wrapped.  Kept
/// persistent across HTTP requests so the prefix graph accumulates; for
/// `--spec-type auto` the first request doubles as the calibration
/// probe.
#[allow(clippy::large_enum_variant)]
pub enum BuiltScheduler {
    Plain(ServeScheduler<Box<dyn BatchStepModel>>),
    Spec {
        serve: ServeScheduler<SpeculativeModel<Box<dyn BatchStepModel>, Box<dyn BatchStepModel>>>,
        /// Auto-calibration pending (first request doubles as probe).
        auto: bool,
        /// User-threshold policy: the calibration base (thresholds kept,
        /// caps curve-derived).
        base: SpecConfig,
    },
}

impl BuiltScheduler {
    pub fn submit(&mut self, prompt: &[u32], max_new: usize) -> Result<SequenceId, ServeError> {
        match self {
            BuiltScheduler::Plain(s) => s.submit(prompt, max_new),
            BuiltScheduler::Spec { serve, .. } => serve.submit(prompt, max_new),
        }
    }

    pub fn run_to_idle(&mut self) -> Vec<StepReport> {
        match self {
            BuiltScheduler::Plain(s) => s.run_to_idle(),
            BuiltScheduler::Spec { serve, .. } => serve.run_to_idle(),
        }
    }

    /// One continuous-batch step (SSE streaming drives the scheduler
    /// manually and emits tokens as they are produced).
    pub fn step(&mut self) -> StepReport {
        match self {
            BuiltScheduler::Plain(s) => s.step(),
            BuiltScheduler::Spec { serve, .. } => serve.step(),
        }
    }

    pub fn stream_of(&self, id: SequenceId) -> Result<Vec<u32>, ServeError> {
        match self {
            BuiltScheduler::Plain(s) => s.stream_of(id),
            BuiltScheduler::Spec { serve, .. } => serve.stream_of(id),
        }
    }

    pub fn spec_telemetry(&self) -> Option<SpecTelemetry> {
        match self {
            BuiltScheduler::Plain(_) => None,
            BuiltScheduler::Spec { serve, .. } => Some(serve.model().telemetry()),
        }
    }

    /// Auto-calibration: measure the acceptance curve accumulated so
    /// far, derive the calibrated tiered horizon, and hot-swap the policy.
    /// Returns the calibrated config + measured curve, or `None` when
    /// calibration is not pending or the probe produced no usable curve
    /// (fewer than 2 tracked positions, or all-zero — an adversarial
    /// draft).  The probe telemetry is reset so later reports cover the
    /// calibrated phase only.
    pub fn calibrate_auto(&mut self, floor: f64) -> Option<(SpecConfig, Vec<f64>)> {
        match self {
            BuiltScheduler::Spec { serve, auto, base } => {
                if !*auto {
                    return None;
                }
                let calibrated = serve.model_mut().calibrate(*base, floor)?;
                *auto = false;
                Some(calibrated)
            }
            BuiltScheduler::Plain(_) => None,
        }
    }
}

/// Resolve the prompt: `--prompt` (text, tokenized natively from the
/// model's GGUF tokenizer) or `--prompt-ids` (raw ids).
pub fn resolve_prompt(cfg: &ServerConfig) -> Result<Vec<u32>, String> {
    if let Some(text) = &cfg.prompt_text {
        match &cfg.backend {
            BackendKind::Gguf { path } => {
                let phenotype = har_model::GgufReader::new(path.clone())
                    .inspect(false)
                    .map_err(|e| format!("GGUF inspect for tokenizer: {e}"))?;
                let tok = phenotype
                    .tokenizer
                    .as_ref()
                    .and_then(Tokenizer::from_gguf)
                    .ok_or("no BPE tokenizer metadata in the model file")?;
                tok.encode(text)
            }
            _ => Err("--prompt requires a GGUF model (--model <file.gguf>)".into()),
        }
    } else {
        Ok(cfg.prompt_ids.clone())
    }
}

/// Run one prompt through the configured runtime; returns the token
/// stream and (in draft/auto mode) the real speculation telemetry.
///
/// `--spec-type auto` runs a probe phase first (wide, ungated horizon on
/// the request prompt itself), derives the calibrated tiered horizon from
/// the observed acceptance curve, prints it, then serves the
/// request under the calibrated policy.
pub fn run_demo(cfg: &ServerConfig) -> Result<(Vec<u32>, Option<SpecTelemetry>), String> {
    let prompt_ids = resolve_prompt(cfg)?;
    let mut s = cfg.build_scheduler()?;

    if cfg.spec_type == SpecType::Auto {
        // Probe phase: serve with the ungated probe horizon and read the
        // acceptance curve from live telemetry.
        s.submit(&prompt_ids, 64)
            .map_err(|e| format!("probe submit: {e}"))?;
        s.run_to_idle();
        match s.calibrate_auto(0.5) {
            Some((calibrated, curve)) => {
                println!(
                    "[auto] observed probe acceptance curve: {:?}",
                    curve.iter().map(|p| format!("{p:.2}")).collect::<Vec<_>>()
                );
                println!(
                    "[auto] calibrated policy: block={} p_high={} p_med={} p_min={} med_cap={} min_cap={}",
                    calibrated.block,
                    calibrated.p_high,
                    calibrated.p_med,
                    calibrated.p_min,
                    calibrated.med_cap,
                    calibrated.min_cap
                );
            }
            None => {
                // The guard's verdict: the draft accepted nothing (an
                // adversarial draft).  Honest fallback: plain decode.
                println!(
                    "[auto] probe: draft acceptance is zero (adversarial draft) — falling back to plain decode"
                );
                let mut plain = cfg.clone();
                plain.spec_type = SpecType::None;
                s = plain.build_scheduler()?;
            }
        }
    }

    let id = s
        .submit(&prompt_ids, cfg.max_new)
        .map_err(|e| format!("submit: {e}"))?;
    s.run_to_idle();
    let stream = s.stream_of(id).map_err(|e| e.to_string())?;
    Ok((stream, s.spec_telemetry()))
}

/// Validate the native GGUF model path before loading its bounded tensor
/// backend.
pub fn validate_model_path(cfg: &ServerConfig) -> Result<(), String> {
    match &cfg.backend {
        BackendKind::Gguf { path, .. } => {
            let p = PathBuf::from(path);
            if !p.exists() {
                return Err(format!("model file not found: {path}"));
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parses_openai_style_flags() {
        let cfg = parse_args(&args(&[
            "--model",
            "q40",
            "--batch",
            "8",
            "--chunked-prefill",
            "2048",
            "--ctx",
            "32768",
            "--kv-dtype",
            "q8_0",
            "--spec-type",
            "draft",
            "--spec-draft-n-max",
            "7",
            "--spec-draft-p-min",
            "0.8",
            "--cache-bytes",
            "1048576",
            "--prompt-ids",
            "1,2,3,4",
            "--max-new",
            "32",
        ]))
        .expect("parse");
        assert_eq!(cfg.backend, BackendKind::Q40Synthetic);
        assert_eq!(cfg.max_batch, 8);
        assert_eq!(cfg.prefill_chunk, 2048);
        assert_eq!(cfg.target_context, 32768);
        assert_eq!(cfg.spec_type, SpecType::Draft);
        assert_eq!(cfg.spec_draft_n_max, 7);
        assert_eq!(cfg.spec_draft_p_min, 0.8);
        assert_eq!(cfg.cache_bytes, Some(1048576));
        assert_eq!(cfg.prompt_ids, vec![1, 2, 3, 4]);
        assert_eq!(cfg.max_new, 32);
    }

    #[test]
    fn supports_equals_form() {
        let cfg =
            parse_args(&args(&["--batch=2", "--spec-type=none", "--model=toy"])).expect("parse");
        assert_eq!(cfg.max_batch, 2);
        assert_eq!(cfg.spec_type, SpecType::None);
        assert_eq!(cfg.backend, BackendKind::Toy);
    }

    #[test]
    fn unknown_flags_warn_and_continue() {
        let cfg = parse_args(&args(&["--future-flag", "x", "--batch", "3"])).expect("parse");
        assert_eq!(cfg.max_batch, 3);
    }

    #[test]
    fn missing_value_is_an_error() {
        assert!(parse_args(&args(&["--batch"])).is_err());
    }

    #[test]
    fn demo_runs_with_native_gguf_model() {
        // Build a tiny real Q4_0 GGUF and serve it via --model (the
        // OpenAI-style native path).
        let dir = std::env::temp_dir().join("har-serve-server-gguf");
        std::fs::create_dir_all(&dir).expect("tmpdir");
        let path = dir.join("server-demo.gguf");
        let rows = 16usize;
        let mut blocks = Vec::new();
        for _ in 0..rows * 8 {
            blocks.push(0x00);
            blocks.push(0x3c); // d = 1.0
            blocks.extend((0..16u8).map(|v| v.wrapping_mul(3)));
        }
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        let key = b"general.alignment";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&32u32.to_le_bytes());
        let name = b"token_embd.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name);
        buf.extend_from_slice(&2u32.to_le_bytes()); // rank
        buf.extend_from_slice(&256u64.to_le_bytes());
        buf.extend_from_slice(&(rows as u64).to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes()); // Q4_0 (ggml id 2, not 7)
        buf.extend_from_slice(&0u64.to_le_bytes());
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(&blocks);
        std::fs::write(&path, &buf).expect("write gguf");

        let model_arg = path.to_string_lossy().to_string();
        let cfg = parse_args(&args(&[
            "--model",
            &model_arg,
            "--rows",
            "16",
            "--max-new",
            "6",
            "--prompt-ids",
            "1,2,3",
        ]))
        .expect("parse");
        validate_model_path(&cfg).expect("valid");
        let (stream, _) = run_demo(&cfg).expect("demo on native gguf");
        assert!(stream.len() >= 9, "prompt + generated: {}", stream.len());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn prompt_text_requires_gguf() {
        let cfg = parse_args(&args(&["--model", "toy", "--prompt", "hello"])).expect("parse");
        assert!(
            resolve_prompt(&cfg).is_err(),
            "--prompt without GGUF must error"
        );
        let cfg2 = parse_args(&args(&["--model", "toy", "--prompt-ids", "1,2"])).expect("parse");
        assert_eq!(resolve_prompt(&cfg2).expect("ids"), vec![1, 2]);
    }

    #[test]
    fn demo_runs_with_dense_backend() {
        let cfg = parse_args(&args(&[
            "--model",
            "dense",
            "--max-new",
            "8",
            "--prompt-ids",
            "5,6,7",
        ]))
        .expect("parse");
        let (stream, _) = run_demo(&cfg).expect("dense demo");
        assert!(stream.len() >= 11, "prompt + generated: {}", stream.len());
        // The dense model's state carries real KV: verify it runs through
        // the scheduler with prefix reuse intact.
        let mut s = cfg.build_scheduler().expect("build");
        s.submit(&[5, 6, 7], 8).expect("submit");
        s.run_to_idle();
        assert!(s.stream_of(SequenceId(0)).expect("stream").len() >= 11);
    }

    #[test]
    fn demo_runs_with_moe_backend() {
        let cfg = parse_args(&args(&[
            "--model",
            "moe",
            "--max-new",
            "8",
            "--prompt-ids",
            "5,6,7",
        ]))
        .expect("parse");
        let (stream, _) = run_demo(&cfg).expect("moe demo");
        assert!(stream.len() >= 11, "prompt + generated: {}", stream.len());
    }

    #[test]
    fn demo_runs_with_flags() {
        let cfg = parse_args(&args(&[
            "--model",
            "toy",
            "--max-new",
            "8",
            "--prompt-ids",
            "5,6,7",
        ]))
        .expect("parse");
        let (stream, _) = run_demo(&cfg).expect("demo");
        assert!(stream.len() >= 11, "prompt + generated: {}", stream.len());
    }

    #[test]
    fn demo_runs_speculative() {
        let cfg = parse_args(&args(&[
            "--model",
            "toy",
            "--max-new",
            "8",
            "--spec-type",
            "draft",
            "--spec-draft-n-max",
            "3",
        ]))
        .expect("parse");
        let (stream, tel) = run_demo(&cfg).expect("demo");
        assert!(stream.len() >= 11);
        assert!(tel.is_some(), "draft mode returns telemetry");
    }

    #[test]
    fn demo_runs_speculative_auto_calibrated() {
        // Telemetry-driven calibration: the probe phase measures the
        // acceptance curve and the calibrated policy is applied before
        // the real serve.
        let cfg = parse_args(&args(&[
            "--model",
            "toy",
            "--max-new",
            "8",
            "--spec-type",
            "auto",
            "--spec-draft-n-max",
            "8",
        ]))
        .expect("parse");
        let (stream, tel) = run_demo(&cfg).expect("auto demo");
        assert!(stream.len() >= 11, "prompt + generated: {}", stream.len());
        let tel = tel.expect("auto mode returns telemetry");
        // Telemetry was reset after the probe, so it covers the
        // calibrated phase only.
        assert!(tel.drafted() > 0, "calibrated phase drafted tokens");
    }

    #[test]
    fn built_scheduler_auto_probe_then_calibrate() {
        let cfg = parse_args(&args(&[
            "--model",
            "toy",
            "--spec-type",
            "auto",
            "--spec-draft-n-max",
            "8",
        ]))
        .expect("parse");
        let mut s = cfg.build_scheduler().expect("build");
        s.submit(&[1, 2, 3], 32).expect("probe submit");
        s.run_to_idle();
        let (calibrated, curve) = s.calibrate_auto(0.5).expect("calibrate");
        assert!(curve.len() >= 2, "curve from probe: {curve:?}");
        assert!(
            calibrated.block <= 8 && calibrated.block >= 1,
            "block within probe horizon: {}",
            calibrated.block
        );
        assert!(
            calibrated.min_cap == 1 && calibrated.med_cap >= 1,
            "lower tiers scale down: med={} min={}",
            calibrated.med_cap,
            calibrated.min_cap
        );
        // A second calibration is a no-op (already calibrated).
        assert!(s.calibrate_auto(0.5).is_none(), "calibration runs once");
    }
}
