//! # lb — local-bench CLI
//!
//! Synthetic benchmarking of local models: inspect any GGUF header, simulate
//! any config against a hardware phenotype, and search the config space for
//! the fastest fit — without ever loading the model.
//!
//! ```text
//! lb inspect <model.gguf>
//! lb sim    <model.gguf> <hw.json> ngl=40 threads=8 batch=2048 ctx=196608 kv=q8_0 fa=on nmax=3
//! lb search <model.gguf> <hw.json> [--ctx 196608] [--beam 12] [--rounds 8]
//! lb calibrate <model.gguf> <hw.json> <anchors.json>
//! lb hw
//! ```

use lb_hardware::HardwareProfile;
use lb_model::ModelMeta;
use lb_search::{Levers, search};
use lb_sim::{Anchor, Calibration, Config, SpecKind, Workload, calibrate, simulate};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
        return;
    }
    let cmd = args[1].as_str();
    let res = match cmd {
        "inspect" => cmd_inspect(&args[2..]),
        "sim" => cmd_sim(&args[2..]),
        "search" => cmd_search(&args[2..]),
        "calibrate" => cmd_calibrate(&args[2..]),
        "measure" => cmd_measure(&args[2..]),
        "hw" => cmd_hw(),
        "help" | "--help" | "-h" => {
            usage();
            Ok(())
        }
        other => Err(format!("unknown command: {other}")),
    };
    if let Err(e) = res {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn usage() {
    println!(
        "local-bench — synthetic local-model benchmarking (no model execution)\n\
         \n\
         usage:\n\
         \x20 lb inspect <model.gguf>                     header-only model inventory\n\
         \x20 lb sim <model.gguf> <hw.json> [k=v ...]     simulate one config\n\
         \x20 lb search <model.gguf> <hw.json> [flags]    beam search best config\n\
         \x20 lb calibrate <model.gguf> <hw.json> <anchors.json>\n\
         \x20 lb hw                                      print the illustrative example profile\n\
         \n\
         sim keys: ngl threads batch ctx kv=f16|q8_0 fa=on|off kvoff=on|off spec=none|mtp nmax pmin mmap=on|off overlap\n\
         search flags: --ctx N --beam N --rounds N --prompt N --gen N"
    );
}

fn load_model(p: &str) -> Result<ModelMeta, String> {
    lb_model::inspect(Path::new(p))
}

fn load_hw(p: &str) -> Result<HardwareProfile, String> {
    if p == "example" {
        serde_json::from_str(lb_hardware::EXAMPLE_PROFILE).map_err(|e| e.to_string())
    } else {
        HardwareProfile::load(Path::new(p))
    }
}

fn cmd_inspect(args: &[String]) -> Result<(), String> {
    let path = args.first().ok_or("usage: lb inspect <model.gguf>")?;
    let m = load_model(path)?;
    println!("model:       {}", m.name);
    println!("arch:        {}", m.arch);
    println!(
        "file:        {:.2} GB ({} tensors, {:.2} GB tensor payload)",
        m.file_bytes as f64 / 1e9,
        m.tensor_count,
        m.total_tensor_bytes as f64 / 1e9
    );
    println!(
        "layers:      {} ({} attention)  embd={} heads={} kv_heads={} kv_len={}",
        m.n_layers,
        m.n_attention_layers,
        m.embedding_length,
        m.head_count,
        m.kv_heads,
        m.key_length
    );
    println!(
        "MTP/NextN:   {} ({:.1} MB)",
        if m.has_mtp { "yes" } else { "no" },
        m.mtp_bytes as f64 / 1e6
    );
    println!("quant mix:");
    for (q, v) in &m.quant_mix {
        println!(
            "  {q:<8} {:>3} tensors  {:>7.2} GB",
            v.tensor_count,
            v.bytes as f64 / 1e9
        );
    }
    println!(
        "KV/token:    {:.1} KB (f16) / {:.1} KB (q8_0)",
        m.kv_bytes_per_token(2, 2) as f64 / 1024.0,
        m.kv_bytes_per_token(1, 1) as f64 / 1024.0
    );
    println!("avg layer:   {:.1} MB", m.avg_layer_bytes() / 1e6);
    println!("params:      {:.2} B", m.total_elements as f64 / 1e9);
    Ok(())
}

fn parse_config(args: &[String], base: Config) -> Result<Config, String> {
    let mut c = base;
    for a in args {
        let Some((k, v)) = a.split_once('=') else {
            return Err(format!("bad key=value: {a}"));
        };
        match k {
            "ngl" => c.ngl = v.parse().map_err(|_| "ngl")?,
            "threads" | "t" => c.threads = v.parse().map_err(|_| "threads")?,
            "batch" | "b" => {
                c.batch = v.parse().map_err(|_| "batch")?;
                c.ubatch = c.batch;
            }
            "ctx" | "c" => c.ctx = v.parse().map_err(|_| "ctx")?,
            "kv" => {
                c.kv_q8 = match v {
                    "q8_0" | "q8" => true,
                    "f16" => false,
                    _ => return Err("kv must be f16|q8_0".into()),
                }
            }
            "fa" => {
                c.fa = match v {
                    "on" | "1" => true,
                    "off" | "0" => false,
                    _ => return Err("fa must be on|off".into()),
                }
            }
            "kvoff" => {
                c.kv_offload = match v {
                    "on" | "1" => true,
                    "off" | "0" => false,
                    _ => return Err("kvoff must be on|off".into()),
                }
            }
            "mmap" => {
                c.mmap = match v {
                    "on" | "1" => true,
                    "off" | "0" => false,
                    _ => return Err("mmap must be on|off".into()),
                }
            }
            "spec" => {
                c.spec = match v {
                    "none" => SpecKind::None,
                    "mtp" => SpecKind::Mtp,
                    _ => return Err("spec must be none|mtp".into()),
                }
            }
            "nmax" => c.n_max = v.parse().map_err(|_| "nmax")?,
            "pmin" => c.p_min = v.parse().map_err(|_| "pmin")?,
            "overlap" => c.overlap = v.parse().map_err(|_| "overlap")?,
            _ => return Err(format!("unknown key: {k}")),
        }
    }
    if c.n_max > 0 {
        c.spec = SpecKind::Mtp;
    }
    Ok(c)
}

fn cmd_sim(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err("usage: lb sim <model.gguf> <hw.json> [k=v ...]".into());
    }
    let m = load_model(&args[0])?;
    let hw = load_hw(&args[1])?;
    let cfg = parse_config(&args[2..], Config::default())?;
    let cal = Calibration::default();
    let wl = Workload::default();
    let r = simulate(&m, &hw, &cfg, &cal, &wl);
    println!(
        "{}",
        serde_json::to_string_pretty(&r).map_err(|e| e.to_string())?
    );
    Ok(())
}

fn cmd_search(args: &[String]) -> Result<(), String> {
    if args.len() < 2 {
        return Err("usage: lb search <model.gguf> <hw.json> [--ctx N --beam N --rounds N --prompt N --gen N]".into());
    }
    let m = load_model(&args[0])?;
    let hw = load_hw(&args[1])?;
    let mut ctx = 196_608usize;
    let mut beam = 12usize;
    let mut rounds = 8usize;
    let mut prompt = 2048usize;
    let mut n_gen = 512usize;
    let mut kv_q8 = true;
    let mut fa = true;
    let mut cal = Calibration::default();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--ctx" => {
                ctx = args
                    .get(i + 1)
                    .ok_or("--ctx N")?
                    .parse()
                    .map_err(|_| "--ctx")?;
                i += 2;
            }
            "--beam" => {
                beam = args
                    .get(i + 1)
                    .ok_or("--beam N")?
                    .parse()
                    .map_err(|_| "--beam")?;
                i += 2;
            }
            "--rounds" => {
                rounds = args
                    .get(i + 1)
                    .ok_or("--rounds N")?
                    .parse()
                    .map_err(|_| "--rounds")?;
                i += 2;
            }
            "--prompt" => {
                prompt = args
                    .get(i + 1)
                    .ok_or("--prompt N")?
                    .parse()
                    .map_err(|_| "--prompt")?;
                i += 2;
            }
            "--gen" => {
                n_gen = args
                    .get(i + 1)
                    .ok_or("--gen N")?
                    .parse()
                    .map_err(|_| "--gen")?;
                i += 2;
            }
            "--kv-f16" => {
                kv_q8 = false;
                i += 1;
            }
            "--fa-off" => {
                fa = false;
                i += 1;
            }
            "--cal" => {
                let p = args.get(i + 1).ok_or("--cal PATH")?;
                let raw = std::fs::read_to_string(p).map_err(|e| e.to_string())?;
                cal = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
                i += 2;
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }
    let wl = Workload {
        prompt_tokens: prompt,
        gen_tokens: n_gen,
    };
    let mut lv = Levers::default();
    while i < args.len() {
        match args[i].as_str() {
            "--dual-channel" => {
                lv.dual_channel = true;
                i += 1;
            }
            "--overlap" => {
                lv.overlap = Some(
                    args.get(i + 1)
                        .ok_or("--overlap X")?
                        .parse()
                        .map_err(|_| "--overlap")?,
                );
                i += 2;
            }
            "--bytes-scale" => {
                lv.bytes_scale = Some(
                    args.get(i + 1)
                        .ok_or("--bytes-scale X")?
                        .parse()
                        .map_err(|_| "--bytes-scale")?,
                );
                i += 2;
            }
            "--graph-cut" => {
                lv.graph_cut = Some(
                    args.get(i + 1)
                        .ok_or("--graph-cut X")?
                        .parse()
                        .map_err(|_| "--graph-cut")?,
                );
                i += 2;
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }
    println!("levers: {}", lv.label());
    let hits = search(&m, &hw, &cal, &wl, ctx, kv_q8, fa, beam, rounds, &lv);
    println!(
        "{:<4} {:<6} {:<4} {:<6} {:<5} {:<6} {:<7} {:<7} {:<12} {:<8} {:<7}",
        "rank",
        "ngl",
        "t",
        "batch",
        "nmax",
        "ctx",
        "decode",
        "prefill",
        "bottleneck",
        "vramGB",
        "wall_s"
    );
    for h in &hits {
        println!(
            "{:<4} {:<6} {:<4} {:<6} {:<5} {:<6} {:<7.2} {:<7.1} {:<12} {:<8.1} {:<7.0}",
            h.rank,
            h.config.ngl,
            h.config.threads,
            h.config.batch,
            h.config.n_max,
            h.config.ctx,
            h.decode_tok_s,
            h.prefill_tok_s,
            h.bottleneck,
            h.vram_gb,
            h.est_wall_s,
        );
    }
    if let Some(top) = hits.first() {
        println!("\nresearch paths for #1 ({}):", top.bottleneck);
        for l in &top.levers {
            println!("  - {l}");
        }
        if !top.risks.is_empty() {
            println!("risks:");
            for r in &top.risks {
                println!("  ! {r}");
            }
        }
    }
    Ok(())
}

fn cmd_calibrate(args: &[String]) -> Result<(), String> {
    if args.len() < 3 {
        return Err("usage: lb calibrate <model.gguf> <hw.json> <anchors.json>".into());
    }
    let m = load_model(&args[0])?;
    let hw = load_hw(&args[1])?;
    let raw = std::fs::read_to_string(&args[2]).map_err(|e| e.to_string())?;
    let anchors: Vec<Anchor> = serde_json::from_str(&raw)
        .or_else(|_| {
            serde_json::from_str::<serde_json::Value>(&raw)
                .and_then(|v| serde_json::from_value(v["anchors"].clone()))
        })
        .map_err(|e| format!("anchors must be an array or {{anchors:[...]}}: {e}"))?;
    let (cal, rmse) = calibrate(&m, &hw, &anchors);
    println!("calibrated (relative RMSE {:.3}):", rmse);
    let out = serde_json::to_string_pretty(&cal).map_err(|e| e.to_string())?;
    println!("{out}");
    if let Some(path) = args.get(3) {
        std::fs::write(path, out).map_err(|e| e.to_string())?;
        println!("wrote calibration: {path}");
    }
    Ok(())
}

fn cmd_measure(args: &[String]) -> Result<(), String> {
    // usage: lb measure <model.gguf> <configs.json> --bin PATH --out anchors.json [--prompt "..."]
    if args.len() < 2 {
        return Err("usage: lb measure <model.gguf> <configs.json> --bin PATH --out anchors.json [--prompt TEXT]".into());
    }
    let model = &args[0];
    let raw = std::fs::read_to_string(&args[1]).map_err(|e| e.to_string())?;
    let cfgs: Vec<(String, Config)> = serde_json::from_str::<Vec<serde_json::Value>>(&raw)
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|v| {
            let label = v["label"].as_str().unwrap_or("run").to_string();
            let cfg: Config =
                serde_json::from_value(v["config"].clone()).map_err(|e| e.to_string())?;
            Ok((label, cfg))
        })
        .collect::<Result<_, String>>()?;
    let mut bin = String::new();
    let mut out = String::from("anchors.json");
    let mut prompt = String::from(
        "Continue this technical note with exact, concise prose. The subject is deterministic local inference, resident memory, and streamed tensor transport. State assumptions explicitly and avoid headings.",
    );
    let mut prompt_tokens = 0usize;
    let mut port = 8102usize;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--bin" => {
                bin = args.get(i + 1).ok_or("--bin PATH")?.clone();
                i += 2;
            }
            "--out" => {
                out = args.get(i + 1).ok_or("--out PATH")?.clone();
                i += 2;
            }
            "--prompt" => {
                prompt = args.get(i + 1).ok_or("--prompt TEXT")?.clone();
                i += 2;
            }
            "--port" => {
                port = args
                    .get(i + 1)
                    .ok_or("--port N")?
                    .parse()
                    .map_err(|_| "--port")?;
                i += 2;
            }
            "--prompt-tokens" => {
                prompt_tokens = args
                    .get(i + 1)
                    .ok_or("--prompt-tokens N")?
                    .parse()
                    .map_err(|_| "--prompt-tokens")?;
                i += 2;
            }
            other => return Err(format!("unknown flag: {other}")),
        }
    }
    if bin.is_empty() {
        return Err("--bin PATH required (external serving binary)".into());
    }
    if prompt_tokens > 0 {
        // build a longer prompt by repetition so prefill measures amortized
        // throughput, not fixed context-setup overhead.
        let unit = prompt.clone();
        let mut n = 1;
        while n < 2 * prompt_tokens / unit.split_whitespace().count().max(1) {
            prompt.push_str(&format!(" {unit}"));
            n += 1;
        }
        println!(
            "prompt built to ~{} words",
            prompt.split_whitespace().count()
        );
    }
    let mut anchors = Vec::new();
    for (label, cfg) in &cfgs {
        println!("measuring {label} ...");
        let res = measure_one(model, &bin, cfg, &prompt, port)?;
        port += 1;
        println!(
            "  {label}: decode={:.3} prefill={:.2} tok/s",
            res.decode_tok_s, res.prefill_tok_s
        );
        anchors.push(Anchor {
            config: cfg.clone(),
            decode_tok_s: res.decode_tok_s,
            prefill_tok_s: res.prefill_tok_s,
            label: label.clone(),
        });
    }
    let payload = serde_json::json!({
        "schema": "local-bench.measured_anchors.v1",
        "measured_at": chrono_like_now(),
        "model": model,
        "binary": bin,
        "anchors": anchors,
    });
    std::fs::write(
        &out,
        serde_json::to_string_pretty(&payload).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    println!("wrote {out}");
    Ok(())
}

fn chrono_like_now() -> String {
    // no chrono dependency: seconds since epoch is enough for anchor provenance
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "?".into())
}

struct Measured {
    decode_tok_s: f64,
    prefill_tok_s: f64,
}

fn measure_one(
    model: &str,
    bin: &str,
    cfg: &Config,
    prompt: &str,
    port: usize,
) -> Result<Measured, String> {
    use std::process::{Command, Stdio};
    let fa = "-fa";
    let mut cmd = Command::new(bin);
    cmd.args([
        "-m",
        model,
        "-ngl",
        &cfg.ngl.to_string(),
        "-c",
        &cfg.ctx.to_string(),
        "-b",
        &cfg.batch.to_string(),
        "-ub",
        &cfg.ubatch.to_string(),
        "-t",
        &cfg.threads.to_string(),
        "-tb",
        &cfg.threads.to_string(),
        fa,
        if cfg.fa { "on" } else { "off" },
        "-ctk",
        if cfg.kv_q8 { "q8_0" } else { "f16" },
        "-ctv",
        if cfg.kv_q8 { "q8_0" } else { "f16" },
    ]);
    if cfg.mmap {
        cmd.arg("--mmap");
    }
    if !cfg.kv_offload {
        cmd.arg("--no-kv-offload");
    }
    cmd.args([
        "--no-warmup",
        "--host",
        "127.0.0.1",
        "--port",
        &port.to_string(),
        "--parallel",
        "1",
    ]);
    match cfg.spec {
        SpecKind::None => {
            cmd.args(["--spec-type", "none"]);
        }
        SpecKind::Mtp => {
            cmd.args([
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                &cfg.n_max.to_string(),
                "--spec-draft-n-min",
                "1",
                "--spec-draft-p-min",
                &format!("{:.2}", cfg.p_min),
            ]);
        }
    }
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("spawn external serving binary: {e}"))?;

    // wait for health
    let mut ok = false;
    for _ in 0..300 {
        let h = Command::new("curl")
            .args(["-s", "-m", "1", &format!("http://127.0.0.1:{port}/health")])
            .output();
        if let Ok(o) = h {
            let body = String::from_utf8_lossy(&o.stdout);
            if body.contains("\"status\":\"ok\"") || body.trim() == "ok" {
                ok = true;
                break;
            }
        }
        if child.try_wait().map_err(|e| e.to_string())?.is_some() {
            return Err("external serving binary exited during startup".into());
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
    if !ok {
        let _ = child.kill();
        return Err("health timeout".into());
    }
    let req = serde_json::json!({
        "prompt": prompt,
        "n_predict": 256,
        "temperature": 0, "top_k": 1, "top_p": 1, "min_p": 0,
        "repeat_penalty": 1, "dry_multiplier": 0, "seed": 42,
        "ignore_eos": true, "return_tokens": true, "stream": false, "cache_prompt": false,
    });
    let resp = Command::new("curl")
        .args([
            "-s",
            "-X",
            "POST",
            &format!("http://127.0.0.1:{port}/completion"),
            "-H",
            "Content-Type: application/json",
            "-d",
            &req.to_string(),
        ])
        .output()
        .map_err(|e| e.to_string())?;
    let _ = child.kill();
    let _ = child.wait();
    let d: serde_json::Value =
        serde_json::from_slice(&resp.stdout).map_err(|e| format!("bad completion JSON: {e}"))?;
    let _ = &d;
    let t = &d["timings"];
    let decode = t["predicted_per_second"].as_f64().unwrap_or(0.0);
    let prefill = t["prompt_per_second"].as_f64().unwrap_or(0.0);
    if decode <= 0.0 || prefill <= 0.0 {
        eprintln!(
            "  ! server returned no timings: {}",
            serde_json::to_string(&d)
                .unwrap_or_default()
                .chars()
                .take(200)
                .collect::<String>()
        );
    }
    Ok(Measured {
        decode_tok_s: decode,
        prefill_tok_s: prefill,
    })
}

fn cmd_hw() -> Result<(), String> {
    let hw: HardwareProfile =
        serde_json::from_str(lb_hardware::EXAMPLE_PROFILE).map_err(|e| e.to_string())?;
    println!(
        "{}",
        serde_json::to_string_pretty(&hw).map_err(|e| e.to_string())?
    );
    Ok(())
}
