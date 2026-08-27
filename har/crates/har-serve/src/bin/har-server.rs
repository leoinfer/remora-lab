//! Native server entry point for the HAR serving runtime.
//!
//! ```text
//! # CLI demo (one prompt, telemetry):
//! cargo run -p har-serve --bin har-server -- \
//!   --model toy --batch 4 --chunked-prefill 16 \
//!   --spec-type draft --spec-draft-n-max 3 --spec-draft-p-min 0.75 \
//!   --ctx 32768 --kv-dtype q8_0 --cache-bytes 1048576 \
//!   --prompt-ids 1,2,3,4 --max-new 32
//!
//! # HTTP server (OpenAI-style completions):
//! cargo run -p har-serve --bin har-server -- \
//!   --model toy --host 127.0.0.1 --port 8080 --batch 4
//! curl -s localhost:8080/v1/completions \
//!   -d '{"prompt_ids":[1,2,3],"max_tokens":16}'
//! ```
//!
//! The resolved configuration is printed on startup.
//! In server mode one persistent scheduler serves all requests, so the
//! prefix graph accumulates across calls (warm prefix reuse).  Backends
//! today: `toy`, `q4k` (synthetic Q4_K block), `q40` (synthetic
//! synthetic Q4_0), native `.gguf` files. The Vulkan kernels plug in
//! behind the same `BatchStepModel` seam without changing the flag
//! surface.

use har_serve::server::{parse_args, run_demo, validate_model_path, ServerConfig};
use std::env;

fn print_banner(cfg: &ServerConfig) {
    println!("har-server (native HAR serving runtime)");
    println!("  model:              {}", cfg.model);
    println!("  batch:              {}", cfg.max_batch);
    println!("  chunked-prefill:    {}", cfg.prefill_chunk);
    println!("  page-size:          {}", cfg.page_size);
    println!("  kv-dtype:           {}", cfg.kv_type);
    println!("  context:            {}", cfg.target_context);
    println!(
        "  cache-bytes:        {}",
        cfg.cache_bytes
            .map_or("unbounded".into(), |v| v.to_string())
    );
    println!(
        "  live-state-bytes:   {}",
        cfg.live_state_bytes
            .map_or("unbounded".into(), |v| v.to_string())
    );
    println!("  pin-live:           {}", cfg.pin_live);
    println!(
        "  spec-type:          {}",
        match cfg.spec_type {
            har_serve::server::SpecType::None => "none".to_string(),
            har_serve::server::SpecType::Draft => format!(
                "draft (n-max {}, p-min {})",
                cfg.spec_draft_n_max, cfg.spec_draft_p_min
            ),
            har_serve::server::SpecType::Auto => format!(
                "auto (probe n-max {}, then calibrated tiers)",
                cfg.spec_draft_n_max
            ),
        }
    );
    println!(
        "  http:               {}",
        cfg.port.map_or("off (CLI demo)".into(), |p| format!(
            "{}:{} (POST /v1/completions)",
            cfg.host, p
        ))
    );
    println!("  max-new:            {}", cfg.max_new);
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let cfg = parse_args(&args)?;
    validate_model_path(&cfg)?;
    print_banner(&cfg);

    // Server mode: `--port` switches from the one-shot CLI demo to a
    // persistent HTTP server with a cross-request scheduler.
    if let Some(port) = cfg.port {
        let addr = format!("{}:{}", cfg.host, port);
        let listener =
            std::net::TcpListener::bind(&addr).map_err(|e| format!("bind {addr}: {e}"))?;
        println!("\nlistening on http://{addr}  (POST /v1/completions)");
        return har_serve::http::serve_http(&cfg, listener);
    }

    let start = std::time::Instant::now();
    let (stream, spec_telemetry) = run_demo(&cfg)?;
    let elapsed = start.elapsed();

    println!(
        "\nserved {} tokens in {:.1} ms ({:.1} tok/s)",
        stream.len(),
        elapsed.as_secs_f64() * 1e3,
        stream.len() as f64 / elapsed.as_secs_f64().max(1e-9)
    );
    println!("token stream: {:?}", stream);

    if let Some(tel) = spec_telemetry {
        println!(
            "spec: drafted={} accepted={} acceptance_length={:.2} target_passes={}",
            tel.drafted(),
            tel.accepted(),
            tel.acceptance_length(),
            tel.target_rows()
        );
    }
    Ok(())
}
