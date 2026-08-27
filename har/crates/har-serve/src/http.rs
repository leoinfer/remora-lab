//! Minimal HTTP serving for `har-server` (std-only, no async deps).
//!
//! One persistent `ServeScheduler` serves all requests, so the prefix graph
//! accumulates across calls. This makes warm prefix reuse an observable
//! server behavior rather than a benchmark-only feature.
//!
//! Endpoint: `POST /v1/completions` with JSON
//! `{"prompt": "<text>" | "prompt_ids": [..], "max_tokens": n}` →
//! `{"choices": [{"text": "..."}], "tokens": [...]}`.
//!
//! Sequential v1 serving: one generation at a time.

use crate::server::ServerConfig;
use crate::tokenizer::Tokenizer;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};

/// Handle one HTTP connection.
fn handle_connection(
    mut stream: TcpStream,
    scheduler: &mut crate::server::BuiltScheduler,
    tokenizer: Option<&Tokenizer>,
) {
    let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    // Headers.
    let mut content_length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
            break;
        }
        if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; content_length];
    if reader.read_exact(&mut body).is_err() {
        return;
    }

    let response = match serde_json::from_slice::<serde_json::Value>(&body) {
        Ok(req) => {
            let max_tokens = req.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(32) as usize;
            let prompt: Result<Vec<u32>, String> = if let Some(ids) = req.get("prompt_ids") {
                ids.as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_u64())
                            .map(|v| v as u32)
                            .collect()
                    })
                    .ok_or("prompt_ids must be an array".into())
            } else if let Some(text) = req.get("prompt").and_then(|v| v.as_str()) {
                match tokenizer {
                    Some(t) => t.encode(text),
                    None => Err("tokenizer unavailable for this model".into()),
                }
            } else {
                Err("need prompt or prompt_ids".into())
            };
            match prompt {
                Ok(ids) => {
                    // Streaming mode: one SSE event
                    // per generated token, flushed as it is produced.
                    if req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false) {
                        handle_streaming(&mut stream, scheduler, tokenizer, &ids, max_tokens);
                        return;
                    }
                    let stream = serve_one(scheduler, &ids, max_tokens);
                    match stream {
                        Ok(tokens) => {
                            let text = tokenizer
                                .map(|t| t.decode(&tokens))
                                .unwrap_or_else(|| format!("{tokens:?}"));
                            let payload = serde_json::json!({
                                "choices": [{"text": text}],
                                "tokens": tokens,
                            });
                            format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                                payload.to_string().len(),
                                payload
                            )
                        }
                        Err(e) => json_error(&e),
                    }
                }
                Err(e) => json_error(&e),
            }
        }
        Err(e) => json_error(&format!("bad JSON: {e}")),
    };
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

/// SSE streaming: `data: {"token": N, "text": "..."}` per generated
/// token (flushed immediately), then `data: [DONE]`.  The scheduler is
/// stepped manually so tokens reach the client as they are produced
/// (accepted speculative drafts included — the stream diff captures
/// them).  Per-token decode may split multi-byte UTF-8 across token
/// boundaries; v1 displays lossy text per event.
fn handle_streaming(
    stream: &mut TcpStream,
    scheduler: &mut crate::server::BuiltScheduler,
    tokenizer: Option<&Tokenizer>,
    prompt: &[u32],
    max_tokens: usize,
) {
    let id = match scheduler.submit(prompt, max_tokens) {
        Ok(id) => id,
        Err(e) => {
            let _ = stream.write_all(json_error(&format!("submit: {e}")).as_bytes());
            return;
        }
    };
    let _ = stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
    );
    let mut last_len = prompt.len();
    loop {
        let report = scheduler.step();
        if report.kind == crate::StepKind::Idle {
            break;
        }
        let full = match scheduler.stream_of(id) {
            Ok(f) => f,
            Err(_) => break,
        };
        if full.len() > last_len {
            for &t in &full[last_len..] {
                let text = tokenizer.map(|tok| tok.decode(&[t])).unwrap_or_default();
                let _ = write!(stream, "data: {{\"token\":{t},\"text\":{text:?}}}\n\n");
            }
            last_len = full.len();
            let _ = stream.flush();
        }
    }
    let _ = stream.write_all(b"data: [DONE]\n\n");
    let _ = stream.flush();
}

fn json_error(message: &str) -> String {
    let payload = serde_json::json!({ "error": message });
    format!(
        "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        payload.to_string().len(),
        payload
    )
}

/// Submit one prompt to the persistent scheduler and collect the stream.
fn serve_one(
    scheduler: &mut crate::server::BuiltScheduler,
    prompt: &[u32],
    max_tokens: usize,
) -> Result<Vec<u32>, String> {
    let id = scheduler
        .submit(prompt, max_tokens)
        .map_err(|e| format!("submit: {e}"))?;
    scheduler.run_to_idle();
    let stream = scheduler
        .stream_of(id)
        .map_err(|e| format!("stream: {e}"))?;
    Ok(stream[prompt.len()..].to_vec())
}

/// Serve HTTP until the listener closes.  `--port <n>` on har-server.
///
/// `--spec-type auto`: the FIRST request doubles as the calibration
/// probe — served ungated, its acceptance curve derives the calibrated
/// tiered horizon, which applies from the second request on.
pub fn serve_http(cfg: &ServerConfig, listener: TcpListener) -> Result<(), String> {
    let mut scheduler = cfg.build_scheduler()?;

    // Tokenizer for text prompts (GGUF models only).
    let tokenizer = match &cfg.backend {
        crate::server::BackendKind::Gguf { path } => {
            let phenotype = har_model::GgufReader::new(path.clone())
                .inspect(false)
                .map_err(|e| format!("GGUF inspect: {e}"))?;
            phenotype.tokenizer.as_ref().and_then(Tokenizer::from_gguf)
        }
        _ => None,
    };

    let mut auto_pending = matches!(cfg.spec_type, crate::server::SpecType::Auto);
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                handle_connection(stream, &mut scheduler, tokenizer.as_ref());
                // Auto mode: the first (probe) request just finished —
                // derive the calibrated tiered horizon from its curve.
                if auto_pending {
                    match scheduler.calibrate_auto(0.5) {
                        Some((calibrated, curve)) => {
                            eprintln!(
                                "[auto] probe acceptance curve: {:?}",
                                curve.iter().map(|p| format!("{p:.2}")).collect::<Vec<_>>()
                            );
                            eprintln!(
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
                            // The guard's verdict: the draft accepted
                            // nothing.  Fall back to plain decode (the
                            // probe prefix states are dropped; the
                            // accumulator restarts cold).
                            eprintln!(
                                "[auto] probe: draft acceptance is zero (adversarial draft) — falling back to plain decode"
                            );
                            let mut plain = cfg.clone();
                            plain.spec_type = crate::server::SpecType::None;
                            scheduler = plain.build_scheduler()?;
                        }
                    }
                    auto_pending = false;
                }
            }
            Err(_) => break,
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::parse_args;
    use std::io::Write;
    use std::net::TcpStream;

    #[test]
    fn http_completion_roundtrip() {
        let cfg = parse_args(&["--model=toy", "--max-new=6"].map(String::from)).expect("cfg");
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let handle = std::thread::spawn(move || serve_http(&cfg, listener).expect("serve"));

        let mut stream = TcpStream::connect(addr).expect("connect");
        let body = r#"{"prompt_ids":[5,6,7],"max_tokens":6}"#;
        let request = format!(
            "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(request.as_bytes()).expect("write");
        stream.flush().expect("flush");
        let mut reader = BufReader::new(stream);
        let mut response = String::new();
        reader.read_to_string(&mut response).expect("read");
        assert!(response.contains("200 OK"), "status: {response}");
        assert!(response.contains("\"tokens\""), "tokens in response");
        assert!(response.contains("\"choices\""), "choices in response");
        drop(handle);
    }

    #[test]
    fn http_requires_prompt_field() {
        let cfg = parse_args(&["--model=toy"].map(String::from)).expect("cfg");
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let handle = std::thread::spawn(move || serve_http(&cfg, listener).expect("serve"));

        let mut stream = TcpStream::connect(addr).expect("connect");
        let body = r#"{"max_tokens":2}"#;
        let request = format!(
            "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(request.as_bytes()).expect("write");
        stream.flush().expect("flush");
        let mut reader = BufReader::new(stream);
        let mut response = String::new();
        reader.read_to_string(&mut response).expect("read");
        assert!(response.contains("400 Bad Request"), "status: {response}");
        assert!(response.contains("\"error\""), "error field");
        drop(handle);
    }

    #[test]
    fn http_streaming_emits_per_token_events() {
        // SSE parity: every generated token arrives as its own event,
        // in order, followed by [DONE]; the token ids match the
        // non-streaming response exactly.
        let cfg = parse_args(&["--model=toy", "--max-new=8"].map(String::from)).expect("cfg");
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let handle = std::thread::spawn(move || serve_http(&cfg, listener).expect("serve"));

        // Non-streaming reference.
        let mut stream = TcpStream::connect(addr).expect("connect");
        let body = r#"{"prompt_ids":[5,6,7],"max_tokens":6}"#;
        let request = format!(
            "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(request.as_bytes()).expect("write");
        stream.flush().expect("flush");
        let mut reader = BufReader::new(stream);
        let mut response = String::new();
        reader.read_to_string(&mut response).expect("read");
        let tokens: Vec<u32> = serde_json::from_slice::<serde_json::Value>(
            response.split("\r\n\r\n").nth(1).expect("body").as_bytes(),
        )
        .expect("json")
        .get("tokens")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64())
                .map(|v| v as u32)
                .collect()
        })
        .expect("tokens array");

        // Streaming request.
        let mut stream = TcpStream::connect(addr).expect("connect");
        let body = r#"{"prompt_ids":[5,6,7],"max_tokens":6,"stream":true}"#;
        let request = format!(
            "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        stream.write_all(request.as_bytes()).expect("write");
        stream.flush().expect("flush");
        let mut reader = BufReader::new(stream);
        let mut response = String::new();
        reader.read_to_string(&mut response).expect("read");

        assert!(response.contains("text/event-stream"), "SSE content type");
        let streamed: Vec<u32> = response
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter(|l| l.starts_with("{\"token\""))
            .filter_map(|l| {
                serde_json::from_str::<serde_json::Value>(l)
                    .ok()
                    .and_then(|v| v.get("token").and_then(|t| t.as_u64()))
                    .map(|t| t as u32)
            })
            .collect();
        assert_eq!(streamed, tokens, "streamed tokens match the batch response");
        assert!(
            response.contains("data: [DONE]"),
            "stream terminates with [DONE]"
        );
        drop(handle);
    }

    #[test]
    fn http_auto_probe_then_calibrated_requests() {
        // First request doubles as the calibration probe; the second is
        // served under the calibrated policy.  Both must produce valid
        // completions.
        let cfg = parse_args(
            &["--model=toy", "--spec-type=auto", "--spec-draft-n-max=8"].map(String::from),
        )
        .expect("cfg");
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let handle = std::thread::spawn(move || serve_http(&cfg, listener).expect("serve"));

        let mut responses = Vec::new();
        for _ in 0..2 {
            let mut stream = TcpStream::connect(addr).expect("connect");
            let body = r#"{"prompt_ids":[5,6,7],"max_tokens":6}"#;
            let request = format!(
                "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(request.as_bytes()).expect("write");
            stream.flush().expect("flush");
            let mut reader = BufReader::new(stream);
            let mut response = String::new();
            reader.read_to_string(&mut response).expect("read");
            assert!(response.contains("200 OK"), "status: {response}");
            responses.push(response);
        }
        assert_eq!(
            responses[0], responses[1],
            "deterministic model: probe and calibrated runs must agree"
        );
        drop(handle);
    }
}
