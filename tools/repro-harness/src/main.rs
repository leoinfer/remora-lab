//! Deterministic, model-free reproduction harnesses for the public research
//! lanes.  These programs deliberately do not load a model or call a foreign
//! runtime.  They exercise the public Rust format definitions and the
//! accounting/falsification rules that can be checked in CI.

use r4kv::f16::{f16_to_f32, f32_to_f16};
use r4kv::page::{pack_page, unpack_page, PageHeader, RestoreExpectation};
use r4kv::profiles::{ALL_PROFILES, N_KV_LAYERS_MAIN, N_MTP_KV_LAYERS};
use r4kv::quant::roundtrip_stats;
use r4kv::Fmt;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::env;

const HARNESS_SCHEMA: &str = "remora.repro_harness.v1";

fn main() {
    let mut args = env::args().skip(1);
    let command = args
        .next()
        .unwrap_or_else(|| usage_and_exit("missing command"));
    let options: Vec<String> = args.collect();
    let value = match command.as_str() {
        "r4kv" => r4kv_report(),
        "r4x-d32a" => r4x_report(),
        "context" => context_report(
            option_usize(&options, "records", 4096),
            option_usize(&options, "addressable", 10_000_000),
        ),
        "mtp" => mtp_report(
            option_usize(&options, "rounds", 263),
            option_usize(&options, "depth", 3),
        ),
        "ngram" => ngram_report(),
        "swmmac" => swmmac_report(),
        "help" | "--help" | "-h" => usage_and_exit(""),
        other => usage_and_exit(&format!("unknown command: {other}")),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&value).expect("serialize report")
    );
}

fn usage_and_exit(message: &str) -> ! {
    if !message.is_empty() {
        eprintln!("error: {message}");
    }
    eprintln!(
        "usage: repro-harness <r4kv|r4x-d32a|context|mtp|ngram|swmmac> [--records N --addressable N --rounds N --depth N]"
    );
    std::process::exit(if message.is_empty() { 0 } else { 2 });
}

fn option_usize(args: &[String], name: &str, default: usize) -> usize {
    let expected = format!("--{name}");
    let Some(index) = args.iter().position(|arg| arg == &expected) else {
        return default;
    };
    let raw = args
        .get(index + 1)
        .unwrap_or_else(|| usage_and_exit(&format!("{expected} requires a value")));
    raw.parse()
        .unwrap_or_else(|_| usage_and_exit(&format!("invalid value for {expected}")))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn sha256_f32(values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    sha256_bytes(&bytes)
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn seeded_float(index: usize) -> f32 {
    let x = (index as u64)
        .wrapping_mul(2_654_435_761)
        .wrapping_add(1_013_904_223);
    let word = splitmix64(x) as u32;
    (word as f32 / u32::MAX as f32) * 8.0 - 4.0
}

fn r4kv_report() -> Value {
    let source: Vec<f32> = (0..4096).map(seeded_float).collect();
    let positive: Vec<f32> = source.iter().map(|value| value.abs() + 0.125).collect();
    let mut profiles = Vec::new();
    for profile in ALL_PROFILES {
        let fmt_k = profile.k_fmt();
        let fmt_v = profile.v_fmt();
        let (encoded, _) = roundtrip_stats(&source, fmt_k);
        let (_, stats) = roundtrip_stats(&positive, fmt_v);
        let kl = normalized_kl(&positive, &roundtrip_positive(&positive, fmt_v));
        profiles.push(json!({
            "name": profile.name(),
            "k_format": format_fmt(fmt_k),
            "v_format": format_fmt(fmt_v),
            "main_bytes_per_token": profile.bytes_per_token(N_KV_LAYERS_MAIN, 0),
            "mtp_bytes_per_token": profile.bytes_per_token(N_KV_LAYERS_MAIN, N_MTP_KV_LAYERS),
            "encoded_fixture_bytes": encoded.len(),
            "value_roundtrip_max_abs_error": stats.max_abs_err,
            "value_roundtrip_rel_rms": stats.rel_rms(),
            "positive_proxy_forward_kl": kl,
            "quality_metric_scope": "synthetic positive-vector proxy; not a model perplexity or attention-quality result"
        }));
    }

    let header = PageHeader {
        prefix_digest: 0x1357_9bdf_2468_ace0,
        token_start: 32,
        token_count: 16,
        pos_start: 32,
        layer_lo: 0,
        layer_hi: 15,
        k_fmt: Fmt::Q6,
        v_fmt: Fmt::Q4,
        epoch: 4,
        generation: 9,
        payload_len: 256,
        sketch_offset: 0,
        flags: 0,
    };
    let body: Vec<u8> = (0..256).map(|i| (i as u8).wrapping_mul(17)).collect();
    let packed = pack_page(&header, &body);
    let (decoded_header, decoded_body) = unpack_page(&packed).expect("page KAT");
    let generation_ok = |generation: u32| generation == 9;
    let expectation = RestoreExpectation {
        prefix_digest: header.prefix_digest,
        token_start: header.token_start,
        pos_start: header.pos_start,
        layer_lo: header.layer_lo,
        layer_hi: header.layer_hi,
        k_fmt: Some(header.k_fmt),
        v_fmt: Some(header.v_fmt),
        generation: Some(&generation_ok),
    };
    let restore_ok = decoded_header
        .check_restore(&expectation, &decoded_body)
        .is_ok();
    let mut corrupted = packed.clone();
    let last = corrupted.len() - 1;
    corrupted[last] ^= 1;
    let corruption_rejected = unpack_page(&corrupted).is_err();

    json!({
        "schema": format!("{HARNESS_SCHEMA}.r4kv"),
        "lane": "r4kv",
        "status": "FULLY_REPRODUCIBLE",
        "seed": 0,
        "source_vector_elements": source.len(),
        "source_vector_sha256": sha256_f32(&source),
        "profiles": profiles,
        "page_known_answer": {
            "packed_bytes": packed.len(),
            "body_sha256": sha256_bytes(&body),
            "restore_gate_passed": restore_ok,
            "payload_corruption_rejected": corruption_rejected,
            "fail_closed": restore_ok && corruption_rejected
        },
        "interpretation": "Storage arithmetic, codec round-trip, proxy KL, and page identity checks are reproducible. They do not reproduce a full-model quality or GPU-parity claim."
    })
}

fn format_fmt(fmt: Fmt) -> &'static str {
    match fmt {
        Fmt::F16 => "F16",
        Fmt::Q8 => "Q8",
        Fmt::Q6 => "Q6",
        Fmt::Q4 => "Q4",
        Fmt::Q3 => "Q3",
    }
}

fn roundtrip_positive(source: &[f32], fmt: Fmt) -> Vec<f32> {
    let (decoded, _) = roundtrip_stats(source, fmt);
    decoded
}

fn normalized_kl(reference: &[f32], candidate: &[f32]) -> f64 {
    let ref_sum: f64 = reference.iter().map(|v| *v as f64).sum();
    let cand_sum: f64 = candidate.iter().map(|v| (*v as f64).max(1.0e-12)).sum();
    reference
        .iter()
        .zip(candidate.iter())
        .map(|(r, c)| {
            let p = (*r as f64 / ref_sum).max(1.0e-12);
            let q = ((*c as f64).max(1.0e-12) / cand_sum).max(1.0e-12);
            p * (p / q).ln()
        })
        .sum()
}

fn round_ties_even(value: f32) -> i32 {
    let lower = value.floor();
    let fraction = value - lower;
    if fraction < 0.5 {
        lower as i32
    } else if fraction > 0.5 {
        lower as i32 + 1
    } else if (lower as i64) % 2 == 0 {
        lower as i32
    } else {
        lower as i32 + 1
    }
}

fn r4x_encode(values: &[f32; 256]) -> (Vec<u8>, f32) {
    let mut scales = Vec::with_capacity(16);
    let mut payload = Vec::with_capacity(128);
    let mut worst = 0.0f32;
    for group in 0..8 {
        let start = group * 32;
        let chunk = &values[start..start + 32];
        let amax = chunk.iter().map(|value| value.abs()).fold(0.0, f32::max);
        let scale_bits = f32_to_f16(if amax == 0.0 { 0.0 } else { amax / 7.0 });
        let scale = f16_to_f32(scale_bits);
        scales.extend_from_slice(&scale_bits.to_le_bytes());
        for pair in 0..16 {
            let mut byte = 0u8;
            for lane in 0..2 {
                let value = chunk[pair * 2 + lane];
                let raw = if scale == 0.0 {
                    0
                } else {
                    round_ties_even(value / scale).clamp(-8, 7)
                };
                let decoded = scale * raw as f32;
                worst = worst.max((value - decoded).abs());
                let nibble = (raw + 8) as u8;
                byte |= nibble << (lane * 4);
            }
            payload.push(byte);
        }
    }
    scales.extend_from_slice(&payload);
    (scales, worst)
}

fn r4x_decode(bytes: &[u8]) -> [f32; 256] {
    assert_eq!(bytes.len(), 144);
    let mut out = [0.0f32; 256];
    for group in 0..8 {
        let scale = f16_to_f32(u16::from_le_bytes([bytes[group * 2], bytes[group * 2 + 1]]));
        let payload = 16 + group * 16;
        for pair in 0..16 {
            let packed = bytes[payload + pair];
            out[group * 32 + pair * 2] = scale * ((packed & 0x0f) as i32 - 8) as f32;
            out[group * 32 + pair * 2 + 1] = scale * ((packed >> 4) as i32 - 8) as f32;
        }
    }
    out
}

fn r4x_report() -> Value {
    let mut values = [0.0f32; 256];
    for (index, value) in values.iter_mut().enumerate() {
        *value = seeded_float(index) * 0.75;
    }
    values[0] = -7.0;
    values[1] = 7.0;
    let (encoded, max_abs_error) = r4x_encode(&values);
    let decoded = r4x_decode(&encoded);
    let decoded_bytes: Vec<u8> = decoded.iter().flat_map(|v| v.to_le_bytes()).collect();
    let exact_extrema = decoded[0] == -7.0 && decoded[1] == 7.0;
    let geometry = json!({
        "block_values": 256,
        "block_bytes": 144,
        "scale_count": 8,
        "scale_encoding": "IEEE-754 binary16 little-endian per 32-value group",
        "payload_bytes": 128,
        "nibble_order": "even element low nibble, odd element high nibble",
        "code_mapping": "signed code = nibble - 8, clamped to [-8, 7]",
        "scale_rule": "round-to-nearest-even binary16(max_abs(group) / 7)",
        "row_5120_bytes": 20 * 144,
        "row_151936_bytes": 151936usize.div_ceil(256) * 144
    });
    json!({
        "schema": format!("{HARNESS_SCHEMA}.r4x_d32a"),
        "lane": "r4x",
        "status": "FULLY_REPRODUCIBLE",
        "format_type_id": 36,
        "fixture_input_sha256": sha256_f32(&values),
        "fixture_encoded_sha256": sha256_bytes(&encoded),
        "fixture_decoded_sha256": sha256_bytes(&decoded_bytes),
        "max_abs_error": max_abs_error,
        "known_answer": {
            "encoded_len_is_144": encoded.len() == 144,
            "extrema_roundtrip": exact_extrema,
            "all_values_finite": decoded.iter().all(|value| value.is_finite())
        },
        "geometry": geometry,
        "scope": "Clean-room D32A vector/geometry KAT. It does not claim full-model compatibility, R4X-D/H/S/XP-S interoperability, QAT quality, or GPU throughput."
    })
}

fn context_report(records: usize, addressable: usize) -> Value {
    if records == 0 || addressable == 0 || records > addressable {
        usage_and_exit("context requires 0 < records <= addressable");
    }
    let mut entries: Vec<(u64, u64)> = (0..records)
        .map(|index| {
            let address = splitmix64(index as u64) % addressable as u64;
            (address, splitmix64(index as u64 ^ 0xa5a5_a5a5))
        })
        .collect();
    entries.sort_unstable();
    let encoded = encode_delta_addresses(&entries);
    let recovered = decode_delta_addresses(&encoded, entries.len()).expect("context decode");
    let recovery_ok = recovered == entries;
    let queries = records.min(512);
    let lexical_hits = (0..queries)
        .filter(|index| {
            let wanted = splitmix64(*index as u64) % addressable as u64;
            entries
                .binary_search_by_key(&wanted, |entry| entry.0)
                .is_ok()
        })
        .count();
    let shortcut = shortcut_probe();
    json!({
        "schema": format!("{HARNESS_SCHEMA}.context"),
        "lane": "effective-context",
        "status": "FULLY_REPRODUCIBLE",
        "seed": 0xa5a5_a5a5u64,
        "records": records,
        "addressable_positions": addressable,
        "stored_representation": "sorted address deltas + unsigned LEB128 values",
        "compressed_bytes": encoded.len(),
        "uncompressed_pair_bytes": entries.len() * 16,
        "recovered_digest": sha256_bytes(&encode_pairs(&recovered)),
        "recovery_exact": recovery_ok,
        "lexical_retrieval": { "queries": queries, "r1": lexical_hits as f64 / queries as f64 },
        "semantic_retrieval": { "r10": null, "status": "NOT_IMPLEMENTED_IN_THIS_MODEL_FREE_FIXTURE" },
        "shortcut_probe": shortcut,
        "claim_boundary": "addressable compressed/effective-context hierarchy; not dense transformer attention or a dense 10M-token KV cache"
    })
}

fn encode_delta_addresses(entries: &[(u64, u64)]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut previous = 0u64;
    for &(address, value) in entries {
        put_varint(address - previous, &mut out);
        put_varint(value, &mut out);
        previous = address;
    }
    out
}

fn encode_pairs(entries: &[(u64, u64)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(entries.len() * 16);
    for &(address, value) in entries {
        out.extend_from_slice(&address.to_le_bytes());
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn put_varint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn decode_delta_addresses(bytes: &[u8], count: usize) -> Option<Vec<(u64, u64)>> {
    let mut offset = 0usize;
    let mut previous = 0u64;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let (delta, used) = get_varint(&bytes[offset..])?;
        offset += used;
        let (value, used) = get_varint(&bytes[offset..])?;
        offset += used;
        let address = previous.checked_add(delta)?;
        out.push((address, value));
        previous = address;
    }
    (offset == bytes.len()).then_some(out)
}

fn get_varint(bytes: &[u8]) -> Option<(u64, usize)> {
    let mut value = 0u64;
    for (index, byte) in bytes.iter().copied().enumerate() {
        let shift = index.checked_mul(7)?;
        if shift >= 64 {
            return None;
        }
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Some((value, index + 1));
        }
    }
    None
}

fn shortcut_probe() -> Value {
    let mut solved = 0usize;
    for index in 0..100usize {
        let prefix = if index < 47 { "x" } else { "y" };
        // Every item has the same answer. The first 47 prefixes are
        // accidentally predictive; the remaining prefixes are adversarial.
        let true_label = true;
        let predicted = prefix == "x";
        if predicted == true_label {
            solved += 1;
        }
    }
    json!({
        "queries": 100,
        "solved_by_two_character_prefix": solved,
        "solved_fraction": solved as f64 / 100.0,
        "is_benchmark_failure_fixture": true,
        "meaning": "A shortcut that ignores the payload solves 47% of this intentionally weak probe; benchmark quality must be tested adversarially."
    })
}

fn mtp_report(rounds: usize, depth: usize) -> Value {
    if rounds == 0 || depth == 0 || depth > 16 {
        usage_and_exit("mtp requires 0 < depth <= 16 and rounds > 0");
    }
    let position_rates = [0.946f64, 0.694, 0.523];
    let mut accepted = 0usize;
    let mut draft = 0usize;
    let mut histogram = vec![0usize; depth + 1];
    let mut trace_digest = Vec::with_capacity(rounds * depth * 8);
    for round in 0..rounds {
        let horizon = if depth <= 3 {
            depth
        } else {
            1 + (splitmix64(round as u64) as usize % depth)
        };
        let mut accepted_this_round = 0usize;
        for position in 0..horizon {
            draft += 1;
            let rate = position_rates.get(position).copied().unwrap_or(0.45);
            let draw =
                (splitmix64((round as u64) << 8 | position as u64) % 10_000) as f64 / 10_000.0;
            let target = splitmix64((round as u64) << 16 | position as u64) as u32 % 97;
            let proposed = if draw < rate {
                target
            } else {
                target.wrapping_add(1)
            };
            trace_digest.extend_from_slice(&target.to_le_bytes());
            trace_digest.extend_from_slice(&proposed.to_le_bytes());
            if proposed != target {
                break;
            }
            accepted += 1;
            accepted_this_round += 1;
        }
        histogram[accepted_this_round] += 1;
    }
    let acceptance = accepted as f64 / draft.max(1) as f64;
    json!({
        "schema": format!("{HARNESS_SCHEMA}.mtp"),
        "lane": "mtp-accounting",
        "status": "FULLY_REPRODUCIBLE",
        "mechanism": "synthetic_neural_acceptance_accounting",
        "rounds": rounds,
        "maximum_draft_depth": depth,
        "draft_tokens": draft,
        "accepted_tokens": accepted,
        "acceptance_ratio": acceptance,
        "mean_accepted_per_round": accepted as f64 / rounds as f64,
        "per_position_reference_rates": position_rates,
        "accepted_prefix_rule": true,
        "horizon_histogram": histogram,
        "trace_sha256": sha256_bytes(&trace_digest),
        "historical_anchor_comparison": {
            "accepted_over_rounds": "historical 240/263 is not asserted by this synthetic fixture",
            "mean_accepted_over_draft": "historical ~3.16 is not asserted by this synthetic fixture"
        },
        "scope": "Acceptance and cost accounting only; no neural model, tokenizer, or throughput claim."
    })
}

fn ngram_report() -> Value {
    let workloads = [
        ("warm-shell", 300usize, 29usize),
        ("template", 300usize, 6usize),
    ];
    let mut reports = Vec::new();
    for (name, length, matching_period) in workloads {
        let mut accepted = 0usize;
        let mut trace = Vec::with_capacity(length * 8);
        for index in 0..length {
            let target = if name == "warm-shell" {
                [101u32, 32, 45, 45, 10][index % 5]
            } else {
                [60u32, 116, 97, 103, 62, 10, 123][index % 7]
            };
            let proposed = if index % matching_period == matching_period - 1 {
                target.wrapping_add(1)
            } else {
                target
            };
            accepted += usize::from(target == proposed);
            trace.extend_from_slice(&target.to_le_bytes());
            trace.extend_from_slice(&proposed.to_le_bytes());
        }
        reports.push(json!({
            "workload": name,
            "tokens": length,
            "accepted": accepted,
            "acceptance_ratio": accepted as f64 / length as f64,
            "trace_sha256": sha256_bytes(&trace),
            "throughput": null,
            "throughput_status": "NOT_MEASURED_BY_THIS_ACCOUNTING_FIXTURE"
        }));
    }
    json!({
        "schema": format!("{HARNESS_SCHEMA}.ngram"),
        "lane": "ngram-speculation",
        "status": "FULLY_REPRODUCIBLE",
        "mechanism": "workload_shaped_token_replay",
        "workloads": reports,
        "caveat": "These are exact synthetic shell/template sequences. They are not generic 27B neural decode and do not reproduce historical t/s values."
    })
}

fn swmmac_report() -> Value {
    let expected = 4u32;
    let replayed = 1u32;
    let corrected_useful = expected == replayed;
    let instruction_count = 13_107_331_072u64;
    let elapsed_ms = 705.9f64;
    let instructions_per_second = instruction_count as f64 / (elapsed_ms / 1000.0);
    let physical_mac_per_second = instructions_per_second * 4096.0;
    json!({
        "schema": format!("{HARNESS_SCHEMA}.swmmac"),
        "lane": "swmmac-gfx1200",
        "status": "FALSIFIED_REPRODUCIBLE",
        "known_answer": {
            "expected_committed_value": expected,
            "replayed_accumulator_value": replayed,
            "observed_over_expected": replayed as f64 / expected as f64,
            "smoking_gun_quarter_result": replayed * 4 == expected,
            "useful_claim_survives": corrected_useful
        },
        "accounting_boundary": {
            "instruction_activity_is_not_useful_work": true,
            "useful_work_requires_independent_accumulator_known_answer": true,
            "reference_sparse_int4_envelope_tops": 821,
            "reference_units": "advertised sparse INT4 TOPS; context only"
        },
        "historical_corrected_reference": {
            "instructions": instruction_count,
            "elapsed_ms": elapsed_ms,
            "instructions_per_second": instructions_per_second,
            "physical_mac_per_second": physical_mac_per_second,
            "device_receipt_rerun": false,
            "useful_llm_tops": null
        },
        "scope": "The Rust fixture reproduces the accumulator-replay falsifier. It is not a fresh GPU ISA measurement and makes no multi-POPS or LLM-throughput claim."
    })
}
