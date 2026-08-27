//! Release gate for the public reproducibility layer.
//!
//! The publication audit checks privacy and payload exclusions. This separate
//! gate checks that every declared lane has a machine-readable disposition and
//! that runnable/falsified lanes carry executable evidence rather than prose
//! alone.

use serde_json::Value;
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const STATUSES: &[&str] = &[
    "FULLY_REPRODUCIBLE",
    "REPRODUCIBLE_WITH_PUBLIC_MODEL_DOWNLOAD",
    "REPRODUCIBLE_ON_REFERENCE_HARDWARE",
    "REPRODUCIBLE_WITH_HARDWARE_RETUNING",
    "HISTORICAL_RECONSTRUCTION_AVAILABLE",
    "PROPOSED_NO_RESULT_YET",
    "FALSIFIED_REPRODUCIBLE",
    "BLOCKED_PROVENANCE",
    "BLOCKED_LICENSE",
    "EXCLUDED_PRIVACY",
    "EXCLUDED_WEIGHTS_DATA",
    "UNRECOVERABLE_HISTORICAL_RESULT",
];

const CLAIM_FIELDS: &[&str] = &[
    "claim_id",
    "experiment_id",
    "status",
    "description",
    "source_commit",
    "source_paths",
    "model_name",
    "model_revision",
    "model_download_source",
    "model_hashes",
    "quant_name",
    "quant_format_version",
    "quant_hash",
    "hardware_phenotype",
    "os",
    "kernel",
    "mesa",
    "radv",
    "vulkan",
    "rust",
    "compiler",
    "gpu_configuration",
    "power_limit",
    "clock_configuration",
    "memory_clock",
    "voltage_offset",
    "command",
    "environment_variables",
    "prompt",
    "prompt_hash",
    "context_size",
    "batch_size",
    "ubatch_size",
    "kv_format",
    "mtp_enabled",
    "ngram_enabled",
    "warmup",
    "measurement_start",
    "measurement_end",
    "tokens_generated",
    "prefill_tokens",
    "decode_tokens",
    "ttft_ms",
    "prefill_tps",
    "decode_tps",
    "acceptance_metrics",
    "raw_output_digest",
    "expected_range",
    "tolerance",
    "known_limitations",
    "receipt_path",
];

const TECHNICAL_INDEX_FIELDS: &[&str] = &[
    "artifact_id",
    "artifact_type",
    "research_family",
    "experiment_series",
    "experiment_id",
    "claim_ids",
    "implementation_path",
    "repro_path",
    "receipt_path",
    "status",
    "metric_boundary",
];

#[derive(Default)]
struct Audit {
    findings: Vec<String>,
    experiment_count: usize,
    status_counts: std::collections::BTreeMap<String, usize>,
}

fn main() {
    let root = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| env::current_dir().expect("current directory"));
    let root = root
        .canonicalize()
        .unwrap_or_else(|error| panic!("cannot resolve {}: {error}", root.display()));
    let mut audit = Audit::default();
    let root_manifest = root.join("repro_manifest.json");
    let status_doc = root.join("REPRODUCIBILITY_STATUS.md");
    let value = read_json(&root_manifest, &mut audit);
    if let Some(value) = value.as_ref() {
        audit_root_manifest(&root, value, &mut audit);
    }
    let claims = read_json(&root.join("claims.json"), &mut audit);
    let technical_index = read_json(&root.join("technical_artifact_index.json"), &mut audit);
    if let (Some(index), Some(claims)) = (technical_index.as_ref(), claims.as_ref()) {
        audit_technical_index(&root, index, claims, &mut audit);
    }
    if !status_doc.exists() {
        audit
            .findings
            .push("REPRODUCIBILITY_STATUS.md is missing".into());
    }
    let required_root_files = [
        "REPRODUCIBILITY.md",
        "repro/README.md",
        "repro/setup/verify-environment.sh",
        "claims.json",
        "technical_artifact_index.json",
    ];
    for relative in required_root_files {
        if !root.join(relative).is_file() {
            audit
                .findings
                .push(format!("missing required reproducibility file: {relative}"));
        }
    }
    if audit.findings.is_empty() {
        let counts = audit
            .status_counts
            .iter()
            .map(|(status, count)| format!("{status}={count}"))
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "REPRO_AUDIT PASS: {} experiments; {}; IMPORTANT_RESULT_WITHOUT_REPRO_DISPOSITION=0",
            audit.experiment_count, counts
        );
    } else {
        eprintln!(
            "REPRO_AUDIT FAIL: {} experiment(s), {} finding(s)",
            audit.experiment_count,
            audit.findings.len()
        );
        for finding in audit.findings {
            eprintln!("- {finding}");
        }
        std::process::exit(1);
    }
}

fn audit_technical_index(root: &Path, value: &Value, claims: &Value, audit: &mut Audit) {
    if value.get("schema").and_then(Value::as_str) != Some("remora.technical_artifact_index.v1") {
        audit
            .findings
            .push("technical_artifact_index.json has the wrong schema".into());
    }
    let Some(records) = value.get("records").and_then(Value::as_array) else {
        audit
            .findings
            .push("technical_artifact_index.json records must be an array".into());
        return;
    };
    if records.is_empty() {
        audit
            .findings
            .push("technical_artifact_index.json records must not be empty".into());
    }
    let mut record_ids = BTreeSet::new();
    let mut indexed_claim_ids = BTreeSet::new();
    for (index, record) in records.iter().enumerate() {
        for field in TECHNICAL_INDEX_FIELDS {
            if record.get(*field).is_none() {
                audit.findings.push(format!(
                    "technical_artifact_index.json records[{index}] missing {field}"
                ));
            }
        }
        let Some(artifact_id) = record.get("artifact_id").and_then(Value::as_str) else {
            continue;
        };
        if !record_ids.insert(artifact_id.to_string()) {
            audit.findings.push(format!(
                "technical artifact id is duplicated: {artifact_id}"
            ));
        }
        let status = record.get("status").and_then(Value::as_str).unwrap_or("");
        if !STATUSES.contains(&status) {
            audit.findings.push(format!(
                "technical artifact {artifact_id}: unknown status {status:?}"
            ));
        }
        let Some(claim_ids) = record.get("claim_ids").and_then(Value::as_array) else {
            continue;
        };
        for claim_id in claim_ids {
            let Some(claim_id) = claim_id.as_str() else {
                audit.findings.push(format!(
                    "technical artifact {artifact_id}: claim_ids must contain strings"
                ));
                continue;
            };
            indexed_claim_ids.insert(claim_id.to_string());
        }
        for path_field in ["repro_path", "receipt_path"] {
            if let Some(path) = record.get(path_field).and_then(Value::as_str) {
                if path.trim().is_empty() || !root.join(path).exists() {
                    audit.findings.push(format!(
                        "technical artifact {artifact_id}: {path_field} does not resolve: {path}"
                    ));
                }
            }
        }
    }

    let Some(claim_list) = claims.get("claims").and_then(Value::as_array) else {
        audit
            .findings
            .push("claims.json claims must be an array".into());
        return;
    };
    let known_claim_ids: BTreeSet<String> = claim_list
        .iter()
        .filter_map(|claim| claim.get("id").and_then(Value::as_str))
        .map(str::to_string)
        .collect();
    for claim_id in &indexed_claim_ids {
        if !known_claim_ids.contains(claim_id) {
            audit.findings.push(format!(
                "technical index references claim not present in claims.json: {claim_id}"
            ));
        }
    }

    let Some(coverage) = value.get("claim_coverage").and_then(Value::as_array) else {
        audit
            .findings
            .push("technical_artifact_index.json claim_coverage must be an array".into());
        return;
    };
    let mut covered_claim_ids = BTreeSet::new();
    for (index, entry) in coverage.iter().enumerate() {
        let Some(claim_id) = entry.get("claim_id").and_then(Value::as_str) else {
            audit.findings.push(format!(
                "technical claim_coverage[{index}] missing claim_id"
            ));
            continue;
        };
        if !covered_claim_ids.insert(claim_id.to_string()) {
            audit.findings.push(format!(
                "technical claim is covered more than once: {claim_id}"
            ));
        }
        if !known_claim_ids.contains(claim_id) {
            audit.findings.push(format!(
                "technical claim_coverage references unknown claim: {claim_id}"
            ));
        }
        let Some(artifact_ids) = entry.get("artifact_ids").and_then(Value::as_array) else {
            audit.findings.push(format!(
                "technical claim_coverage[{index}] has no artifact_ids: {claim_id}"
            ));
            continue;
        };
        if artifact_ids.is_empty() {
            audit.findings.push(format!(
                "technical claim_coverage[{index}] has empty artifact_ids: {claim_id}"
            ));
        }
        for artifact_id in artifact_ids.iter().filter_map(Value::as_str) {
            if !record_ids.contains(artifact_id) {
                audit.findings.push(format!(
                    "technical claim {claim_id} points to missing artifact: {artifact_id}"
                ));
            }
        }
        let status = entry.get("status").and_then(Value::as_str).unwrap_or("");
        if !STATUSES.contains(&status) {
            audit.findings.push(format!(
                "technical claim {claim_id}: unknown status {status:?}"
            ));
        }
        for path_field in ["repro_path", "receipt_path"] {
            if let Some(path) = entry.get(path_field).and_then(Value::as_str) {
                if path.trim().is_empty() || !root.join(path).exists() {
                    audit.findings.push(format!(
                        "technical claim {claim_id}: {path_field} does not resolve: {path}"
                    ));
                }
            }
        }
    }
    for claim_id in known_claim_ids {
        if !covered_claim_ids.contains(&claim_id) {
            audit.findings.push(format!(
                "claim in claims.json has no technical artifact coverage: {claim_id}"
            ));
        }
    }
}

fn read_json(path: &Path, audit: &mut Audit) -> Option<Value> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => {
            audit.findings.push(format!("{}: {error}", path.display()));
            return None;
        }
    };
    match serde_json::from_str(&text) {
        Ok(value) => Some(value),
        Err(error) => {
            audit
                .findings
                .push(format!("{}: invalid JSON: {error}", path.display()));
            None
        }
    }
}

fn audit_root_manifest(root: &Path, value: &Value, audit: &mut Audit) {
    if value.get("schema").and_then(Value::as_str) != Some("remora.reproducibility.v1") {
        audit
            .findings
            .push("repro_manifest.json has the wrong schema".into());
    }
    if value.get("important_results_without_repro_disposition") != Some(&Value::from(0)) {
        audit
            .findings
            .push("important_results_without_repro_disposition must be 0".into());
    }
    let Some(experiments) = value.get("experiments").and_then(Value::as_array) else {
        audit
            .findings
            .push("repro_manifest.json experiments must be an array".into());
        return;
    };
    for experiment in experiments {
        audit.experiment_count += 1;
        let Some(relative) = experiment.get("path").and_then(Value::as_str) else {
            audit
                .findings
                .push("root experiment entry is missing path".into());
            continue;
        };
        let status = experiment
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("");
        if !STATUSES.contains(&status) {
            audit
                .findings
                .push(format!("{relative}: unknown status {status:?}"));
        }
        *audit.status_counts.entry(status.to_string()).or_default() += 1;
        let lane = root.join(relative);
        if !lane.is_dir() {
            audit
                .findings
                .push(format!("{relative}: experiment directory missing"));
            continue;
        }
        audit_lane(root, &lane, relative, status, audit);
    }
}

fn audit_lane(root: &Path, lane: &Path, relative: &str, declared_status: &str, audit: &mut Audit) {
    for file in ["README.md", "manifest.json", "expected.json", "run.sh"] {
        if !lane.join(file).is_file() {
            audit.findings.push(format!("{relative}: missing {file}"));
        }
    }
    let manifest = match read_json(&lane.join("manifest.json"), audit) {
        Some(value) => value,
        None => return,
    };
    for field in CLAIM_FIELDS {
        if manifest.get(*field).is_none() {
            audit
                .findings
                .push(format!("{relative}/manifest.json: missing {field}"));
        }
    }
    let actual_status = manifest.get("status").and_then(Value::as_str).unwrap_or("");
    if actual_status != declared_status {
        audit.findings.push(format!(
            "{relative}: root status {declared_status} != manifest status {actual_status}"
        ));
    }
    if !STATUSES.contains(&actual_status) {
        audit.findings.push(format!(
            "{relative}: invalid manifest status {actual_status:?}"
        ));
    }
    let runnable = matches!(
        actual_status,
        "FULLY_REPRODUCIBLE"
            | "REPRODUCIBLE_WITH_PUBLIC_MODEL_DOWNLOAD"
            | "REPRODUCIBLE_ON_REFERENCE_HARDWARE"
            | "REPRODUCIBLE_WITH_HARDWARE_RETUNING"
            | "HISTORICAL_RECONSTRUCTION_AVAILABLE"
            | "FALSIFIED_REPRODUCIBLE"
    );
    if runnable {
        let command = manifest
            .get("command")
            .and_then(Value::as_str)
            .unwrap_or("");
        if command.trim().is_empty() {
            audit
                .findings
                .push(format!("{relative}: runnable manifest has empty command"));
        }
        let receipt = manifest
            .get("receipt_path")
            .and_then(Value::as_str)
            .unwrap_or("");
        if receipt.trim().is_empty() || !root.join(receipt).is_file() {
            audit.findings.push(format!(
                "{relative}: runnable manifest receipt_path is missing"
            ));
        }
    }
    if manifest
        .get("source_paths")
        .and_then(Value::as_array)
        .is_none()
    {
        audit
            .findings
            .push(format!("{relative}: source_paths must be an array"));
    }
    if manifest
        .get("model_hashes")
        .and_then(Value::as_object)
        .is_none()
    {
        audit
            .findings
            .push(format!("{relative}: model_hashes must be an object"));
    }
}
