//! Rust model compiler for HAR.
//!
//! This crate owns the offline decisions.  The runtime receives a
//! `har-model-package` manifest and never reinterprets model policy per token.
//! The GGUF reader is intentionally narrow but lossless for the tensor
//! directory and provenance needed by the first Qwen subset.

use har_model_package::{
    sha256_bytes, sha256_file, AllocationRecord, CalibrationEvidence, HardwarePhenotype,
    MemoryTier, PackageManifest, PackageWriter, SourceModel, StorageLocation, TensorRecord,
    TensorRole,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

// Keep the public error name local while exposing the package error through
// Result. This alias is useful to downstream tools that do not need a second
// error hierarchy.
pub type CompilerResult<T> = std::result::Result<T, CompilerError>;

#[derive(Debug)]
pub enum CompilerError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Package(har_model_package::PackageError),
    Invalid(String),
}

impl std::fmt::Display for CompilerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Json(e) => write!(f, "JSON error: {e}"),
            Self::Package(e) => write!(f, "package error: {e}"),
            Self::Invalid(e) => f.write_str(e),
        }
    }
}
impl std::error::Error for CompilerError {}
impl From<std::io::Error> for CompilerError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<serde_json::Error> for CompilerError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}
impl From<har_model_package::PackageError> for CompilerError {
    fn from(e: har_model_package::PackageError) -> Self {
        Self::Package(e)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ModelTensorDescriptor {
    #[serde(default)]
    pub ordinal: u64,
    pub name: String,
    #[serde(default)]
    pub role: String,
    #[serde(default = "minus_one")]
    pub layer: i32,
    #[serde(default, alias = "shape", alias = "dims")]
    pub dimensions: Vec<u64>,
    #[serde(default)]
    pub ggml_type: u32,
    #[serde(default, alias = "quant_format", alias = "format")]
    pub quantization: String,
    #[serde(default)]
    pub element_count: u64,
    #[serde(default, alias = "bytes", alias = "n_bytes")]
    pub payload_bytes: u64,
    #[serde(default)]
    pub raw_span_bytes: u64,
    #[serde(default, alias = "source_offset", alias = "offset")]
    pub file_offset: u64,
    #[serde(default, alias = "alignment")]
    pub alignment_bytes: u64,
    #[serde(default)]
    pub hotness: f64,
    #[serde(default = "true_value")]
    pub is_weight: bool,
    #[serde(default)]
    pub is_mtp: bool,
    #[serde(default)]
    pub expert_id: Option<u32>,
    #[serde(default)]
    pub projection: Option<String>,
}

fn minus_one() -> i32 {
    -1
}
fn true_value() -> bool {
    true
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ModelPhenotype {
    #[serde(default, alias = "source_path")]
    pub path: String,
    #[serde(default, alias = "source_sha256", alias = "model_sha256")]
    pub sha256: String,
    #[serde(default)]
    pub file_bytes: u64,
    #[serde(default)]
    pub gguf_version: u32,
    #[serde(default)]
    pub tensor_count: u64,
    #[serde(default)]
    pub metadata_count: u64,
    #[serde(default)]
    pub data_offset: u64,
    #[serde(default)]
    pub tensor_payload_bytes: u64,
    #[serde(default)]
    pub tensor_padding_bytes: u64,
    #[serde(default)]
    pub architecture: String,
    #[serde(default, alias = "model_id", alias = "model_identity")]
    pub model_name: String,
    #[serde(default)]
    pub block_count: u32,
    #[serde(default)]
    pub embedding_length: u32,
    #[serde(default)]
    pub attention_heads: u32,
    #[serde(default)]
    pub kv_heads: u32,
    #[serde(default)]
    pub key_length: u32,
    #[serde(default)]
    pub value_length: u32,
    #[serde(default)]
    pub nextn_predict_layers: u32,
    #[serde(default)]
    pub expert_count: u32,
    #[serde(default)]
    pub expert_used_count: u32,
    #[serde(default)]
    pub kv_geometry: String,
    #[serde(default)]
    pub quantization_bytes: BTreeMap<String, u64>,
    #[serde(default)]
    pub quantization_tensor_counts: BTreeMap<String, u64>,
    #[serde(default)]
    pub tensors: Vec<ModelTensorDescriptor>,
}

impl ModelPhenotype {
    pub fn from_json(path: impl AsRef<Path>) -> CompilerResult<Self> {
        let bytes = fs::read(path)?;
        Ok(serde_json::from_slice(&bytes)?)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorRoleInventoryRow {
    pub tensor_id: String,
    pub name: String,
    pub layer: Option<u32>,
    pub expert_id: Option<u32>,
    pub projection: Option<String>,
    pub role: TensorRole,
    pub tensor_class: String,
    pub rationale: String,
}

fn lower_name(name: &str) -> String {
    name.to_ascii_lowercase().replace('-', "_")
}

fn layer_from_name(name: &str) -> Option<u32> {
    for marker in ["blk.", "block.", "layer.", "layers.", "h."] {
        if let Some(start) = name.find(marker) {
            let digits = name[start + marker.len()..]
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>();
            if !digits.is_empty() {
                return digits.parse().ok();
            }
        }
    }
    None
}

fn expert_from_name(name: &str) -> Option<u32> {
    for marker in ["experts.", "expert.", "exps.", "expert_", "experts_"] {
        if let Some(start) = name.find(marker) {
            let digits = name[start + marker.len()..]
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>();
            if !digits.is_empty() {
                return digits.parse().ok();
            }
        }
    }
    None
}

fn projection_from_name(name: &str) -> Option<String> {
    if name.contains("qkv") {
        return Some("qkv".to_owned());
    }
    let checks: [(&str, &[&str]); 7] = [
        ("q", &["attn_q", "attention_q", "q_proj", ".wq", "_wq"]),
        ("k", &["attn_k", "attention_k", "k_proj", ".wk", "_wk"]),
        ("v", &["attn_v", "attention_v", "v_proj", ".wv", "_wv"]),
        (
            "o",
            &[
                "attn_output",
                "attn_o",
                "attention_o",
                "o_proj",
                ".wo",
                "_wo",
            ],
        ),
        ("gate", &["ffn_gate", "gate_proj", "mlp.gate", "_gate"]),
        ("up", &["ffn_up", "up_proj", "mlp.up", "_up"]),
        ("down", &["ffn_down", "down_proj", "mlp.down", "_down"]),
    ];
    for (projection, needles) in checks {
        if needles.iter().any(|needle| name.contains(needle)) {
            return Some(projection.to_owned());
        }
    }
    None
}

pub fn classify_tensor(descriptor: &ModelTensorDescriptor) -> TensorRoleInventoryRow {
    let name = lower_name(&descriptor.name);
    let layer = if descriptor.layer >= 0 {
        Some(descriptor.layer as u32)
    } else {
        layer_from_name(&name)
    };
    let expert_id = descriptor.expert_id.or_else(|| expert_from_name(&name));
    let projection = descriptor
        .projection
        .clone()
        .or_else(|| projection_from_name(&name));
    let attention_marker = [
        "attn",
        "attention",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "wq",
        "wk",
        "wv",
        "wo",
    ]
    .iter()
    .any(|needle| name.contains(needle));
    let expert_marker = expert_id.is_some() || name.contains("expert") || name.contains("exps");
    let role_hint = descriptor.role.to_ascii_uppercase();

    let (role, tensor_class, rationale) = if descriptor.is_mtp
        || name.contains("mtp")
        || name.contains("nextn")
        || name.contains("medusa")
    {
        (TensorRole::Mtp, "mtp", "MTP/next-token marker")
    } else if name.contains("token_embd")
        || name.contains("embed_tokens")
        || name.ends_with("embedding.weight")
    {
        (TensorRole::Embedding, "embedding", "token embedding marker")
    } else if name.contains("router")
        || name.contains("gate_inp")
        || name.contains("routing")
        || name.contains("e_score")
    {
        (TensorRole::Router, "router", "router/gate-in marker")
    } else if name.contains("shared_expert") || name.contains("ffn_shared") {
        (
            TensorRole::SharedExpert,
            "shared_expert",
            "shared expert marker",
        )
    } else if attention_marker && projection.as_deref() == Some("qkv") {
        (
            TensorRole::MetadataOther,
            "attention_fused_qkv",
            "fused QKV tensor retained as one source payload",
        )
    } else if attention_marker && matches!(projection.as_deref(), Some("q" | "k" | "v" | "o")) {
        let role = match projection.as_deref().unwrap() {
            "q" => TensorRole::AttentionQ,
            "k" => TensorRole::AttentionK,
            "v" => TensorRole::AttentionV,
            _ => TensorRole::AttentionO,
        };
        (role, "attention", "attention projection marker")
    } else if name.contains("rms_norm")
        || name.contains("layer_norm")
        || name.contains("norm.weight")
        || name.ends_with("_norm")
    {
        (
            TensorRole::Normalization,
            "normalization",
            "normalization marker",
        )
    } else if (name.contains("lm_head")
        || name.ends_with("output.weight")
        || name.contains("output_weight"))
        && !attention_marker
    {
        (TensorRole::Output, "output", "output head marker")
    } else if expert_marker && matches!(projection.as_deref(), Some("gate" | "up" | "down")) {
        let role = match projection.as_deref().unwrap() {
            "gate" => TensorRole::RoutedExpertGate,
            "up" => TensorRole::RoutedExpertUp,
            _ => TensorRole::RoutedExpertDown,
        };
        (role, "routed_expert", "expert marker plus FFN projection")
    } else if matches!(projection.as_deref(), Some("gate" | "up" | "down"))
        && (name.contains("ffn") || name.contains("mlp") || name.contains("feed_forward"))
    {
        let role = match projection.as_deref().unwrap() {
            "gate" => TensorRole::DenseFfnGate,
            "up" => TensorRole::DenseFfnUp,
            _ => TensorRole::DenseFfnDown,
        };
        (role, "dense_ffn", "dense FFN projection marker")
    } else if role_hint == "NORMALIZATION" {
        (
            TensorRole::Normalization,
            "normalization",
            "compiler generic role hint",
        )
    } else {
        (
            TensorRole::MetadataOther,
            "metadata_other",
            "no compiler compute-role marker",
        )
    };
    TensorRoleInventoryRow {
        tensor_id: if descriptor.name.is_empty() {
            format!("tensor-{}", descriptor.ordinal)
        } else {
            descriptor.name.clone()
        },
        name: descriptor.name.clone(),
        layer,
        expert_id,
        projection,
        role,
        tensor_class: tensor_class.to_owned(),
        rationale: rationale.to_owned(),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct HardwarePolicy {
    pub hardware: HardwarePhenotype,
    pub total_model_bytes_budget: Option<u64>,
    pub vram_bytes_budget: Option<u64>,
    pub ram_bytes_budget: Option<u64>,
    pub nvme_bytes_budget: Option<u64>,
    pub candidate_formats: Vec<String>,
    pub protect_roles: Vec<TensorRole>,
    pub active_experts_per_token: u32,
    pub routing_top_k_floor: Option<f64>,
    pub mtp_acceptance_floor: Option<f64>,
}

impl HardwarePolicy {
    pub fn rdna4_default() -> Self {
        let formats = ["Q3_K_S", "Q4_K_S", "Q5_K_M", "Q6_K", "Q8_K_XL"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect::<Vec<_>>();
        let kernels = formats
            .iter()
            .map(|format| kernel_for(format, "vulkan"))
            .chain([
                "har.vulkan.router.topk".to_owned(),
                "har.vulkan.mtp.verify".to_owned(),
                "har.vulkan.residual.add".to_owned(),
            ])
            .collect::<Vec<_>>();
        Self {
            hardware: HardwarePhenotype {
                hardware_id: "rdna4-vulkan".into(),
                backend: "vulkan".into(),
                gpu_arch: "RDNA4".into(),
                supported_formats: formats.clone(),
                kernel_paths: kernels,
                tensor_alignment: 256,
                sidecar_alignment: 4096,
                persistent_vulkan_slots: 4,
                ..Default::default()
            },
            candidate_formats: formats,
            protect_roles: vec![TensorRole::Router, TensorRole::Output, TensorRole::Mtp],
            active_experts_per_token: 1,
            ..Default::default()
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct FormatSpec {
    name: &'static str,
    bits_per_weight: f64,
    block_elements: u64,
    block_bytes: u64,
}

fn format_spec(format: &str) -> Option<FormatSpec> {
    match format.to_ascii_uppercase().as_str() {
        "Q3_K" | "Q3_K_S" => Some(FormatSpec {
            name: "Q3_K_S",
            bits_per_weight: 3.4375,
            block_elements: 256,
            block_bytes: 110,
        }),
        "Q4_K" | "Q4_K_S" => Some(FormatSpec {
            name: "Q4_K_S",
            bits_per_weight: 4.5,
            block_elements: 256,
            block_bytes: 144,
        }),
        "Q4_K_M" => Some(FormatSpec {
            name: "Q4_K_M",
            bits_per_weight: 4.75,
            block_elements: 256,
            block_bytes: 152,
        }),
        "Q5_K" | "Q5_K_M" => Some(FormatSpec {
            name: "Q5_K_M",
            bits_per_weight: 5.75,
            block_elements: 256,
            block_bytes: 176,
        }),
        "Q6_K" => Some(FormatSpec {
            name: "Q6_K",
            bits_per_weight: 6.5625,
            block_elements: 256,
            block_bytes: 210,
        }),
        "Q8_K" | "Q8_K_XL" | "UD-Q8_K_XL" | "UD_Q8_K_XL" => Some(FormatSpec {
            name: "Q8_K_XL",
            bits_per_weight: 9.125,
            block_elements: 256,
            block_bytes: 292,
        }),
        _ => None,
    }
}

fn quantized_bytes(format: &str, elements: u64) -> Option<u64> {
    let spec = format_spec(format)?;
    Some(elements.div_ceil(spec.block_elements) * spec.block_bytes)
}

pub fn kernel_for(format: &str, backend: &str) -> String {
    format!(
        "har.{}.gemm.{}",
        backend.to_ascii_lowercase(),
        format.to_ascii_lowercase().replace('-', "_")
    )
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct CalibrationRow {
    pub model_identity: String,
    pub tensor_group_id: String,
    pub candidate_format: String,
    #[serde(default)]
    pub layer: Option<u32>,
    #[serde(default)]
    pub projection: Option<String>,
    #[serde(default)]
    pub tensor_class: String,
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
    #[serde(default)]
    pub quality_loss: Option<f64>,
    #[serde(default)]
    pub behavioral_evidence: bool,
    #[serde(default)]
    pub reference_output_sha256: Option<String>,
    #[serde(default)]
    pub candidate_output_sha256: Option<String>,
}

impl CalibrationRow {
    pub fn measured_loss(&self) -> Option<f64> {
        if !self.behavioral_evidence {
            return None;
        }
        if let Some(loss) = self.quality_loss {
            return Some(loss);
        }
        let mut values = Vec::new();
        if let Some(value) = self.metrics.get("logit_kl") {
            values.push(value.max(0.0));
        }
        for key in [
            "top_token_agreement",
            "routing_topk_agreement",
            "mtp_acceptance",
            "rare_fact_exact",
            "code_math_score",
            "long_context_retrieval",
        ] {
            if let Some(value) = self.metrics.get(key) {
                values.push((1.0 - value).max(0.0));
            }
        }
        if values.is_empty() {
            None
        } else {
            Some(values.iter().sum::<f64>() / values.len() as f64)
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct CalibrationEvidenceSet {
    pub rows: Vec<CalibrationRow>,
    pub source_paths: Vec<String>,
    pub root_sha256: Option<String>,
    pub status: String,
}

impl CalibrationEvidenceSet {
    pub fn empty() -> Self {
        Self {
            status: "blocked_no_capture".into(),
            ..Default::default()
        }
    }
    pub fn from_json(path: impl AsRef<Path>) -> CompilerResult<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)?;
        let value: Value = serde_json::from_slice(&bytes)?;
        let rows_value = if value.is_array() {
            value
        } else {
            value
                .get("results")
                .cloned()
                .unwrap_or(Value::Array(Vec::new()))
        };
        let rows: Vec<CalibrationRow> = serde_json::from_value(rows_value)?;
        Ok(Self {
            rows,
            source_paths: vec![path.display().to_string()],
            root_sha256: Some(sha256_bytes(&bytes)),
            status: "parsed".into(),
        })
    }

    pub fn from_jsonl(path: impl AsRef<Path>) -> CompilerResult<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)?;
        let mut rows = Vec::new();
        for line in String::from_utf8_lossy(&bytes)
            .lines()
            .filter(|line| !line.trim().is_empty())
        {
            rows.push(serde_json::from_str::<CalibrationRow>(line)?);
        }
        Ok(Self {
            rows,
            source_paths: vec![path.display().to_string()],
            root_sha256: Some(sha256_bytes(&bytes)),
            status: "parsed".into(),
        })
    }

    pub fn from_csv(path: impl AsRef<Path>) -> CompilerResult<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)?;
        let lines = String::from_utf8_lossy(&bytes)
            .lines()
            .map(csv_fields)
            .collect::<Vec<_>>();
        if lines.is_empty() {
            return Ok(Self::empty());
        }
        let headers = &lines[0];
        let index: HashMap<&str, usize> = headers
            .iter()
            .enumerate()
            .map(|(i, key)| (key.as_str(), i))
            .collect();
        let get = |row: &Vec<String>, key: &str| -> String {
            index
                .get(key)
                .and_then(|i| row.get(*i))
                .cloned()
                .unwrap_or_default()
        };
        let mut rows = Vec::new();
        for fields in lines.iter().skip(1) {
            if fields.len() < headers.len() {
                continue;
            }
            let mut row = CalibrationRow {
                model_identity: get(fields, "model_identity"),
                tensor_group_id: get(fields, "tensor_group_id"),
                candidate_format: get(fields, "candidate_format"),
                tensor_class: get(fields, "tensor_class"),
                quality_loss: parse_optional(&get(fields, "quality_loss")),
                behavioral_evidence: get(fields, "behavioral_evidence") == "True"
                    || get(fields, "behavioral_evidence") == "true",
                ..Default::default()
            };
            row.layer = parse_optional(&get(fields, "layer")).map(|x| x as u32);
            row.projection = match get(fields, "projection").as_str() {
                "" => None,
                value => Some(value.to_owned()),
            };
            for key in [
                "logit_kl",
                "top_token_agreement",
                "routing_topk_agreement",
                "mtp_acceptance",
                "mtp_acceptance_delta",
                "rare_fact_exact",
                "code_math_score",
                "long_context_retrieval",
                "activation_error",
                "tensor_mse",
            ] {
                if let Some(value) = parse_optional(&get(fields, key)) {
                    row.metrics.insert(key.to_owned(), value);
                }
            }
            rows.push(row);
        }
        Ok(Self {
            rows,
            source_paths: vec![path.display().to_string()],
            root_sha256: Some(sha256_bytes(&bytes)),
            status: "parsed".into(),
        })
    }

    pub fn has_behavioral_evidence(&self) -> bool {
        self.rows
            .iter()
            .any(|row| row.behavioral_evidence && row.measured_loss().is_some())
    }
    pub fn by_group_format(&self) -> HashMap<(String, String), CalibrationRow> {
        let mut result = HashMap::new();
        for row in &self.rows {
            result.insert(
                (
                    row.tensor_group_id.clone(),
                    row.candidate_format.to_ascii_uppercase(),
                ),
                row.clone(),
            );
        }
        result
    }
}

fn parse_optional(value: &str) -> Option<f64> {
    let value = value.trim();
    if value.is_empty() || value == "None" || value == "null" {
        None
    } else {
        value.parse().ok()
    }
}

fn csv_fields(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(character) = chars.next() {
        match character {
            '"' if quoted && chars.peek() == Some(&'"') => {
                current.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => {
                fields.push(current);
                current = String::new();
            }
            other => current.push(other),
        }
    }
    fields.push(current);
    fields
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalibrationPlan {
    pub schema: String,
    pub executor: String,
    pub reference_format: String,
    pub candidate_formats: Vec<String>,
    pub model_targets: Vec<String>,
    pub probe_families: Vec<String>,
    pub required_metrics: Vec<String>,
    pub bounded_max_prompts: u32,
    pub bounded_max_tokens_per_prompt: u32,
    pub reject_mse_only: bool,
    pub full_model_quality_claim: String,
}

impl Default for CalibrationPlan {
    fn default() -> Self {
        Self {
            schema: "har.calibration_capture.v1".into(),
            executor: "rust-calibration-tool".into(),
            reference_format: "Q8_K_XL".into(),
            candidate_formats: vec![
                "Q3_K_S".into(),
                "Q4_K_S".into(),
                "Q5_K_M".into(),
                "Q6_K".into(),
                "Q8_K_XL".into(),
            ],
            model_targets: vec![
                "dense-mtp-reference Q4_K_S".into(),
                "dense-mtp-reference UD-Q8_K_XL".into(),
                "hybrid-flash-reference UD-Q8_K_XL".into(),
            ],
            probe_families: vec![
                "exact_identifier".into(),
                "uuid".into(),
                "filename".into(),
                "numbers".into(),
                "exceptions_arbitrary_mappings".into(),
                "routing_sensitive".into(),
                "code_math".into(),
                "long_context_retrieval".into(),
            ],
            required_metrics: vec![
                "logit_kl".into(),
                "top_token_agreement".into(),
                "routing_topk_agreement".into(),
                "mtp_acceptance".into(),
                "code_math_score".into(),
                "rare_fact_exact".into(),
                "long_context_retrieval".into(),
            ],
            bounded_max_prompts: 24,
            bounded_max_tokens_per_prompt: 128,
            reject_mse_only: true,
            full_model_quality_claim:
                "blocked until model-validation pass full-model captures pass held-out gates".into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct AllocationChoice {
    pub tensor_id: String,
    pub selected_format: String,
    pub expected_bytes: u64,
    pub quality_loss: Option<f64>,
    pub behavioral_evidence: bool,
    pub protected: bool,
    pub routed_bytes_per_active_token: u64,
    pub required_kernel: String,
    pub sensitivity_justification: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct AllocationPlan {
    pub schema: String,
    pub feasible: bool,
    pub budget_bytes: u64,
    pub expected_total_bytes: u64,
    pub source_bytes: u64,
    pub routed_bytes_per_active_token: u64,
    pub selected: Vec<AllocationChoice>,
    pub required_kernels: Vec<String>,
    pub unresolved_risks: Vec<String>,
    pub calibration_status: String,
}

pub fn allocate_formats(
    tensors: &[TensorRecord],
    policy: &HardwarePolicy,
    evidence: &CalibrationEvidenceSet,
) -> AllocationPlan {
    let budget = policy.total_model_bytes_budget.unwrap_or(u64::MAX);
    let candidates = if policy.candidate_formats.is_empty() {
        vec![
            "Q4_K_S".into(),
            "Q5_K_M".into(),
            "Q6_K".into(),
            "Q8_K_XL".into(),
        ]
    } else {
        policy.candidate_formats.clone()
    };
    let evidence_map = evidence.by_group_format();
    let mut choices = Vec::new();
    let mut unresolved = Vec::new();
    for tensor in tensors {
        let protected = policy.protect_roles.iter().any(|role| role == &tensor.role)
            || tensor.role.is_protected_default();
        let floor = if protected { Some("Q8_K_XL") } else { None };
        let mut options = candidates
            .iter()
            .filter_map(|format| {
                let spec = format_spec(format)?;
                if let Some(floor) = floor {
                    if spec.bits_per_weight + f64::EPSILON < format_spec(floor)?.bits_per_weight {
                        return None;
                    }
                }
                let kernel = kernel_for(spec.name, &policy.hardware.backend);
                if !policy.hardware.supported_formats.is_empty()
                    && !policy
                        .hardware
                        .supported_formats
                        .iter()
                        .any(|item| item.eq_ignore_ascii_case(spec.name))
                {
                    return None;
                }
                if !policy.hardware.kernel_paths.is_empty()
                    && !policy
                        .hardware
                        .kernel_paths
                        .iter()
                        .any(|item| item == &kernel)
                {
                    return None;
                }
                let bytes = quantized_bytes(spec.name, tensor.element_count)?;
                let evidence =
                    evidence_map.get(&(tensor.tensor_id.clone(), format.to_ascii_uppercase()));
                let quality_loss = evidence.and_then(CalibrationRow::measured_loss);
                let behavioral = evidence
                    .map(|row| row.behavioral_evidence && row.measured_loss().is_some())
                    .unwrap_or(false);
                Some((spec, bytes, kernel, quality_loss, behavioral))
            })
            .collect::<Vec<_>>();
        options.sort_by(|left, right| {
            left.1
                .cmp(&right.1)
                .then_with(|| left.0.bits_per_weight.total_cmp(&right.0.bits_per_weight))
        });
        if options.is_empty() {
            unresolved.push(format!(
                "{}: no supported option satisfies protection/kernel constraints",
                tensor.tensor_id
            ));
            continue;
        }
        let (spec, bytes, kernel, quality_loss, behavioral) = options[0].clone();
        if !behavioral {
            unresolved.push(format!(
                "{}: selected {} without behavioral calibration",
                tensor.tensor_id, spec.name
            ));
        }
        choices.push(AllocationChoice {
            tensor_id: tensor.tensor_id.clone(),
            selected_format: spec.name.into(),
            expected_bytes: bytes,
            quality_loss,
            behavioral_evidence: behavioral,
            protected,
            routed_bytes_per_active_token: if tensor.role.is_routed_expert() {
                bytes
            } else {
                0
            },
            required_kernel: kernel,
            sensitivity_justification: if behavioral {
                "measured behavioral evidence".into()
            } else {
                "unmeasured format prior; no quality claim".into()
            },
        });
    }
    // Spend spare bytes on measured upgrades only. A prior cannot turn a
    // diagnostic result into a quality claim.
    loop {
        let total: u64 = choices.iter().map(|choice| choice.expected_bytes).sum();
        let remaining = budget.saturating_sub(total);
        let mut best: Option<(usize, AllocationChoice, f64)> = None;
        for (index, current) in choices.iter().enumerate() {
            let tensor = match tensors
                .iter()
                .find(|tensor| tensor.tensor_id == current.tensor_id)
            {
                Some(value) => value,
                None => continue,
            };
            let current_spec = match format_spec(&current.selected_format) {
                Some(value) => value,
                None => continue,
            };
            for candidate in &candidates {
                let spec = match format_spec(candidate) {
                    Some(value) => value,
                    None => continue,
                };
                if spec.bits_per_weight <= current_spec.bits_per_weight
                    || spec.name == current.selected_format
                {
                    continue;
                }
                let evidence = match evidence_map
                    .get(&(tensor.tensor_id.clone(), candidate.to_ascii_uppercase()))
                {
                    Some(value) if value.behavioral_evidence => value,
                    _ => continue,
                };
                let loss = match evidence.measured_loss() {
                    Some(value) => value,
                    None => continue,
                };
                let extra = match quantized_bytes(spec.name, tensor.element_count) {
                    Some(value) => value.saturating_sub(current.expected_bytes),
                    None => continue,
                };
                if extra == 0 || extra > remaining {
                    continue;
                }
                let improvement = current.quality_loss.unwrap_or(1.0) - loss;
                if improvement <= 0.0 {
                    continue;
                }
                let score = improvement / extra as f64;
                let candidate_choice = AllocationChoice {
                    tensor_id: tensor.tensor_id.clone(),
                    selected_format: spec.name.into(),
                    expected_bytes: current.expected_bytes + extra,
                    quality_loss: Some(loss),
                    behavioral_evidence: true,
                    protected: current.protected,
                    routed_bytes_per_active_token: if tensor.role.is_routed_expert() {
                        current.expected_bytes + extra
                    } else {
                        0
                    },
                    required_kernel: kernel_for(spec.name, &policy.hardware.backend),
                    sensitivity_justification: "measured upgrade by loss-per-byte".into(),
                };
                if best
                    .as_ref()
                    .map(|(_, _, best_score)| score > *best_score)
                    .unwrap_or(true)
                {
                    best = Some((index, candidate_choice, score));
                }
            }
        }
        match best {
            Some((index, choice, _)) => choices[index] = choice,
            None => break,
        }
    }
    let expected_total_bytes = choices
        .iter()
        .map(|choice| choice.expected_bytes)
        .sum::<u64>();
    let mut by_layer: BTreeMap<Option<u32>, BTreeMap<Option<u32>, u64>> = BTreeMap::new();
    for choice in &choices {
        if choice.routed_bytes_per_active_token == 0 {
            continue;
        }
        if let Some(tensor) = tensors
            .iter()
            .find(|tensor| tensor.tensor_id == choice.tensor_id)
        {
            let layer = tensor.layer;
            let expert = tensor.expert_id;
            let layer_map = by_layer.entry(layer).or_default();
            *layer_map.entry(expert).or_default() += choice.routed_bytes_per_active_token;
        }
    }
    let routed_bytes_per_active_token = by_layer
        .values()
        .map(|experts| {
            experts.values().copied().max().unwrap_or(0)
                * policy.active_experts_per_token.max(1) as u64
        })
        .sum();
    let required_kernels = choices
        .iter()
        .map(|choice| choice.required_kernel.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    if expected_total_bytes > budget {
        unresolved.push(format!(
            "minimum/selected allocation {expected_total_bytes} exceeds budget {budget}"
        ));
    }
    AllocationPlan {
        schema: "har.heterogeneous_plan.v0".into(),
        feasible: unresolved
            .iter()
            .all(|risk| !risk.contains("exceeds budget"))
            && choices.len() == tensors.len(),
        budget_bytes: budget,
        expected_total_bytes,
        source_bytes: tensors.iter().map(|tensor| tensor.source_bytes).sum(),
        routed_bytes_per_active_token,
        selected: choices,
        required_kernels,
        unresolved_risks: unresolved,
        calibration_status: if evidence.has_behavioral_evidence() {
            "partial_behavioral".into()
        } else {
            "blocked_no_behavioral_capture".into()
        },
    }
}

#[derive(Clone, Debug)]
pub struct CompiledModel {
    pub phenotype: ModelPhenotype,
    pub manifest: PackageManifest,
    pub allocation: AllocationPlan,
    pub inventory: Vec<TensorRoleInventoryRow>,
}

pub fn compile_phenotype(
    phenotype: ModelPhenotype,
    mut policy: HardwarePolicy,
    evidence: CalibrationEvidenceSet,
    compiler_version: &str,
) -> CompilerResult<CompiledModel> {
    if policy.hardware.hardware_id.is_empty() {
        policy = HardwarePolicy::rdna4_default();
    }
    let source_root = if !phenotype.sha256.is_empty() {
        phenotype.sha256.clone()
    } else if !phenotype.path.is_empty() && Path::new(&phenotype.path).exists() {
        sha256_file(&phenotype.path)?
    } else {
        String::new()
    };
    let source = SourceModel {
        path: phenotype.path.clone(),
        root_sha256: source_root.clone(),
        file_bytes: phenotype.file_bytes,
        gguf_version: phenotype.gguf_version,
        architecture: phenotype.architecture.clone(),
        model_name: phenotype.model_name.clone(),
        tensor_count: phenotype.tensor_count.max(phenotype.tensors.len() as u64),
        metadata_count: phenotype.metadata_count,
        data_offset: phenotype.data_offset,
        tensor_payload_bytes: phenotype.tensor_payload_bytes,
    };
    let mut records = Vec::new();
    let mut inventory = Vec::new();
    let mut required_kernels = std::collections::BTreeSet::new();
    for descriptor in &phenotype.tensors {
        let row = classify_tensor(descriptor);
        inventory.push(row.clone());
        let layer = row.layer;
        let role = row.role.clone();
        let (storage, tier) = if role.is_routed_expert() {
            (StorageLocation::ExpertSidecar, MemoryTier::NvmeCold)
        } else if matches!(
            role,
            TensorRole::Router
                | TensorRole::Output
                | TensorRole::Mtp
                | TensorRole::AttentionQ
                | TensorRole::AttentionK
                | TensorRole::AttentionV
                | TensorRole::AttentionO
                | TensorRole::Normalization
        ) {
            (StorageLocation::Model, MemoryTier::VramResident)
        } else if role == TensorRole::MetadataOther {
            (StorageLocation::Metadata, MemoryTier::RamMapped)
        } else {
            (StorageLocation::Model, MemoryTier::RamMapped)
        };
        let base_kernel = kernel_for(&descriptor.quantization, &policy.hardware.backend);
        required_kernels.insert(base_kernel.clone());
        let mut metadata = BTreeMap::new();
        metadata.insert("role_hint".into(), descriptor.role.clone());
        metadata.insert("ggml_type".into(), descriptor.ggml_type.to_string());
        records.push(TensorRecord {
            tensor_id: row.tensor_id.clone(),
            name: descriptor.name.clone(),
            dimensions: descriptor.dimensions.clone(),
            element_count: descriptor
                .element_count
                .max(descriptor.dimensions.iter().product()),
            source_bytes: descriptor.payload_bytes,
            source_quant_format: descriptor.quantization.clone(),
            planned_bytes: None,
            planned_quant_format: None,
            layer,
            expert_id: row.expert_id,
            projection: row.projection.clone(),
            role: role.clone(),
            tensor_class: row.tensor_class.clone(),
            sensitivity_placeholder: None,
            supported_kernels: vec![base_kernel.clone()],
            required_kernels: vec![base_kernel],
            alignment: descriptor.alignment_bytes.max(
                if storage == StorageLocation::ExpertSidecar {
                    policy.hardware.sidecar_alignment.max(4096)
                } else {
                    policy.hardware.tensor_alignment.max(1)
                },
            ),
            source_offset: Some(descriptor.file_offset),
            source_file: if phenotype.path.is_empty() {
                None
            } else {
                Some(phenotype.path.clone())
            },
            storage_location: storage,
            planned_memory_tier: tier,
            payload_location_id: None,
            metadata,
        });
    }
    let allocation = allocate_formats(&records, &policy, &evidence);
    let mut allocation_map = HashMap::new();
    for choice in &allocation.selected {
        allocation_map.insert(choice.tensor_id.clone(), choice.clone());
    }
    for record in &mut records {
        if let Some(choice) = allocation_map.get(&record.tensor_id) {
            record.planned_bytes = Some(choice.expected_bytes);
            record.planned_quant_format = Some(choice.selected_format.clone());
            record.required_kernels = vec![choice.required_kernel.clone()];
            required_kernels.insert(choice.required_kernel.clone());
        }
    }
    let claims_allowed = evidence.has_behavioral_evidence()
        && allocation
            .unresolved_risks
            .iter()
            .all(|risk| !risk.contains("without behavioral"));
    let mut manifest = PackageManifest::new(source, policy.hardware.clone(), compiler_version);
    manifest.tensors = records;
    manifest.required_kernels = required_kernels.into_iter().collect();
    manifest.quality_evidence = CalibrationEvidence {
        schema: "har.calibration_capture.v1".into(),
        root_sha256: evidence.root_sha256.clone(),
        source_paths: evidence.source_paths.clone(),
        row_count: evidence.rows.len() as u64,
        behavioral_row_count: evidence
            .rows
            .iter()
            .filter(|row| row.behavioral_evidence)
            .count() as u64,
        status: evidence.status.clone(),
        metrics: CalibrationPlan::default().required_metrics,
        quality_claim_allowed: claims_allowed,
    };
    manifest.allocation = allocation
        .selected
        .iter()
        .map(|choice| AllocationRecord {
            tensor_id: choice.tensor_id.clone(),
            selected_format: choice.selected_format.clone(),
            expected_bytes: choice.expected_bytes,
            quality_loss: choice.quality_loss,
            behavioral_evidence: choice.behavioral_evidence,
            protected: choice.protected,
            routed_bytes_per_active_token: choice.routed_bytes_per_active_token,
            sensitivity_justification: choice.sensitivity_justification.clone(),
            unresolved_risks: allocation.unresolved_risks.clone(),
        })
        .collect();
    manifest.unresolved_risks = allocation.unresolved_risks.clone();
    manifest.claims.insert(
        "quality_claim".into(),
        if claims_allowed {
            "measured_scope_only"
        } else {
            "blocked"
        }
        .into(),
    );
    manifest
        .claims
        .insert("tensor_mse_alone".into(), "insufficient".into());
    manifest.claims.insert(
        "evidence_policy".into(),
        "preserve_exact_approximate_hybrid_and_negative_states".into(),
    );
    Ok(CompiledModel {
        phenotype,
        manifest,
        allocation,
        inventory,
    })
}

pub fn write_compiled_package(
    compiled: &CompiledModel,
    path: impl AsRef<Path>,
) -> CompilerResult<PackageManifest> {
    Ok(PackageWriter::write(path, &compiled.manifest, &[])?)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GgufInspection {
    pub phenotype: ModelPhenotype,
    pub inventory: Vec<TensorRoleInventoryRow>,
}

pub struct GgufInspector;

impl GgufInspector {
    pub fn inspect(path: impl AsRef<Path>) -> CompilerResult<GgufInspection> {
        let path = path.as_ref();
        let file_bytes = fs::metadata(path)?.len();
        let mut cursor = StreamReader::open(path)?;
        if cursor.bytes(4)? != b"GGUF" {
            return Err(CompilerError::Invalid("not a GGUF file".into()));
        }
        let version = cursor.u32()?;
        if !(1..=3).contains(&version) {
            return Err(CompilerError::Invalid(format!(
                "unsupported GGUF version {version}"
            )));
        }
        let tensor_count = cursor.u64()?;
        let metadata_count = cursor.u64()?;
        if tensor_count > 10_000_000 || metadata_count > 1_000_000 {
            return Err(CompilerError::Invalid("unreasonable GGUF counts".into()));
        }
        let mut metadata: HashMap<String, String> = HashMap::new();
        for _ in 0..metadata_count {
            let key = cursor.string()?;
            let type_id = cursor.u32()?;
            if let Some(value) = cursor.metadata_value(type_id)? {
                metadata.insert(key, value);
            }
        }
        let alignment = metadata
            .get("general.alignment")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(32);
        let mut raw = Vec::with_capacity(tensor_count as usize);
        for ordinal in 0..tensor_count {
            let name = cursor.string()?;
            let rank = cursor.u32()?;
            if rank > 16 {
                return Err(CompilerError::Invalid(format!(
                    "tensor {name} has rank {rank}"
                )));
            }
            let mut dimensions = Vec::with_capacity(rank as usize);
            for _ in 0..rank {
                dimensions.push(cursor.u64()?);
            }
            let ggml_type = cursor.u32()?;
            let offset = cursor.u64()?;
            raw.push((ordinal, name, dimensions, ggml_type, offset));
        }
        // GGUF data offsets are relative to the start of the tensor-data
        // section, which begins AFTER all tensor-info records (aligned to
        // `general.alignment`, default 32).  The base must be derived from the
        // logical consumed position, never from `BufReader::stream_position`
        // (read-ahead makes that nondeterministic) and never from the
        // metadata end (format-compatibility-v2: an earlier writer produced a data base 51,232 bytes too
        // early, shifting every packed payload).
        let data_offset = align(cursor.logical_position()?, alignment);
        let mut sorted = raw
            .iter()
            .enumerate()
            .map(|(index, item)| (item.4, index))
            .collect::<Vec<_>>();
        sorted.sort_unstable();
        let mut next_span = HashMap::new();
        for pair in sorted.windows(2) {
            next_span.insert(pair[0].1, pair[1].0.saturating_sub(pair[0].0));
        }
        let mut tensors = Vec::with_capacity(raw.len());
        let mut inventory = Vec::with_capacity(raw.len());
        let mut quantization_bytes = BTreeMap::new();
        let mut quantization_counts = BTreeMap::new();
        for (index, (ordinal, name, dimensions, ggml_type, offset)) in raw.into_iter().enumerate() {
            let elements = dimensions.iter().product::<u64>();
            let quantization = ggml_type_name(ggml_type).to_owned();
            let payload_bytes = tensor_payload_bytes(ggml_type, elements).unwrap_or_else(|| {
                next_span.get(&index).copied().unwrap_or_else(|| {
                    file_bytes.saturating_sub(data_offset.saturating_add(offset))
                })
            });
            let descriptor = ModelTensorDescriptor {
                ordinal,
                name,
                role: String::new(),
                layer: -1,
                dimensions,
                ggml_type,
                quantization: quantization.clone(),
                element_count: elements,
                payload_bytes,
                raw_span_bytes: next_span.get(&index).copied().unwrap_or(payload_bytes),
                file_offset: data_offset + offset,
                alignment_bytes: alignment,
                hotness: 0.0,
                is_weight: true,
                is_mtp: false,
                expert_id: None,
                projection: None,
            };
            let row = classify_tensor(&descriptor);
            *quantization_bytes.entry(quantization.clone()).or_default() += payload_bytes;
            *quantization_counts.entry(quantization).or_default() += 1;
            inventory.push(row);
            tensors.push(descriptor);
        }
        let sha256 = sha256_file(path)?;
        let architecture = metadata
            .get("general.architecture")
            .cloned()
            .unwrap_or_else(|| "unknown".into());
        let model_name = metadata
            .get("general.name")
            .or_else(|| metadata.get("general.basename"))
            .cloned()
            .unwrap_or_else(|| {
                path.file_stem()
                    .and_then(|value| value.to_str())
                    .unwrap_or("model")
                    .into()
            });
        let prefix = format!("{architecture}.");
        let get_u32 = |suffix: &str| {
            metadata
                .get(&(prefix.clone() + suffix))
                .or_else(|| metadata.get(suffix))
                .and_then(|value| value.parse().ok())
                .unwrap_or(0)
        };
        let payload_total = quantization_bytes.values().sum::<u64>();
        let phenotype = ModelPhenotype {
            path: path.display().to_string(),
            sha256,
            file_bytes,
            gguf_version: version,
            tensor_count,
            metadata_count,
            data_offset,
            tensor_payload_bytes: payload_total,
            tensor_padding_bytes: file_bytes.saturating_sub(data_offset + payload_total),
            architecture,
            model_name,
            block_count: get_u32("block_count"),
            embedding_length: get_u32("embedding_length"),
            attention_heads: get_u32("attention.head_count"),
            kv_heads: get_u32("attention.head_count_kv"),
            key_length: get_u32("attention.key_length"),
            value_length: get_u32("attention.value_length"),
            nextn_predict_layers: get_u32("nextn_predict_layers"),
            expert_count: get_u32("expert_count"),
            expert_used_count: get_u32("expert_used_count"),
            kv_geometry: String::new(),
            quantization_bytes,
            quantization_tensor_counts: quantization_counts,
            tensors,
        };
        Ok(GgufInspection {
            phenotype,
            inventory,
        })
    }
}

fn align(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        value
    } else {
        value.div_ceil(alignment) * alignment
    }
}

fn ggml_type_name(type_id: u32) -> &'static str {
    match type_id {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        30 => "BF16",
        34 => "TQ1_0",
        35 => "TQ2_0",
        39 => "MXFP4",
        40 => "NVFP4",
        41 => "Q1_0",
        42 => "Q2_0",
        _ => "UNKNOWN",
    }
}

fn tensor_payload_bytes(type_id: u32, elements: u64) -> Option<u64> {
    let (block, bytes) = match type_id {
        0 => (1, 4),
        1 => (1, 2),
        2 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        9 => (32, 36),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        15 => (256, 292),
        30 => (1, 2),
        _ => return None,
    };
    Some(elements.div_ceil(block) * bytes)
}

struct StreamReader {
    file: BufReader<File>,
    /// Logical consumed byte position.  `BufReader::stream_position` is
    /// influenced by read-ahead buffering and must never be used as the
    /// parsing cursor (format-compatibility-v2).
    pos: u64,
}
impl StreamReader {
    fn open(path: &Path) -> CompilerResult<Self> {
        Ok(Self {
            file: BufReader::with_capacity(1024 * 1024, File::open(path)?),
            pos: 0,
        })
    }
    fn take(&mut self, count: usize) -> CompilerResult<Vec<u8>> {
        let mut bytes = vec![0u8; count];
        self.file.read_exact(&mut bytes)?;
        self.pos = self.pos.saturating_add(count as u64);
        Ok(bytes)
    }
    fn bytes(&mut self, count: usize) -> CompilerResult<Vec<u8>> {
        self.take(count)
    }
    fn u8(&mut self) -> CompilerResult<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> CompilerResult<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> CompilerResult<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn logical_position(&mut self) -> CompilerResult<u64> {
        Ok(self.pos)
    }
    fn skip(&mut self, count: u64) -> CompilerResult<()> {
        self.file.seek(SeekFrom::Current(count as i64))?;
        self.pos = self.pos.saturating_add(count);
        Ok(())
    }
    fn discard(&mut self, mut count: u64) -> CompilerResult<()> {
        let mut scratch = [0u8; 1024 * 1024];
        while count > 0 {
            let wanted = count.min(scratch.len() as u64) as usize;
            self.file.read_exact(&mut scratch[..wanted])?;
            count -= wanted as u64;
            self.pos = self.pos.saturating_add(wanted as u64);
        }
        Ok(())
    }
    fn string(&mut self) -> CompilerResult<String> {
        let length = self.u64()?;
        if length > 1 << 30 {
            return Err(CompilerError::Invalid("unreasonable GGUF string".into()));
        }
        Ok(String::from_utf8_lossy(&self.take(length as usize)?).into_owned())
    }
    fn fixed_width(type_id: u32) -> Option<u64> {
        match type_id {
            0 | 1 | 7 => Some(1),
            2 | 3 => Some(2),
            4..=6 => Some(4),
            10..=12 => Some(8),
            _ => None,
        }
    }
    fn skip_array(&mut self, element_type: u32, count: u64) -> CompilerResult<()> {
        if let Some(width) = Self::fixed_width(element_type) {
            return self.skip(width.saturating_mul(count));
        }
        for _ in 0..count {
            self.skip_value(element_type)?;
        }
        Ok(())
    }
    fn skip_value(&mut self, type_id: u32) -> CompilerResult<()> {
        match type_id {
            0 | 1 | 7 => self.skip(1),
            2 | 3 => self.skip(2),
            4..=6 => self.skip(4),
            8 => {
                let length = self.u64()?;
                self.discard(length)
            }
            10..=12 => self.skip(8),
            9 => {
                let element_type = self.u32()?;
                let count = self.u64()?;
                if count > 10_000_000 {
                    return Err(CompilerError::Invalid(
                        "unreasonable GGUF metadata array".into(),
                    ));
                }
                self.skip_array(element_type, count)
            }
            _ => Err(CompilerError::Invalid(format!(
                "unsupported GGUF metadata type {type_id}"
            ))),
        }
    }
    fn metadata_value(&mut self, type_id: u32) -> CompilerResult<Option<String>> {
        match type_id {
            0 => Ok(Some(self.u8()?.to_string())),
            1 => Ok(Some((self.u8()? as i8).to_string())),
            2 => Ok(Some(
                (u16::from_le_bytes(self.take(2)?.try_into().unwrap())).to_string(),
            )),
            3 => Ok(Some(
                (i16::from_le_bytes(self.take(2)?.try_into().unwrap())).to_string(),
            )),
            4 => Ok(Some(self.u32()?.to_string())),
            5 => Ok(Some((self.u32()? as i32).to_string())),
            6 => Ok(Some(
                f32::from_le_bytes(self.take(4)?.try_into().unwrap()).to_string(),
            )),
            7 => Ok(Some((self.u8()? != 0).to_string())),
            8 => Ok(Some(self.string()?)),
            9 => {
                let element_type = self.u32()?;
                let count = self.u64()?;
                if count > 10_000_000 {
                    return Err(CompilerError::Invalid(
                        "unreasonable GGUF metadata array".into(),
                    ));
                }
                self.skip_array(element_type, count)?;
                Ok(None)
            }
            10 => Ok(Some(self.u64()?.to_string())),
            11 => Ok(Some((self.u64()? as i64).to_string())),
            12 => Ok(Some(
                f64::from_le_bytes(self.take(8)?.try_into().unwrap()).to_string(),
            )),
            _ => Err(CompilerError::Invalid(format!(
                "unsupported GGUF metadata type {type_id}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    /// Minimal deterministic GGUF v3 writer that mirrors the real-file
    /// structure that exposed format-compatibility-v2: a large metadata section (string
    /// arrays like tokenizer vocabularies) followed by tensor-info records
    /// and a 32-byte-aligned tensor-data section.
    fn write_synthetic_gguf(
        tensor_infos: &[(String, Vec<u64>, u32)],
        metadata_string_arrays: usize,
        metadata_string_count: usize,
    ) -> (PathBuf, u64, u64) {
        let path = std::env::temp_dir().join(format!(
            "har-format-compat-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut file = File::create(&path).unwrap();
        let write_str = |f: &mut File, s: &str| {
            f.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
            f.write_all(s.as_bytes()).unwrap();
        };
        let write_kv_str = |f: &mut File, key: &str, value: &str| {
            write_str(f, key);
            f.write_all(&8u32.to_le_bytes()).unwrap();
            write_str(f, value);
        };
        let write_kv_u32 = |f: &mut File, key: &str, value: u32| {
            write_str(f, key);
            f.write_all(&4u32.to_le_bytes()).unwrap();
            f.write_all(&value.to_le_bytes()).unwrap();
        };
        let write_kv_str_array = |f: &mut File, key: &str, count: usize| {
            write_str(f, key);
            f.write_all(&9u32.to_le_bytes()).unwrap();
            f.write_all(&8u32.to_le_bytes()).unwrap();
            f.write_all(&(count as u64).to_le_bytes()).unwrap();
            for i in 0..count {
                let value = format!("token-{i:08x}-with-enough-padding-to-fill-buffers");
                write_str(f, &value);
            }
        };
        file.write_all(b"GGUF").unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&(tensor_infos.len() as u64).to_le_bytes())
            .unwrap();
        let n_kv = 3 + metadata_string_arrays;
        file.write_all(&(n_kv as u64).to_le_bytes()).unwrap();
        write_kv_str(&mut file, "general.name", "synthetic-format-compat");
        write_kv_u32(&mut file, "test.architecture.block_count", 1);
        for i in 0..metadata_string_arrays {
            write_kv_str_array(
                &mut file,
                &format!("tokenizer.ggml.tokens.{i}"),
                metadata_string_count,
            );
        }
        write_kv_str(&mut file, "tokenizer.chat_template", "{synthetic}");
        let metadata_end = file.stream_position().unwrap();
        for (name, dims, ggml_type) in tensor_infos {
            write_str(&mut file, name);
            file.write_all(&(dims.len() as u32).to_le_bytes()).unwrap();
            for dim in dims {
                file.write_all(&dim.to_le_bytes()).unwrap();
            }
            file.write_all(&ggml_type.to_le_bytes()).unwrap();
            file.write_all(&0u64.to_le_bytes()).unwrap();
        }
        let infos_end = file.stream_position().unwrap();
        let data_start = align(infos_end, 32);
        let mut running = data_start;
        {
            let mut patch = File::options().read(true).write(true).open(&path).unwrap();
            let mut cursor = metadata_end;
            for (name, dims, ggml_type) in tensor_infos {
                cursor += 8 + name.len() as u64 + 4 + 8 * dims.len() as u64 + 4;
                patch.seek(SeekFrom::Start(cursor)).unwrap();
                patch
                    .write_all(&(running - data_start).to_le_bytes())
                    .unwrap();
                cursor += 8;
                let elements: u64 = dims.iter().product();
                let payload = tensor_payload_bytes(*ggml_type, elements).unwrap_or(elements * 2);
                running += payload;
            }
        }
        {
            let mut data = File::options().write(true).open(&path).unwrap();
            data.seek(SeekFrom::Start(data_start)).unwrap();
            let payload_total = (running - data_start) as usize;
            data.write_all(&vec![0xABu8; payload_total]).unwrap();
        }
        (path, data_start, infos_end)
    }

    #[test]
    fn inspector_data_offset_starts_after_tensor_infos_not_after_metadata() {
        // format-compatibility-v2 regression: the data base must be the aligned end of the
        // tensor-info section, not the metadata end (the metadata section
        // here is ~2.5 MiB, larger than the 1 MiB read buffer).
        let infos = vec![
            (
                "blk.0.attn_gate.weight".to_owned(),
                vec![5120u64, 6144],
                12u32,
            ),
            (
                "blk.0.ffn_gate.weight".to_owned(),
                vec![5120u64, 17408],
                12u32,
            ),
            (
                "blk.0.ffn_down.weight".to_owned(),
                vec![17408u64, 5120],
                13u32,
            ),
            ("output.weight".to_owned(), vec![5120u64, 972], 14u32),
        ];
        let (path, data_start, infos_end) = write_synthetic_gguf(&infos, 3, 20_000);
        let inspection = GgufInspector::inspect(&path).unwrap();
        assert_eq!(
            inspection.phenotype.data_offset, data_start,
            "data base must be the aligned tensor-info end"
        );
        assert!(
            inspection.phenotype.data_offset >= infos_end,
            "data base must not be the metadata end"
        );
        for (name, dims, ggml_type) in &infos {
            let descriptor = inspection
                .phenotype
                .tensors
                .iter()
                .find(|t| &t.name == name)
                .unwrap();
            let elements: u64 = dims.iter().product();
            let payload = tensor_payload_bytes(*ggml_type, elements).unwrap();
            assert_eq!(descriptor.payload_bytes, payload);
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn inspector_offsets_are_deterministic_and_independent_of_buffer_refills() {
        let infos = vec![
            (
                "blk.0.ffn_gate.weight".to_owned(),
                vec![5120u64, 17408],
                12u32,
            ),
            (
                "blk.0.ffn_up.weight".to_owned(),
                vec![5120u64, 17408],
                12u32,
            ),
        ];
        let (path, data_start, _) = write_synthetic_gguf(&infos, 4, 30_000);
        let first = GgufInspector::inspect(&path).unwrap();
        let second = GgufInspector::inspect(&path).unwrap();
        assert_eq!(first.phenotype.data_offset, second.phenotype.data_offset);
        assert_eq!(first.phenotype.data_offset, data_start);
        for (a, b) in first
            .phenotype
            .tensors
            .iter()
            .zip(second.phenotype.tensors.iter())
        {
            assert_eq!(
                a.file_offset, b.file_offset,
                "file offsets must be deterministic"
            );
            assert_eq!(a.payload_bytes, b.payload_bytes);
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn inspector_file_offsets_match_true_tensor_data_offsets() {
        // format-compatibility-v2 acceptance property: file_offset must equal the aligned
        // tensor-info end plus the tensor's relative offset (the caller file
        // shape: 866 tensor infos; the buggy base was 51,232 B early).
        let infos = vec![
            (
                "blk.0.ffn_gate.weight".to_owned(),
                vec![5120u64, 17408],
                12u32,
            ),
            (
                "blk.0.ffn_up.weight".to_owned(),
                vec![5120u64, 17408],
                12u32,
            ),
        ];
        let (path, data_start, _) = write_synthetic_gguf(&infos, 2, 12_000);
        let inspection = GgufInspector::inspect(&path).unwrap();
        let gate = inspection
            .phenotype
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_gate.weight")
            .unwrap();
        // First tensor's relative offset is 0 -> file_offset == data_start.
        assert_eq!(gate.file_offset, data_start);
        // Row geometry: 5120 elements/row = 20 Q4_K blocks of 144 B = 2,880 B.
        let row_bytes = (5120u64 / 256) * 144;
        assert_eq!(row_bytes, 2880);
        // rows 0..32 of the source tensor = the first 92,160 bytes of the
        // tensor payload; verify they live at the reported file_offset.
        let span = 32 * row_bytes;
        let mut source_span = vec![0u8; span as usize];
        {
            let mut f = File::open(&path).unwrap();
            f.seek(SeekFrom::Start(gate.file_offset)).unwrap();
            f.read_exact(&mut source_span).unwrap();
        }
        assert!(source_span.iter().all(|b| *b == 0xAB));
        assert_eq!(gate.payload_bytes, (5120u64 * 17408 / 256) * 144);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn role_classifier_recovers_attention_o_from_classifier_output_hint() {
        let tensor = ModelTensorDescriptor {
            name: "blk.0.attn_output.weight".into(),
            role: "OUTPUT_PROJECTION".into(),
            layer: 0,
            dimensions: vec![128, 128],
            payload_bytes: 9216,
            ..Default::default()
        };
        let row = classify_tensor(&tensor);
        assert_eq!(row.role, TensorRole::AttentionO);
        assert_eq!(row.projection.as_deref(), Some("o"));
    }

    #[test]
    fn allocation_uses_protected_q8_and_measured_ordinary_formats() {
        let policy = HardwarePolicy::rdna4_default();
        let tensors = vec![
            TensorRecord {
                tensor_id: "router".into(),
                role: TensorRole::Router,
                tensor_class: "router".into(),
                element_count: 256,
                ..Default::default()
            },
            TensorRecord {
                tensor_id: "ffn".into(),
                role: TensorRole::DenseFfnDown,
                tensor_class: "dense_ffn".into(),
                element_count: 256,
                ..Default::default()
            },
        ];
        let evidence = CalibrationEvidenceSet {
            rows: vec![
                CalibrationRow {
                    tensor_group_id: "ffn".into(),
                    candidate_format: "Q4_K_S".into(),
                    behavioral_evidence: true,
                    quality_loss: Some(0.2),
                    ..Default::default()
                },
                CalibrationRow {
                    tensor_group_id: "ffn".into(),
                    candidate_format: "Q6_K".into(),
                    behavioral_evidence: true,
                    quality_loss: Some(0.01),
                    ..Default::default()
                },
            ],
            status: "parsed".into(),
            ..Default::default()
        };
        let plan = allocate_formats(&tensors, &policy, &evidence);
        assert!(plan
            .selected
            .iter()
            .any(|choice| choice.tensor_id == "router" && choice.selected_format == "Q8_K_XL"));
        assert!(plan
            .selected
            .iter()
            .any(|choice| choice.tensor_id == "ffn" && choice.behavioral_evidence));
    }
}
