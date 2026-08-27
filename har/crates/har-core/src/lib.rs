//! HAR core contracts.
//!
//! This crate contains only typed, serializable control-plane vocabulary.  It
//! does not import a device backend, a model loader, or a token loop.
//! Runtime crates consume these values after a plan has been compiled.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

pub const CORE_INTERFACE: &str = "har.core.v1";
pub const IR_VERSION: &str = "har.ir.v1";
pub const PLAN_VERSION: &str = "har.plan.v1";
pub const MANIFEST_VERSION: &str = "har.runtime_manifest.v1";

/// Typed interface identity.  `schema()` is the canonical `har.<name>.v<N>`
/// string frozen in `HAR_CORE_INTERFACE_V0.json`.  V0 freezes the *current*
/// `v1` wire forms; the type is the single source of truth for that naming.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InterfaceVersion {
    pub name: &'static str,
    pub version: u32,
}
impl InterfaceVersion {
    pub const fn new(name: &'static str, version: u32) -> Self {
        Self { name, version }
    }
    pub fn schema(&self) -> String {
        format!("har.{}.v{}", self.name, self.version)
    }
}
impl fmt::Display for InterfaceVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.schema())
    }
}

/// V0 frozen interface identities.  These are the control-plane wire forms
/// frozen by `HAR_CORE_INTERFACE_V0.md`; changing them requires the V0 change
/// process recorded there.
pub mod v0 {
    use super::InterfaceVersion;
    pub const CORE: InterfaceVersion = InterfaceVersion::new("core", 1);
    pub const MODEL_PHENOTYPE: InterfaceVersion = InterfaceVersion::new("model_phenotype", 1);
    pub const TENSOR_DESCRIPTOR: InterfaceVersion = InterfaceVersion::new("tensor_descriptor", 1);
    pub const IR: InterfaceVersion = InterfaceVersion::new("ir", 1);
    pub const MEMORY: InterfaceVersion = InterfaceVersion::new("memory", 1);
    pub const PLAN: InterfaceVersion = InterfaceVersion::new("execution_plan", 1);
    pub const EVENTS: InterfaceVersion = InterfaceVersion::new("events", 1);
    pub const TELEMETRY: InterfaceVersion = InterfaceVersion::new("telemetry", 1);
    pub const EXECUTION: InterfaceVersion = InterfaceVersion::new("execution", 1);
    pub const CERTIFICATES: InterfaceVersion = InterfaceVersion::new("certificates", 1);
    pub const RUNTIME_MANIFEST: InterfaceVersion = InterfaceVersion::new("runtime_manifest", 1);
    pub const COMPILED_PROGRAM: InterfaceVersion = InterfaceVersion::new("compiled-program", 1);
}

/// Typed quantized-format identity shared by model descriptors, plans, and
/// kernel requirements.  Serialized with the exact GGUF type names, so the
/// wire form is byte-identical to the previous string form.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum QuantFormat {
    #[serde(rename = "F32")]
    F32,
    #[serde(rename = "F16")]
    F16,
    #[serde(rename = "BF16")]
    Bf16,
    #[serde(rename = "Q4_0")]
    Q4_0,
    #[serde(rename = "Q4_1")]
    Q4_1,
    #[serde(rename = "Q5_0")]
    Q5_0,
    #[serde(rename = "Q5_1")]
    Q5_1,
    #[serde(rename = "Q8_0")]
    Q8_0,
    #[serde(rename = "Q2_K")]
    Q2K,
    #[serde(rename = "Q3_K")]
    Q3K,
    #[serde(rename = "Q4_K")]
    Q4K,
    #[serde(rename = "Q5_K")]
    Q5K,
    #[serde(rename = "Q6_K")]
    Q6K,
    #[serde(rename = "Q8_K")]
    Q8K,
    #[serde(rename = "MXFP4")]
    Mxfp4,
    #[serde(rename = "UNKNOWN")]
    #[default]
    Unknown,
}
impl QuantFormat {
    pub fn from_ggml_type(type_id: u32) -> Self {
        match type_id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            30 => Self::Bf16,
            39 => Self::Mxfp4,
            _ => Self::Unknown,
        }
    }
    pub fn from_gguf_name(name: &str) -> Self {
        match name {
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "Q4_0" => Self::Q4_0,
            "Q4_1" => Self::Q4_1,
            "Q5_0" => Self::Q5_0,
            "Q5_1" => Self::Q5_1,
            "Q8_0" => Self::Q8_0,
            "Q2_K" => Self::Q2K,
            "Q3_K" => Self::Q3K,
            "Q4_K" => Self::Q4K,
            "Q5_K" => Self::Q5K,
            "Q6_K" => Self::Q6K,
            "Q8_K" => Self::Q8K,
            "MXFP4" => Self::Mxfp4,
            _ => Self::Unknown,
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::Mxfp4 => "MXFP4",
            Self::Unknown => "UNKNOWN",
        }
    }
    pub fn is_quantized(&self) -> bool {
        !matches!(self, Self::F32 | Self::F16 | Self::Bf16 | Self::Unknown)
    }
    /// Conservative kernel hint used only for planning; the backend remains
    /// the authority for actual kernel availability.
    pub fn kernel_hint(&self) -> KernelKind {
        match self {
            Self::Q4K | Self::Q4_0 | Self::Q4_1 => KernelKind::Q4KMatVec,
            Self::F32 | Self::F16 | Self::Bf16 => KernelKind::DenseMulMat,
            Self::Unknown => KernelKind::Unknown,
            _ => KernelKind::QuantizedMulMat,
        }
    }
}
impl fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
/// Typed tensor role identity.  Serialization is SCREAMING_SNAKE_CASE and
/// byte-identical to the previous string classification, so model phenotype
/// hashes are unchanged by the V0 freeze.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TensorRole {
    TokenEmbedding,
    OutputProjection,
    Attention,
    FeedForward,
    Router,
    Normalization,
    Mtp,
    Other,
}
impl TensorRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TokenEmbedding => "TOKEN_EMBEDDING",
            Self::OutputProjection => "OUTPUT_PROJECTION",
            Self::Attention => "ATTENTION",
            Self::FeedForward => "FEED_FORWARD",
            Self::Router => "ROUTER",
            Self::Normalization => "NORMALIZATION",
            Self::Mtp => "MTP",
            Self::Other => "OTHER",
        }
    }
    pub fn is_routed_expert(&self) -> bool {
        matches!(self, Self::FeedForward | Self::Router)
    }
    pub fn classify(name: &str) -> Self {
        let lower = name.to_ascii_lowercase();
        if lower.contains("mtp") || lower.contains("nextn") {
            Self::Mtp
        } else if lower.contains("router") || lower.contains("ffn_gate_inp") {
            Self::Router
        } else if lower.contains("ffn_") {
            Self::FeedForward
        } else if lower.contains("attn") || lower.contains("ssm") {
            Self::Attention
        } else if lower.contains("norm") {
            Self::Normalization
        } else if lower.contains("token_embd") {
            Self::TokenEmbedding
        } else if lower.contains("output") {
            Self::OutputProjection
        } else {
            Self::Other
        }
    }
}
impl fmt::Display for TensorRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Explicit model identity: display name plus source SHA-256.  The hash is
/// the authority; the name is for humans and manifests.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModelRoot {
    pub name: String,
    pub sha256: String,
}
impl ModelRoot {
    pub fn new(name: impl Into<String>, sha256: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            sha256: sha256.into(),
        }
    }
    pub fn from_name(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            sha256: String::new(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.name.is_empty() && self.sha256.is_empty()
    }
    /// Canonical identity string: `name@sha256` when a hash is present.
    pub fn identity(&self) -> String {
        if self.sha256.is_empty() {
            self.name.clone()
        } else {
            format!("{}@{}", self.name, self.sha256)
        }
    }
    pub fn short_sha256(&self) -> &str {
        self.sha256.get(..16).unwrap_or(&self.sha256)
    }
}
impl fmt::Display for ModelRoot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.identity())
    }
}
impl From<&str> for ModelRoot {
    fn from(value: &str) -> Self {
        Self::from_name(value)
    }
}
impl From<String> for ModelRoot {
    fn from(value: String) -> Self {
        Self::from_name(value)
    }
}

/// Explicit epoch/generation identity.  A `Generation` tags buffers,
/// transfers, and leases; stale generations are rejected by the residency
/// machine before any state transition.
#[derive(
    Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord,
)]
pub struct Generation {
    pub graph: u64,
    pub decode_epoch: u64,
}
impl Generation {
    pub const fn new(graph: u64, decode_epoch: u64) -> Self {
        Self {
            graph,
            decode_epoch,
        }
    }
    pub const fn zero() -> Self {
        Self {
            graph: 0,
            decode_epoch: 0,
        }
    }
    pub fn next_graph(&self) -> Self {
        Self {
            graph: self.graph.saturating_add(1),
            ..*self
        }
    }
    pub fn next_decode(&self) -> Self {
        Self {
            decode_epoch: self.decode_epoch.saturating_add(1),
            ..*self
        }
    }
}
impl fmt::Display for Generation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "g{}d{}", self.graph, self.decode_epoch)
    }
}

/// Typed capability set over deterministic `BTreeMap` storage.  Serialized
/// transparently (a plain JSON object), so hardware phenotype JSON and hashes
/// are unchanged.  Capability validation fails before execution, never
/// silently downgrades.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(transparent)]
pub struct CapabilitySet(BTreeMap<String, bool>);
impl CapabilitySet {
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }
    pub fn insert(&mut self, key: impl Into<String>, supported: bool) {
        self.0.insert(key.into(), supported);
    }
    pub fn supports(&self, key: &str) -> bool {
        self.0.get(key).copied().unwrap_or(false)
    }
    pub fn contains(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.0.keys()
    }
    /// Returns the sorted list of required capabilities that are unsupported.
    pub fn missing(&self, required: &[&str]) -> Vec<String> {
        required
            .iter()
            .filter(|key| !self.supports(key))
            .map(|key| (*key).to_string())
            .collect()
    }
    pub fn require_all(&self, required: &[&str]) -> Result<()> {
        let missing = self.missing(required);
        if missing.is_empty() {
            Ok(())
        } else {
            Err(HarError::Unsupported {
                kind: "capability",
                message: missing.join(", "),
            })
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = (&String, &bool)> {
        self.0.iter()
    }
}
impl From<BTreeMap<String, bool>> for CapabilitySet {
    fn from(value: BTreeMap<String, bool>) -> Self {
        Self(value)
    }
}
impl std::ops::Deref for CapabilitySet {
    type Target = BTreeMap<String, bool>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum HarError {
    #[error("invalid {kind}: {message}")]
    Invalid { kind: &'static str, message: String },
    #[error("identity mismatch for {field}: expected {expected}, got {actual}")]
    IdentityMismatch {
        field: String,
        expected: String,
        actual: String,
    },
    #[error("unsupported {kind}: {message}")]
    Unsupported { kind: &'static str, message: String },
    #[error("I/O error: {0}")]
    Io(String),
    #[error("serialization error: {0}")]
    Serialization(String),
}

impl From<io::Error> for HarError {
    fn from(value: io::Error) -> Self {
        Self::Io(value.to_string())
    }
}
impl From<serde_json::Error> for HarError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serialization(value.to_string())
    }
}

pub type Result<T> = std::result::Result<T, HarError>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MemoryTier {
    NvmeCold,
    RamMapped,
    RamPinned,
    VramResident,
    VramSlot,
    ReconstructionScratch,
    CpuHeap,
}

impl MemoryTier {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NvmeCold => "NVME_COLD",
            Self::RamMapped => "RAM_MAPPED",
            Self::RamPinned => "RAM_PINNED",
            Self::VramResident => "VRAM_RESIDENT",
            Self::VramSlot => "VRAM_SLOT",
            Self::ReconstructionScratch => "RECONSTRUCTION_SCRATCH",
            Self::CpuHeap => "CPU_HEAP",
        }
    }
}
impl fmt::Display for MemoryTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ResidencyState {
    Unavailable,
    Indexed,
    ReadQueued,
    Reading,
    ReadyHost,
    TransferQueued,
    CopyingToVram,
    ReadyVram,
    Computing,
    Evictable,
    Evicting,
    Error,
}

impl ResidencyState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Unavailable => "UNAVAILABLE",
            Self::Indexed => "INDEXED",
            Self::ReadQueued => "READ_QUEUED",
            Self::Reading => "READING",
            Self::ReadyHost => "READY_HOST",
            Self::TransferQueued => "TRANSFER_QUEUED",
            Self::CopyingToVram => "COPYING_TO_VRAM",
            Self::ReadyVram => "READY_VRAM",
            Self::Computing => "COMPUTING",
            Self::Evictable => "EVICTABLE",
            Self::Evicting => "EVICTING",
            Self::Error => "ERROR",
        }
    }
    pub fn can_transition_to(&self, next: &Self) -> bool {
        use ResidencyState::*;
        match self {
            Unavailable => matches!(next, Indexed | Error),
            Indexed => matches!(next, ReadQueued | ReadyHost | Error),
            ReadQueued => matches!(next, Reading | Error),
            Reading => matches!(next, ReadyHost | Error),
            ReadyHost => matches!(next, TransferQueued | Computing | Evictable | Error),
            TransferQueued => matches!(next, CopyingToVram | Error),
            CopyingToVram => matches!(next, ReadyVram | Error),
            ReadyVram => matches!(next, Computing | Evictable | Evicting | Error),
            Computing => matches!(next, ReadyHost | ReadyVram | Evictable | Error),
            Evictable => matches!(next, Evicting | Computing | Error),
            Evicting => matches!(next, ReadyHost | Unavailable | Error),
            Error => false,
        }
    }
}
impl fmt::Display for ResidencyState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BackendKind {
    None,
    Cpu,
    Vulkan,
}
impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::None => "NONE",
            Self::Cpu => "CPU",
            Self::Vulkan => "VULKAN",
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum KernelKind {
    Unknown,
    DenseMulMat,
    QuantizedMulMat,
    EmbeddingLookup,
    Normalization,
    Attention,
    MtpVerify,
    Sampling,
    Copy,
    Q4KMatVec,
}
impl KernelKind {
    /// Canonical capability name required by a plan that dispatches this
    /// kernel.  The plan loader validates these before execution; a missing
    /// capability is an error unless the fallback contract permits a
    /// reference adapter.
    pub fn capability(&self) -> &'static str {
        match self {
            Self::Unknown => "unknown_kernel",
            Self::DenseMulMat => "dense_mul_mat",
            Self::QuantizedMulMat => "quantized_mul_mat",
            Self::EmbeddingLookup => "embedding_lookup",
            Self::Normalization => "normalization",
            Self::Attention => "attention",
            Self::MtpVerify => "mtp_verify",
            Self::Sampling => "sampling",
            Self::Copy => "copy",
            Self::Q4KMatVec => "q4_k_matvec",
        }
    }
}
impl fmt::Display for KernelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Unknown => "UNKNOWN",
            Self::DenseMulMat => "DENSE_MUL_MAT",
            Self::QuantizedMulMat => "QUANTIZED_MUL_MAT",
            Self::EmbeddingLookup => "EMBEDDING_LOOKUP",
            Self::Normalization => "NORMALIZATION",
            Self::Attention => "ATTENTION",
            Self::MtpVerify => "MTP_VERIFY",
            Self::Sampling => "SAMPLING",
            Self::Copy => "COPY",
            Self::Q4KMatVec => "Q4_K_MATVEC",
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ReasonCode {
    FitsVram,
    ExceedsVramBudget,
    KernelUnavailable,
    DependencyClosureCpu,
    OutputConsumerCpu,
    MtpReservedMemory,
    KvReservedMemory,
    StagingReservedMemory,
    HotnessPriority,
    UserPolicy,
    ModelPackageImported,
}
impl fmt::Display for ReasonCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryBudget {
    pub tier: MemoryTier,
    pub capacity_bytes: u64,
    pub reserved_bytes: u64,
    pub assigned_bytes: u64,
    pub reservation_basis: String,
}
impl MemoryBudget {
    pub fn available_bytes(&self) -> u64 {
        self.capacity_bytes
            .saturating_sub(self.reserved_bytes.saturating_add(self.assigned_bytes))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ResourceBudget {
    pub tiers: Vec<MemoryBudget>,
    pub kv_bytes: u64,
    pub staging_bytes: u64,
    pub scratch_bytes: u64,
    pub model_bytes: u64,
}
impl ResourceBudget {
    pub fn budget_for(&self, tier: &MemoryTier) -> Option<&MemoryBudget> {
        self.tiers.iter().find(|item| &item.tier == tier)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TransferPlan {
    pub id: String,
    pub resource_id: String,
    pub source: MemoryTier,
    pub destination: MemoryTier,
    pub bytes: u64,
    pub alignment_bytes: u64,
    pub staging_required: bool,
    pub queue: String,
    pub dependency: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct KernelRequirement {
    pub kernel: KernelKind,
    pub backend: BackendKind,
    pub format: QuantFormat,
    pub required_capability: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ExactnessMode {
    Exact,
    WithinTolerance,
    Approximate,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExactnessContract {
    pub mode: ExactnessMode,
    pub output_hash_required: bool,
    pub state_boundary_hash_required: bool,
    pub numerical_tolerance: f64,
    pub authority: String,
}
impl Default for ExactnessContract {
    fn default() -> Self {
        Self {
            mode: ExactnessMode::Exact,
            output_hash_required: true,
            state_boundary_hash_required: true,
            numerical_tolerance: 0.0,
            authority: "native-rust".into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FallbackContract {
    pub authority_backend: BackendKind,
    pub on_unknown_capacity: String,
    pub on_stale_generation: String,
    pub on_kernel_unavailable: String,
}
impl Default for FallbackContract {
    fn default() -> Self {
        Self {
            authority_backend: BackendKind::Cpu,
            on_unknown_capacity: "reject_plan".into(),
            on_stale_generation: "fail_closed".into(),
            on_kernel_unavailable: "reject_plan".into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TelemetryContract {
    pub schema: String,
    pub record_residency: bool,
    pub record_transfers: bool,
    pub record_operation_hashes: bool,
    pub timing_is_advisory: bool,
}
impl Default for TelemetryContract {
    fn default() -> Self {
        Self {
            schema: "har.telemetry.v1".into(),
            record_residency: true,
            record_transfers: true,
            record_operation_hashes: true,
            timing_is_advisory: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct EpochNamespace {
    pub model_root: ModelRoot,
    pub graph_generation: u64,
    pub decode_epoch: u64,
    pub sequence_id: u64,
}
impl EpochNamespace {
    pub fn new(model_root: impl Into<ModelRoot>, sequence_id: u64) -> Self {
        Self {
            model_root: model_root.into(),
            graph_generation: 0,
            decode_epoch: 0,
            sequence_id,
        }
    }
    pub fn next_decode(&self) -> Self {
        Self {
            decode_epoch: self.decode_epoch + 1,
            ..self.clone()
        }
    }
    pub fn generation(&self) -> Generation {
        Generation::new(self.graph_generation, self.decode_epoch)
    }
    pub fn with_generation(mut self, generation: Generation) -> Self {
        self.graph_generation = generation.graph;
        self.decode_epoch = generation.decode_epoch;
        self
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CpuPhenotype {
    pub model: String,
    pub architecture: String,
    pub logical_threads: u32,
    pub physical_cores: u32,
    pub simd_features: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuPhenotype {
    pub name: String,
    pub vendor: String,
    pub driver: String,
    pub api_version: String,
    pub rdna_generation: String,
    pub subgroup_size: u32,
    pub transfer_queue_count: u32,
    pub compute_queue_count: u32,
    pub vram_total_bytes: u64,
    pub safe_allocatable_vram_bytes: u64,
    pub supported_formats: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RamPhenotype {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub pinned_limit_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct StoragePhenotype {
    pub probe_path: String,
    pub filesystem: String,
    pub source: String,
    pub device_model: String,
    pub on_nvme: bool,
    pub odirect_supported: bool,
    pub required_alignment_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct HardwarePhenotype {
    pub schema: String,
    pub hostname: String,
    pub kernel: String,
    pub os_release: String,
    pub cpu: CpuPhenotype,
    pub gpu: GpuPhenotype,
    pub ram: RamPhenotype,
    pub storage: StoragePhenotype,
    pub capabilities: CapabilitySet,
}
impl Default for HardwarePhenotype {
    fn default() -> Self {
        Self::synthetic_rdna4()
    }
}
impl HardwarePhenotype {
    pub fn synthetic_rdna4() -> Self {
        let mut capabilities = CapabilitySet::new();
        for key in [
            "vulkan",
            "q4_k",
            "q5_k",
            "q6_k",
            "q8_k",
            "subgroup_64",
            "dense_mul_mat",
            "quantized_mul_mat",
            "q4_k_matvec",
            "embedding_lookup",
            "normalization",
            "attention",
            "mtp_verify",
            "sampling",
            "copy",
        ] {
            capabilities.insert(key, true);
        }
        Self {
            schema: "har.hardware_phenotype.v1".into(),
            hostname: String::new(),
            kernel: String::new(),
            os_release: String::new(),
            cpu: CpuPhenotype {
                model: String::new(),
                architecture: "x86_64".into(),
                logical_threads: 0,
                physical_cores: 0,
                simd_features: Vec::new(),
            },
            gpu: GpuPhenotype {
                name: "Vulkan RDNA4 reference device".into(),
                vendor: "AMD".into(),
                driver: "RADV".into(),
                api_version: "Vulkan".into(),
                rdna_generation: "RDNA4".into(),
                subgroup_size: 64,
                transfer_queue_count: 4,
                compute_queue_count: 1,
                vram_total_bytes: 16_000_000_000,
                safe_allocatable_vram_bytes: 15_000_000_000,
                supported_formats: vec!["Q4_K".into(), "Q5_K".into(), "Q6_K".into(), "Q8_K".into()],
            },
            ram: RamPhenotype {
                total_bytes: 0,
                available_bytes: 0,
                pinned_limit_bytes: 0,
            },
            storage: StoragePhenotype {
                probe_path: String::new(),
                filesystem: String::new(),
                source: String::new(),
                device_model: String::new(),
                on_nvme: true,
                odirect_supported: false,
                required_alignment_bytes: 4096,
            },
            capabilities,
        }
    }
    pub fn identity(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}",
            self.gpu.name,
            self.gpu.driver,
            self.gpu.rdna_generation,
            self.gpu.subgroup_size,
            self.gpu.vram_total_bytes
        )
    }
    /// Import read-only external probe JSON without importing any Vulkan
    /// handle. Unknown probe fields remain outside the stable Rust contract.
    pub fn from_probe_json(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let root: serde_json::Value = serde_json::from_slice(&bytes)?;
        let mut result = Self::synthetic_rdna4();
        let text = |object: &serde_json::Value, key: &str| {
            object
                .get(key)
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_owned()
        };
        result.hostname = text(&root, "hostname");
        result.kernel = text(&root, "kernel");
        result.os_release = text(&root, "os_release");
        if let Some(cpu) = root.get("cpu") {
            result.cpu.model = text(cpu, "model");
            result.cpu.architecture = text(cpu, "architecture");
            result.cpu.logical_threads = cpu
                .get("logical_threads")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as u32;
            result.cpu.physical_cores = cpu
                .get("physical_cores")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as u32;
            result.cpu.simd_features = cpu
                .get("simd_features")
                .and_then(serde_json::Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(serde_json::Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default();
        }
        if let Some(ram) = root.get("ram") {
            result.ram.total_bytes = ram
                .get("total_bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            result.ram.available_bytes = ram
                .get("available_bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            result.ram.pinned_limit_bytes = ram
                .get("pinned_limit_bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
        }
        if let Some(gpu) = root.get("gpu") {
            result.gpu.name = text(gpu, "name");
            result.gpu.vendor = text(gpu, "vendor");
            result.gpu.driver = text(gpu, "driver");
            result.gpu.api_version = text(gpu, "api_version");
            result.gpu.rdna_generation = text(gpu, "rdna_generation");
            result.gpu.subgroup_size = gpu
                .get("subgroup_size")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as u32;
            result.gpu.transfer_queue_count = gpu
                .get("transfer_queue_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as u32;
            result.gpu.compute_queue_count = gpu
                .get("compute_queue_count")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as u32;
            result.gpu.vram_total_bytes = gpu
                .get("vram_total_bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            result.gpu.safe_allocatable_vram_bytes = gpu
                .get("safe_allocatable_vram_bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            result.gpu.supported_formats = gpu
                .get("supported_formats")
                .and_then(serde_json::Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(serde_json::Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default();
        }
        if let Some(storage) = root.get("storage") {
            result.storage.probe_path = text(storage, "probe_path");
            result.storage.filesystem = text(storage, "filesystem");
            result.storage.source = text(storage, "source");
            result.storage.device_model = text(storage, "device_model");
            result.storage.on_nvme = storage
                .get("on_nvme")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            result.storage.odirect_supported = storage
                .get("odirect_supported")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false);
            result.storage.required_alignment_bytes = storage
                .get("required_alignment_bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
        }
        result.schema = root
            .get("schema")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("har.hardware_phenotype.v1")
            .into();
        Ok(result)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeManifest {
    pub schema: String,
    pub runtime_name: String,
    pub runtime_version: String,
    pub source_commit: String,
    pub reference_commit: String,
    pub model_sha256: String,
    pub hardware_sha256: String,
    pub plan_sha256: String,
    pub operation_id: String,
    pub exactness: ExactnessMode,
    pub fallback: FallbackContract,
    pub telemetry: TelemetryContract,
    pub owned_subsystems: Vec<String>,
    pub reference_adapters: Vec<String>,
    pub plan_validation: String,
    pub notes: Vec<String>,
}
impl Default for RuntimeManifest {
    fn default() -> Self {
        Self {
            schema: MANIFEST_VERSION.into(),
            runtime_name: "Hardware-Aware Runtime".into(),
            runtime_version: "0.1.0-rust".into(),
            source_commit: String::new(),
            reference_commit: String::new(),
            model_sha256: String::new(),
            hardware_sha256: String::new(),
            plan_sha256: String::new(),
            operation_id: String::new(),
            exactness: ExactnessMode::Exact,
            fallback: FallbackContract::default(),
            telemetry: TelemetryContract::default(),
            owned_subsystems: vec![
                "typed-control-plane".into(),
                "plan-validation".into(),
                "residency-identities".into(),
                "operation-ordering".into(),
                "telemetry".into(),
            ],
            reference_adapters: Vec::new(),
            plan_validation: "NOT_RUN".into(),
            notes: Vec::new(),
        }
    }
}

pub fn unix_timestamp_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    hex::encode(digest.finalize())
}
pub fn sha256_file(path: impl AsRef<Path>) -> Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buf)?;
        if count == 0 {
            break;
        }
        digest.update(&buf[..count]);
    }
    Ok(hex::encode(digest.finalize()))
}
pub fn sha256_f32(values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    sha256_bytes(&bytes)
}

/// Canonical JSON uses recursively sorted object keys and compact separators.
pub fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let value = serde_json::to_value(value)?;
    let normalized = canonicalize(value);
    Ok(serde_json::to_vec(&normalized)?)
}
pub fn canonical_sha256<T: Serialize>(value: &T) -> Result<String> {
    Ok(sha256_bytes(&canonical_json(value)?))
}
fn canonicalize(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(canonicalize).collect())
        }
        serde_json::Value::Object(values) => {
            let mut sorted = BTreeMap::new();
            for (key, value) in values {
                sorted.insert(key, canonicalize(value));
            }
            let mut result = serde_json::Map::new();
            for (key, value) in sorted {
                result.insert(key, value);
            }
            serde_json::Value::Object(result)
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residency_graph_is_fail_closed() {
        assert!(ResidencyState::Unavailable.can_transition_to(&ResidencyState::Indexed));
        assert!(!ResidencyState::ReadyVram.can_transition_to(&ResidencyState::Reading));
        assert!(!ResidencyState::Error.can_transition_to(&ResidencyState::ReadyHost));
    }

    #[test]
    fn canonical_hash_is_map_order_independent() {
        let a: BTreeMap<&str, u64> = [("b", 2), ("a", 1)].into_iter().collect();
        let b: BTreeMap<&str, u64> = [("a", 1), ("b", 2)].into_iter().collect();
        assert_eq!(canonical_sha256(&a).unwrap(), canonical_sha256(&b).unwrap());
    }

    #[test]
    fn quant_format_wire_form_matches_gguf_names() {
        assert_eq!(
            serde_json::to_string(&QuantFormat::Q4K).unwrap(),
            "\"Q4_K\""
        );
        assert_eq!(
            serde_json::to_string(&QuantFormat::Bf16).unwrap(),
            "\"BF16\""
        );
        assert_eq!(QuantFormat::from_ggml_type(12), QuantFormat::Q4K);
        assert_eq!(QuantFormat::from_ggml_type(0).as_str(), "F32");
        assert!(QuantFormat::Q4K.is_quantized());
        assert!(!QuantFormat::F32.is_quantized());
        assert_eq!(QuantFormat::Q4K.kernel_hint(), KernelKind::Q4KMatVec);
        assert_eq!(QuantFormat::F16.kernel_hint(), KernelKind::DenseMulMat);
    }

    #[test]
    fn tensor_role_wire_form_matches_legacy_strings() {
        assert_eq!(
            serde_json::to_string(&TensorRole::FeedForward).unwrap(),
            "\"FEED_FORWARD\""
        );
        assert_eq!(serde_json::to_string(&TensorRole::Mtp).unwrap(), "\"MTP\"");
        assert_eq!(
            TensorRole::classify("blk.0.ffn_gate.weight"),
            TensorRole::FeedForward
        );
        assert_eq!(
            TensorRole::classify("blk.0.attn_qkv.weight"),
            TensorRole::Attention
        );
        assert_eq!(
            TensorRole::classify("output.weight"),
            TensorRole::OutputProjection
        );
    }

    #[test]
    fn model_root_and_generation_identities_are_explicit() {
        let root = ModelRoot::new(
            "test-model",
            "0000000000000000000000000000000000000000000000000000000000000000",
        );
        assert_eq!(root.short_sha256(), "0000000000000000");
        assert!(root.identity().starts_with("test-model@00000000"));
        assert!(!root.is_empty());
        assert!(!ModelRoot::from_name("m").is_empty());
        let generation = Generation::zero().next_graph().next_decode();
        assert_eq!(generation, Generation::new(1, 1));
        let namespace = EpochNamespace::new(root, 7).with_generation(generation);
        assert_eq!(namespace.generation(), generation);
        assert_eq!(namespace.model_root.name, "test-model");
    }

    #[test]
    fn capability_set_is_fail_closed_and_transparent() {
        let mut set = CapabilitySet::new();
        set.insert("vulkan", true);
        set.insert("q4_k", false);
        assert!(set.supports("vulkan"));
        assert!(!set.supports("q4_k"));
        let missing = set.missing(&["vulkan", "q4_k"]);
        assert_eq!(missing, vec!["q4_k"]);
        assert!(set.require_all(&["vulkan"]).is_ok());
        assert!(set.require_all(&["vulkan", "q4_k"]).is_err());
        // Transparent serialization: a plain JSON object, not a tagged newtype.
        assert_eq!(
            serde_json::to_string(&set).unwrap(),
            "{\"q4_k\":false,\"vulkan\":true}"
        );
        let hardware = HardwarePhenotype::synthetic_rdna4();
        assert!(serde_json::to_string(&hardware.capabilities)
            .unwrap()
            .starts_with('{'));
    }

    #[test]
    fn interface_version_names_are_v0_frozen() {
        assert_eq!(v0::CORE.schema(), "har.core.v1");
        assert_eq!(v0::PLAN.schema(), "har.execution_plan.v1");
        assert_eq!(v0::RUNTIME_MANIFEST.schema(), "har.runtime_manifest.v1");
    }
}
