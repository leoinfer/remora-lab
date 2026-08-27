//! Runtime-consumable HAR model package and sparse expert-sidecar formats.
//!
//! The package is deliberately boring: a fixed little-endian header, a
//! canonical JSON manifest, aligned payloads, and SHA-256 checks.  Policy is
//! resolved by the compiler before this file reaches the token loop.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const PACKAGE_SCHEMA: &str = "har.model_package.v1";
pub const PACKED_PACKAGE_SCHEMA: &str = "har.packed_model_package.v0";
pub const SIDECAR_SCHEMA: &str = "har.expert_sidecar.v1";
pub const PACKAGE_MAGIC: &[u8; 8] = b"HARPKG01";
pub const SIDECAR_MAGIC: &[u8; 8] = b"HARSIDE1";
pub const ALIGNMENT: u64 = 4096;
pub const PACKAGE_HEADER_BYTES: u64 = 128;
pub const SIDECAR_HEADER_BYTES: u64 = 4096;

#[derive(Debug)]
pub enum PackageError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Invalid(String),
}

impl fmt::Display for PackageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Json(error) => write!(f, "JSON error: {error}"),
            Self::Invalid(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for PackageError {}
impl From<std::io::Error> for PackageError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}
impl From<serde_json::Error> for PackageError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

pub type Result<T> = std::result::Result<T, PackageError>;

pub fn align_up(value: u64, alignment: u64) -> u64 {
    assert!(alignment > 0);
    value.saturating_add(alignment - 1) / alignment * alignment
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex::encode(digest)
}

pub fn sha256_file(path: impl AsRef<Path>) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TensorRole {
    Embedding,
    AttentionQ,
    AttentionK,
    AttentionV,
    AttentionO,
    DenseFfnGate,
    DenseFfnUp,
    DenseFfnDown,
    RoutedExpertGate,
    RoutedExpertUp,
    RoutedExpertDown,
    SharedExpert,
    Router,
    Normalization,
    Output,
    Mtp,
    #[default]
    MetadataOther,
}

impl TensorRole {
    pub fn is_routed_expert(&self) -> bool {
        matches!(
            self,
            Self::RoutedExpertGate | Self::RoutedExpertUp | Self::RoutedExpertDown
        )
    }
    pub fn is_protected_default(&self) -> bool {
        matches!(self, Self::Router | Self::Output | Self::Mtp)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTier {
    NvmeCold,
    #[default]
    RamMapped,
    RamPinned,
    VramResident,
    VramSlot,
    ReconstructionScratch,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StorageLocation {
    #[default]
    Model,
    ExpertSidecar,
    Metadata,
    External,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SourceModel {
    pub path: String,
    pub root_sha256: String,
    pub file_bytes: u64,
    pub gguf_version: u32,
    pub architecture: String,
    pub model_name: String,
    pub tensor_count: u64,
    pub metadata_count: u64,
    pub data_offset: u64,
    pub tensor_payload_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct HardwarePhenotype {
    pub hardware_id: String,
    pub backend: String,
    pub gpu_arch: String,
    pub supported_formats: Vec<String>,
    pub kernel_paths: Vec<String>,
    pub tensor_alignment: u64,
    pub sidecar_alignment: u64,
    pub vram_bytes: Option<u64>,
    pub ram_bytes: Option<u64>,
    pub nvme_bytes: Option<u64>,
    pub persistent_vulkan_slots: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct TensorRecord {
    pub tensor_id: String,
    pub name: String,
    pub dimensions: Vec<u64>,
    pub element_count: u64,
    pub source_bytes: u64,
    pub source_quant_format: String,
    pub planned_bytes: Option<u64>,
    pub planned_quant_format: Option<String>,
    pub layer: Option<u32>,
    pub expert_id: Option<u32>,
    pub projection: Option<String>,
    pub role: TensorRole,
    pub tensor_class: String,
    pub sensitivity_placeholder: Option<f64>,
    pub supported_kernels: Vec<String>,
    pub required_kernels: Vec<String>,
    pub alignment: u64,
    pub source_offset: Option<u64>,
    pub source_file: Option<String>,
    pub storage_location: StorageLocation,
    pub planned_memory_tier: MemoryTier,
    pub payload_location_id: Option<String>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct CalibrationEvidence {
    pub schema: String,
    pub root_sha256: Option<String>,
    pub source_paths: Vec<String>,
    pub row_count: u64,
    pub behavioral_row_count: u64,
    pub status: String,
    pub metrics: Vec<String>,
    pub quality_claim_allowed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct AllocationRecord {
    pub tensor_id: String,
    pub selected_format: String,
    pub expected_bytes: u64,
    pub quality_loss: Option<f64>,
    pub behavioral_evidence: bool,
    pub protected: bool,
    pub routed_bytes_per_active_token: u64,
    pub sensitivity_justification: String,
    pub unresolved_risks: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct PayloadLocation {
    pub id: String,
    pub offset: u64,
    pub bytes: u64,
    pub alignment: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SidecarEntry {
    pub source_tensor_id: String,
    pub layer: u32,
    pub expert_id: u32,
    pub projection: String,
    pub projection_order: u8,
    pub quant_format: String,
    pub offset: u64,
    pub payload_bytes: u64,
    pub alignment: u64,
    pub scale_metadata_location: BTreeMap<String, serde_json::Value>,
    pub kernel_requirement: String,
    pub checksum_sha256: String,
    pub native_payload: bool,
    pub slot_class: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct SidecarManifest {
    pub schema: String,
    pub version: u32,
    pub model_identity: String,
    pub source_model_identity: String,
    pub source_model_sha256: Option<String>,
    pub compiler_version: String,
    pub alignment: u64,
    pub projection_order: Vec<String>,
    pub direct_index: bool,
    pub runtime_tensor_scanning: bool,
    pub entry_index: BTreeMap<String, usize>,
    pub entries: Vec<SidecarEntry>,
    pub index_sha256: Option<String>,
    pub total_bytes: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackageManifest {
    pub schema: String,
    pub package_version: u32,
    pub compiler_version: String,
    pub source_model: SourceModel,
    pub source_model_root_sha256: String,
    pub hardware: HardwarePhenotype,
    pub tensors: Vec<TensorRecord>,
    pub required_kernels: Vec<String>,
    pub quality_evidence: CalibrationEvidence,
    pub allocation: Vec<AllocationRecord>,
    pub sidecar: Option<SidecarManifest>,
    pub payload_locations: Vec<PayloadLocation>,
    pub fallback_source_relationship: String,
    pub unresolved_risks: Vec<String>,
    pub claims: BTreeMap<String, String>,
    #[serde(default)]
    pub packed_entries: Vec<PackedEntry>,
}

/// One tensor slice in a packed, directly indexed package.  The source bytes
/// are copied byte-for-byte from the GGUF payload (no requantisation), each
/// entry is independently addressable and 4 KiB aligned, and the runtime
/// resolves it by manifest index, never by scanning tensor names.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct PackedEntry {
    pub source_tensor_id: String,
    pub source_file: String,
    pub source_offset: u64,
    pub source_bytes: u64,
    pub payload_location_id: String,
    pub dimensions: Vec<u64>,
    pub element_count: u64,
    pub quant_format: String,
    pub layer: Option<u32>,
    pub expert_id: Option<u32>,
    pub projection: Option<String>,
    pub role: TensorRole,
    pub tensor_class: String,
    pub kernel_requirement: String,
    pub representation_identity: String,
}

impl PackageManifest {
    pub fn new(
        source_model: SourceModel,
        hardware: HardwarePhenotype,
        compiler_version: impl Into<String>,
    ) -> Self {
        let source_root = source_model.root_sha256.clone();
        Self {
            schema: PACKAGE_SCHEMA.to_owned(),
            package_version: 1,
            compiler_version: compiler_version.into(),
            source_model,
            source_model_root_sha256: source_root,
            hardware,
            tensors: Vec::new(),
            required_kernels: Vec::new(),
            quality_evidence: CalibrationEvidence {
                schema: "har.calibration_capture.v1".to_owned(),
                status: "blocked_pending_review".to_owned(),
                metrics: vec![
                    "logit_kl".to_owned(),
                    "top_token_agreement".to_owned(),
                    "routing_topk_agreement".to_owned(),
                    "mtp_acceptance".to_owned(),
                    "rare_fact_exact".to_owned(),
                    "code_math_score".to_owned(),
                    "long_context_retrieval".to_owned(),
                ],
                ..Default::default()
            },
            allocation: Vec::new(),
            sidecar: None,
            payload_locations: Vec::new(),
            fallback_source_relationship:
                "source GGUF remains authoritative; package is a compiled representation".to_owned(),
            unresolved_risks: Vec::new(),
            claims: BTreeMap::new(),
            packed_entries: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct PackagePayload {
    pub id: String,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
struct PackageHeader {
    manifest_offset: u64,
    manifest_bytes: u64,
    payload_offset: u64,
    payload_bytes: u64,
    manifest_sha256: [u8; 32],
    package_root_sha256: [u8; 32],
}

impl PackageHeader {
    fn encode(&self) -> [u8; PACKAGE_HEADER_BYTES as usize] {
        let mut bytes = [0u8; PACKAGE_HEADER_BYTES as usize];
        bytes[..8].copy_from_slice(PACKAGE_MAGIC);
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&(PACKAGE_HEADER_BYTES as u32).to_le_bytes());
        bytes[16..20].copy_from_slice(&(ALIGNMENT as u32).to_le_bytes());
        bytes[24..32].copy_from_slice(&self.manifest_offset.to_le_bytes());
        bytes[32..40].copy_from_slice(&self.manifest_bytes.to_le_bytes());
        bytes[40..48].copy_from_slice(&self.payload_offset.to_le_bytes());
        bytes[48..56].copy_from_slice(&self.payload_bytes.to_le_bytes());
        bytes[56..88].copy_from_slice(&self.manifest_sha256);
        bytes[88..120].copy_from_slice(&self.package_root_sha256);
        bytes
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < PACKAGE_HEADER_BYTES as usize || &bytes[..8] != PACKAGE_MAGIC {
            return Err(PackageError::Invalid("invalid HARPKG01 header".to_owned()));
        }
        let u32_at = |offset: usize| -> u32 {
            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
        };
        let u64_at = |offset: usize| -> u64 {
            u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
        };
        if u32_at(8) != 1
            || u32_at(12) != PACKAGE_HEADER_BYTES as u32
            || u32_at(16) != ALIGNMENT as u32
        {
            return Err(PackageError::Invalid(
                "unsupported HAR package header version/alignment".to_owned(),
            ));
        }
        let mut manifest_hash = [0u8; 32];
        manifest_hash.copy_from_slice(&bytes[56..88]);
        let mut root_hash = [0u8; 32];
        root_hash.copy_from_slice(&bytes[88..120]);
        Ok(Self {
            manifest_offset: u64_at(24),
            manifest_bytes: u64_at(32),
            payload_offset: u64_at(40),
            payload_bytes: u64_at(48),
            manifest_sha256: manifest_hash,
            package_root_sha256: root_hash,
        })
    }
}

pub struct PackageWriter;

impl PackageWriter {
    pub fn write(
        path: impl AsRef<Path>,
        manifest: &PackageManifest,
        payloads: &[PackagePayload],
    ) -> Result<PackageManifest> {
        let mut output_manifest = manifest.clone();
        let mut payload_offset = ALIGNMENT;
        let mut locations = Vec::new();
        let mut manifest_bytes: Vec<u8>;
        for _ in 0..4 {
            locations.clear();
            let mut cursor = payload_offset;
            for payload in payloads {
                cursor = align_up(cursor, ALIGNMENT);
                locations.push(PayloadLocation {
                    id: payload.id.clone(),
                    offset: cursor,
                    bytes: payload.bytes.len() as u64,
                    alignment: ALIGNMENT,
                    sha256: sha256_bytes(&payload.bytes),
                });
                cursor += payload.bytes.len() as u64;
            }
            output_manifest.payload_locations = locations.clone();
            manifest_bytes = serde_json::to_vec(&output_manifest)?;
            let next = align_up(
                PACKAGE_HEADER_BYTES + manifest_bytes.len() as u64,
                ALIGNMENT,
            );
            if next == payload_offset {
                break;
            }
            payload_offset = next;
        }
        // One final pass makes the offsets stable if the final JSON length
        // crossed an alignment boundary.
        locations.clear();
        let mut cursor = payload_offset;
        for payload in payloads {
            cursor = align_up(cursor, ALIGNMENT);
            locations.push(PayloadLocation {
                id: payload.id.clone(),
                offset: cursor,
                bytes: payload.bytes.len() as u64,
                alignment: ALIGNMENT,
                sha256: sha256_bytes(&payload.bytes),
            });
            cursor += payload.bytes.len() as u64;
        }
        output_manifest.payload_locations = locations.clone();
        manifest_bytes = serde_json::to_vec(&output_manifest)?;
        let final_offset = align_up(
            PACKAGE_HEADER_BYTES + manifest_bytes.len() as u64,
            ALIGNMENT,
        );
        if final_offset != payload_offset {
            return Err(PackageError::Invalid(
                "package manifest did not reach a stable aligned offset".to_owned(),
            ));
        }
        let payload_bytes = cursor.saturating_sub(payload_offset);
        let mut payload_region = vec![0u8; payload_bytes as usize];
        for (payload, location) in payloads.iter().zip(locations.iter()) {
            let start = (location.offset - payload_offset) as usize;
            payload_region[start..start + payload.bytes.len()].copy_from_slice(&payload.bytes);
        }
        let mut root_input = Vec::with_capacity(manifest_bytes.len() + payload_region.len());
        root_input.extend_from_slice(&manifest_bytes);
        root_input.extend_from_slice(&payload_region);
        let manifest_digest = sha256_bytes(&manifest_bytes);
        let package_root = sha256_bytes(&root_input);
        let header = PackageHeader {
            manifest_offset: PACKAGE_HEADER_BYTES,
            manifest_bytes: manifest_bytes.len() as u64,
            payload_offset,
            payload_bytes,
            manifest_sha256: hex::decode(&manifest_digest).unwrap().try_into().unwrap(),
            package_root_sha256: hex::decode(&package_root).unwrap().try_into().unwrap(),
        };
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)?;
        file.write_all(&header.encode())?;
        file.write_all(&manifest_bytes)?;
        let after_manifest = PACKAGE_HEADER_BYTES + manifest_bytes.len() as u64;
        file.write_all(&vec![0u8; (payload_offset - after_manifest) as usize])?;
        file.write_all(&payload_region)?;
        file.sync_all()?;
        output_manifest
            .claims
            .insert("package_root_sha256".to_owned(), package_root);
        output_manifest
            .claims
            .insert("manifest_sha256".to_owned(), manifest_digest);
        Ok(output_manifest)
    }

    /// Write a bounded real-subset package.  Only the payloads listed in
    /// `entries` are copied from the GGUF source; the full model is never
    /// duplicated.  Each entry receives an independently 4 KiB-aligned,
    /// checksummed payload location and the manifest carries the exact source
    /// offset so a runtime can verify byte identity without scanning names.
    pub fn write_packed(
        path: impl AsRef<Path>,
        mut manifest: PackageManifest,
        source_gguf: impl AsRef<Path>,
        entries: &[PackedEntry],
    ) -> Result<PackageManifest> {
        if entries.is_empty() {
            return Err(PackageError::Invalid(
                "packed package needs at least one entry".to_owned(),
            ));
        }
        manifest.schema = PACKED_PACKAGE_SCHEMA.to_owned();
        manifest.packed_entries = entries.to_vec();
        manifest.claims.insert(
            "packed_format".to_owned(),
            "har.packed_model_package.v0".to_owned(),
        );
        manifest.claims.insert(
            "source_layout".to_owned(),
            "byte-identical GGUF payload slices, no requantisation".to_owned(),
        );
        let mut payloads = Vec::with_capacity(entries.len());
        let mut source = File::open(source_gguf.as_ref())?;
        for entry in entries {
            let mut bytes = vec![0u8; entry.source_bytes as usize];
            source.seek(SeekFrom::Start(entry.source_offset))?;
            source.read_exact(&mut bytes)?;
            payloads.push(PackagePayload {
                id: entry.payload_location_id.clone(),
                bytes,
            });
        }
        // `tensors` mirrors the packed entries as source descriptors so
        // existing consumers keep a stable view; the packed index is the
        // authoritative runtime index.
        Self::write(path, &manifest, &payloads)
    }
}

#[derive(Clone, Debug)]
pub struct VerifiedPackage {
    pub manifest: PackageManifest,
    pub package_root_sha256: String,
    pub manifest_sha256: String,
    pub path: PathBuf,
}

pub struct PackageReader;

impl PackageReader {
    pub fn verify(path: impl AsRef<Path>) -> Result<VerifiedPackage> {
        let path = path.as_ref();
        let bytes = fs::read(path)?;
        if bytes.len() < PACKAGE_HEADER_BYTES as usize {
            return Err(PackageError::Invalid("HAR package is truncated".to_owned()));
        }
        let header = PackageHeader::decode(&bytes[..PACKAGE_HEADER_BYTES as usize])?;
        let manifest_end = header
            .manifest_offset
            .checked_add(header.manifest_bytes)
            .ok_or_else(|| PackageError::Invalid("manifest range overflow".to_owned()))?;
        let payload_end = header
            .payload_offset
            .checked_add(header.payload_bytes)
            .ok_or_else(|| PackageError::Invalid("payload range overflow".to_owned()))?;
        if manifest_end as usize > bytes.len()
            || payload_end as usize > bytes.len()
            || header.payload_offset % ALIGNMENT != 0
        {
            return Err(PackageError::Invalid(
                "HAR package ranges are outside the file or unaligned".to_owned(),
            ));
        }
        let manifest_blob = &bytes[header.manifest_offset as usize..manifest_end as usize];
        let manifest_hash = sha256_bytes(manifest_blob);
        if hex::decode(&manifest_hash).unwrap().as_slice() != header.manifest_sha256.as_slice() {
            return Err(PackageError::Invalid(
                "HAR package manifest checksum mismatch".to_owned(),
            ));
        }
        let manifest: PackageManifest = serde_json::from_slice(manifest_blob)?;
        let payload_region = &bytes[header.payload_offset as usize..payload_end as usize];
        let mut root_input = Vec::with_capacity(manifest_blob.len() + payload_region.len());
        root_input.extend_from_slice(manifest_blob);
        root_input.extend_from_slice(payload_region);
        let package_root = sha256_bytes(&root_input);
        if hex::decode(&package_root).unwrap().as_slice() != header.package_root_sha256.as_slice() {
            return Err(PackageError::Invalid(
                "HAR package root checksum mismatch".to_owned(),
            ));
        }
        for location in &manifest.payload_locations {
            if location.alignment == 0
                || location.offset % location.alignment != 0
                || location.offset < header.payload_offset
                || location.offset + location.bytes > payload_end
            {
                return Err(PackageError::Invalid(format!(
                    "payload {} has an invalid range/alignment",
                    location.id
                )));
            }
            let payload =
                &bytes[location.offset as usize..(location.offset + location.bytes) as usize];
            if sha256_bytes(payload) != location.sha256 {
                return Err(PackageError::Invalid(format!(
                    "payload {} checksum mismatch",
                    location.id
                )));
            }
        }
        Ok(VerifiedPackage {
            manifest,
            package_root_sha256: package_root,
            manifest_sha256: manifest_hash,
            path: path.to_path_buf(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct VerifiedSidecar {
    pub manifest: SidecarManifest,
    pub index_sha256: String,
    pub path: PathBuf,
}

pub struct SidecarReader;

impl SidecarReader {
    pub fn verify(
        path: impl AsRef<Path>,
        expected_source_sha256: Option<&str>,
    ) -> Result<VerifiedSidecar> {
        let path = path.as_ref();
        let bytes = fs::read(path)?;
        if bytes.len() < SIDECAR_HEADER_BYTES as usize || &bytes[..8] != SIDECAR_MAGIC {
            return Err(PackageError::Invalid("invalid HARSIDE1 sidecar".to_owned()));
        }
        let u32_at = |offset: usize| -> u32 {
            u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
        };
        let u64_at = |offset: usize| -> u64 {
            u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
        };
        if u32_at(8) != 1
            || u32_at(12) != SIDECAR_HEADER_BYTES as u32
            || u32_at(16) != ALIGNMENT as u32
        {
            return Err(PackageError::Invalid(
                "unsupported sidecar version/header/alignment".to_owned(),
            ));
        }
        let index_offset = u64_at(24);
        let index_bytes = u64_at(32);
        let data_offset = u64_at(40);
        let data_bytes = u64_at(48);
        let index_end = index_offset + index_bytes;
        let data_end = data_offset + data_bytes;
        if index_end as usize > bytes.len()
            || data_end as usize > bytes.len()
            || data_offset % ALIGNMENT != 0
        {
            return Err(PackageError::Invalid(
                "sidecar index/data range is invalid".to_owned(),
            ));
        }
        let index_blob = &bytes[index_offset as usize..index_end as usize];
        let index_hash = sha256_bytes(index_blob);
        if hex::decode(&index_hash).unwrap().as_slice() != &bytes[56..88] {
            return Err(PackageError::Invalid(
                "sidecar index checksum mismatch".to_owned(),
            ));
        }
        let manifest: SidecarManifest = serde_json::from_slice(index_blob)?;
        if manifest.schema != SIDECAR_SCHEMA
            || manifest.alignment != ALIGNMENT
            || !manifest.direct_index
            || manifest.runtime_tensor_scanning
        {
            return Err(PackageError::Invalid(
                "sidecar manifest policy fields are invalid".to_owned(),
            ));
        }
        if let Some(expected) = expected_source_sha256 {
            if manifest.source_model_sha256.as_deref() != Some(expected) {
                return Err(PackageError::Invalid(
                    "sidecar source model root mismatch".to_owned(),
                ));
            }
        }
        let mut ranges: Vec<(u64, u64)> = Vec::new();
        for entry in &manifest.entries {
            if entry.alignment != ALIGNMENT
                || entry.offset % ALIGNMENT != 0
                || entry.offset < data_offset
                || entry.offset + entry.payload_bytes > data_end
            {
                return Err(PackageError::Invalid(format!(
                    "sidecar entry {} has invalid alignment/range",
                    entry.source_tensor_id
                )));
            }
            let payload =
                &bytes[entry.offset as usize..(entry.offset + entry.payload_bytes) as usize];
            if sha256_bytes(payload) != entry.checksum_sha256 {
                return Err(PackageError::Invalid(format!(
                    "sidecar entry {} checksum mismatch",
                    entry.source_tensor_id
                )));
            }
            if !matches!(entry.projection.as_str(), "gate" | "up" | "down") {
                return Err(PackageError::Invalid(format!(
                    "sidecar entry {} has invalid projection",
                    entry.source_tensor_id
                )));
            }
            ranges.push((entry.offset, entry.offset + entry.payload_bytes));
        }
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(PackageError::Invalid("sidecar payloads overlap".to_owned()));
            }
        }
        Ok(VerifiedSidecar {
            manifest,
            index_sha256: index_hash,
            path: path.to_path_buf(),
        })
    }

    pub fn lookup<'a>(
        manifest: &'a SidecarManifest,
        layer: u32,
        expert_id: u32,
        projection: &str,
    ) -> Option<&'a SidecarEntry> {
        let key = format!("{layer}/{expert_id}/{projection}");
        manifest
            .entry_index
            .get(&key)
            .and_then(|index| manifest.entries.get(*index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("har-{name}-{nonce}"))
    }

    fn manifest() -> PackageManifest {
        let source = SourceModel {
            path: "fixture.gguf".into(),
            root_sha256: "a".repeat(64),
            file_bytes: 10,
            ..Default::default()
        };
        PackageManifest::new(
            source,
            HardwarePhenotype {
                hardware_id: "rdna4".into(),
                backend: "vulkan".into(),
                ..Default::default()
            },
            "test",
        )
    }

    #[test]
    fn package_round_trip_verifies_manifest_payload_and_root() {
        let path = temp_path("package");
        let result = PackageWriter::write(
            &path,
            &manifest(),
            &[PackagePayload {
                id: "expert-gate".into(),
                bytes: vec![7; 8192],
            }],
        )
        .unwrap();
        assert_eq!(result.payload_locations.len(), 1);
        let verified = PackageReader::verify(&path).unwrap();
        assert_eq!(verified.manifest.source_model_root_sha256, "a".repeat(64));
        assert_eq!(verified.manifest.payload_locations[0].bytes, 8192);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn alignment_is_ceil_not_round() {
        assert_eq!(align_up(4097, 4096), 8192);
        assert_eq!(align_up(8192, 4096), 8192);
    }
}
