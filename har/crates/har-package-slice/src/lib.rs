//! Deterministic, git-safe bounded package slices (HARSLICE1).
//!
//! A package slice is a self-describing, bounded byte range of one source
//! tensor (exact row range), stored verbatim from the source GGUF with a
//! canonical manifest.  It exists so bounded execution evidence can be
//! committed to Git without ever committing model weights or oversized
//! package blobs:
//!
//! - payload: the exact source bytes of the row range (no conversion);
//! - manifest: model root, tensor identity, row range, offsets, logical and
//!   physical bytes, quant/block geometry, checksums, package root of the
//!   deterministic reconstruction, reproduction command;
//! - no timestamps, no absolute paths, no machine identity: the file is
//!   byte-deterministic for identical inputs;
//! - fail-closed readers: wrong source root, wrong row range, truncated
//!   data, stale metadata, and checksum mismatches are rejected.
//!
//! Format (all little-endian):
//!
//! ```text
//! [0..8)    magic b"HARSLC01"
//! [8..12)   format_version u32 = 1
//! [12..16)  header_bytes u32 = 64
//! [16..20)  alignment u32 = 4096
//! [20..24)  reserved u32 = 0
//! [24..32)  manifest_offset u64
//! [32..40)  manifest_bytes u64
//! [40..48)  payload_offset u64
//! [48..56)  payload_bytes u64          (logical payload bytes)
//! [56..88)  manifest_sha256 [32]
//! [88..120) slice_root_sha256 [32]     (canonical manifest + payload)
//! [120..128) reserved 8 bytes
//! ```
//!
//! The payload region is 4 KiB aligned inside the file and zero-padded to
//! the next 4 KiB boundary; checksums cover only the logical bytes.

use har_model_compiler::{ModelPhenotype, ModelTensorDescriptor};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const SLICE_SCHEMA: &str = "har.package_slice.v1";
pub const SLICE_MAGIC: &[u8; 8] = b"HARSLC01";
pub const SLICE_FORMAT_VERSION: u32 = 1;
pub const SLICE_HEADER_BYTES: u64 = 128;
pub const SLICE_ALIGNMENT: u32 = 4096;
/// Repository-safe fixture budget for one committed slice payload.
pub const MAX_SLICE_FIXTURE_PAYLOAD_BYTES: u64 = 128 * 1024;
/// Repository-safe total budget for all committed model-derived fixtures.
pub const MAX_COMMITTED_MODEL_FIXTURE_BYTES: u64 = 384 * 1024;
/// GitHub hard per-file limit; the repository never tracks files at/above it.
pub const MAX_TRACKED_FILE_BYTES: u64 = 100 * 1024 * 1024;
/// Canonical MXFP4 block geometry (GGUF type id 39, `block_mxfp4`).
pub const MXFP4_BLOCK_ELEMENTS: u64 = 32;
pub const MXFP4_BLOCK_BYTES: u64 = 17;

#[derive(Debug)]
pub enum SliceError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Invalid(String),
}

impl fmt::Display for SliceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Json(error) => write!(f, "JSON error: {error}"),
            Self::Invalid(message) => f.write_str(message),
        }
    }
}
impl std::error::Error for SliceError {}
impl From<std::io::Error> for SliceError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}
impl From<serde_json::Error> for SliceError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}
impl From<har_model_compiler::CompilerError> for SliceError {
    fn from(value: har_model_compiler::CompilerError) -> Self {
        Self::Invalid(value.to_string())
    }
}
impl From<har_model_package::PackageError> for SliceError {
    fn from(value: har_model_package::PackageError) -> Self {
        Self::Invalid(value.to_string())
    }
}

pub type Result<T> = std::result::Result<T, SliceError>;

pub fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub fn align_up(value: u64, alignment: u64) -> u64 {
    assert!(alignment > 0);
    value.saturating_add(alignment - 1) / alignment * alignment
}

/// Source tensor identity and block geometry (canonical GGML layout).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceSlice {
    pub model_root: String,
    pub shard_basename: String,
    pub model_root_sha256: String,
    pub tensor_identity: String,
    pub ggml_type_id: u32,
    pub quant_format: String,
    pub tensor_dimensions: Vec<u64>,
    pub element_count: u64,
    pub row_elements: u64,
    pub block_elements: u64,
    pub block_bytes: u64,
    pub row_bytes: u64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct RowRange {
    pub start: u64,
    pub count: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SliceOffsets {
    /// Absolute file offset of the tensor payload start in the source shard.
    pub source_tensor_offset: u64,
    /// Absolute file offset of the first sliced row in the source shard.
    pub source_offset: u64,
    /// Exact logical bytes of the slice (row_count * row_bytes).
    pub logical_bytes: u64,
    /// Aligned physical window bytes (4096 alignment) covering the slice.
    pub physical_bytes: u64,
    pub leading_padding_bytes: u64,
    pub trailing_padding_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SlicePackage {
    pub package_schema: String,
    pub package_root_sha256: String,
    pub generation: u64,
    pub compiler_version: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SliceReconstruction {
    pub command: String,
    pub deterministic: bool,
    pub fixture_class: String,
    pub approval: String,
}

/// Canonical, deterministic manifest for one bounded package slice.
/// No timestamps, no absolute paths, no machine identity.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SliceManifest {
    pub schema: String,
    pub format_version: u32,
    pub slice_id: String,
    pub source: SourceSlice,
    pub row_range: RowRange,
    pub offsets: SliceOffsets,
    pub payload_checksum_sha256: String,
    /// SHA-256 of the identical byte range in the source shard.  Equal to
    /// `payload_checksum_sha256` because the payload is a verbatim copy;
    /// both are recorded so verification against the source stays explicit.
    pub source_span_checksum_sha256: String,
    pub package: SlicePackage,
    pub reconstruction: SliceReconstruction,
}

impl SliceManifest {
    /// Canonical serialization: compact JSON with recursively sorted keys.
    pub fn canonical_json(&self) -> Result<Vec<u8>> {
        let value = serde_json::to_value(self)?;
        Ok(serde_json::to_vec(&sort_json_value(value))?)
    }

    pub fn stable_hash(&self) -> Result<String> {
        Ok(sha256_bytes(&self.canonical_json()?))
    }
}

/// Fixed 64-byte slice header.
#[derive(Clone, Debug)]
struct SliceHeader {
    manifest_offset: u64,
    manifest_bytes: u64,
    payload_offset: u64,
    payload_bytes: u64,
    manifest_sha256: [u8; 32],
    slice_root_sha256: [u8; 32],
}

impl SliceHeader {
    fn encode(&self) -> [u8; SLICE_HEADER_BYTES as usize] {
        let mut bytes = [0u8; SLICE_HEADER_BYTES as usize];
        bytes[..8].copy_from_slice(SLICE_MAGIC);
        bytes[8..12].copy_from_slice(&SLICE_FORMAT_VERSION.to_le_bytes());
        bytes[12..16].copy_from_slice(&(SLICE_HEADER_BYTES as u32).to_le_bytes());
        bytes[16..20].copy_from_slice(&SLICE_ALIGNMENT.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.manifest_offset.to_le_bytes());
        bytes[32..40].copy_from_slice(&self.manifest_bytes.to_le_bytes());
        bytes[40..48].copy_from_slice(&self.payload_offset.to_le_bytes());
        bytes[48..56].copy_from_slice(&self.payload_bytes.to_le_bytes());
        bytes[56..88].copy_from_slice(&self.manifest_sha256);
        bytes[88..120].copy_from_slice(&self.slice_root_sha256);
        bytes
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < SLICE_HEADER_BYTES as usize || &bytes[..8] != SLICE_MAGIC {
            return Err(SliceError::Invalid("invalid HARSLC01 slice header".into()));
        }
        let u32_at = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
        if u32_at(8) != SLICE_FORMAT_VERSION
            || u32_at(12) != SLICE_HEADER_BYTES as u32
            || u32_at(16) != SLICE_ALIGNMENT
        {
            return Err(SliceError::Invalid(
                "unsupported slice header version/alignment".into(),
            ));
        }
        let mut manifest_sha256 = [0u8; 32];
        manifest_sha256.copy_from_slice(&bytes[56..88]);
        let mut slice_root_sha256 = [0u8; 32];
        slice_root_sha256.copy_from_slice(&bytes[88..120]);
        Ok(Self {
            manifest_offset: u64_at(24),
            manifest_bytes: u64_at(32),
            payload_offset: u64_at(40),
            payload_bytes: u64_at(48),
            manifest_sha256,
            slice_root_sha256,
        })
    }
}

/// Writer: extract a bounded row range from a source GGUF and serialize it.
pub struct SliceWriter;

impl SliceWriter {
    /// Validate slice geometry against a real tensor descriptor.  Fail-closed:
    /// the declared format, row size, and row range must match the source.
    #[allow(clippy::manual_is_multiple_of)]
    pub fn validate_geometry(
        descriptor: &ModelTensorDescriptor,
        row_elements: u64,
        block_elements: u64,
        block_bytes: u64,
        row_range: &RowRange,
    ) -> Result<(u64, u64, u64)> {
        if row_elements == 0 || descriptor.element_count == 0 {
            return Err(SliceError::Invalid(
                "slice geometry requires nonzero rows/elements".into(),
            ));
        }
        if row_elements % block_elements != 0 {
            return Err(SliceError::Invalid(format!(
                "row of {row_elements} elements is not a multiple of block size {block_elements}"
            )));
        }
        let row_bytes = (row_elements / block_elements) * block_bytes;
        if descriptor.dimensions.first().copied().unwrap_or(0) != row_elements {
            return Err(SliceError::Invalid(format!(
                "tensor {} row size {} does not match declared row_elements {row_elements}",
                descriptor.name,
                descriptor.dimensions.first().copied().unwrap_or(0)
            )));
        }
        let rows = descriptor.element_count / row_elements;
        if row_range.start >= rows
            || row_range.count == 0
            || row_range.start + row_range.count > rows
        {
            return Err(SliceError::Invalid(format!(
                "row range {}..{} is outside tensor row count {rows}",
                row_range.start,
                row_range.start + row_range.count
            )));
        }
        Ok((row_bytes, rows, row_bytes * row_range.count))
    }

    /// Write a slice file.  `payload` must be exactly the source bytes of the
    /// row range; `manifest` must carry matching checksums (the writer
    /// recomputes and overwrites them, so the caller only supplies identity).
    /// Payloads above the repository fixture budget are rejected: committed
    /// evidence must stay bounded (use `write_large` only for local-only
    /// hash evidence, never for committed fixtures).
    pub fn write(
        path: impl AsRef<Path>,
        manifest: SliceManifest,
        payload: &[u8],
    ) -> Result<SliceManifest> {
        if payload.len() as u64 > MAX_SLICE_FIXTURE_PAYLOAD_BYTES {
            return Err(SliceError::Invalid(format!(
                "slice payload {} exceeds the repository fixture budget of {MAX_SLICE_FIXTURE_PAYLOAD_BYTES} bytes",
                payload.len()
            )));
        }
        Self::write_unbounded(path, manifest, payload)
    }

    /// Local-only variant for hash evidence above the committed-fixture
    /// budget (e.g. a full 4,456,448-byte MXFP4 expert payload).  The file it
    /// produces is still deterministic and self-verifying; it must never be
    /// committed (CI rejects over-budget slices).
    pub fn write_large(
        path: impl AsRef<Path>,
        manifest: SliceManifest,
        payload: &[u8],
    ) -> Result<SliceManifest> {
        Self::write_unbounded(path, manifest, payload)
    }

    fn write_unbounded(
        path: impl AsRef<Path>,
        mut manifest: SliceManifest,
        payload: &[u8],
    ) -> Result<SliceManifest> {
        if payload.len() as u64 != manifest.offsets.logical_bytes {
            return Err(SliceError::Invalid(format!(
                "payload is {} bytes but manifest declares {}",
                payload.len(),
                manifest.offsets.logical_bytes
            )));
        }
        manifest.payload_checksum_sha256 = sha256_bytes(payload);
        manifest.source_span_checksum_sha256 = manifest.payload_checksum_sha256.clone();
        let manifest_blob = manifest.canonical_json()?;
        let manifest_sha256 = sha256_bytes(&manifest_blob);
        let payload_offset = align_up(
            SLICE_HEADER_BYTES + manifest_blob.len() as u64,
            SLICE_ALIGNMENT as u64,
        );
        let mut root_input = Vec::with_capacity(manifest_blob.len() + payload.len());
        root_input.extend_from_slice(&manifest_blob);
        root_input.extend_from_slice(payload);
        let slice_root = sha256_bytes(&root_input);
        let header = SliceHeader {
            manifest_offset: SLICE_HEADER_BYTES,
            manifest_bytes: manifest_blob.len() as u64,
            payload_offset,
            payload_bytes: payload.len() as u64,
            manifest_sha256: hex::decode(&manifest_sha256).unwrap().try_into().unwrap(),
            slice_root_sha256: hex::decode(&slice_root).unwrap().try_into().unwrap(),
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
        file.write_all(&manifest_blob)?;
        let after_manifest = SLICE_HEADER_BYTES + manifest_blob.len() as u64;
        let padding = payload_offset - after_manifest;
        file.write_all(&vec![0u8; padding as usize])?;
        file.write_all(payload)?;
        file.sync_all()?;
        Ok(manifest)
    }
}

/// A fully validated slice.
#[derive(Clone, Debug)]
pub struct VerifiedSlice {
    pub manifest: SliceManifest,
    pub manifest_sha256: String,
    pub slice_root_sha256: String,
    pub payload_offset: u64,
    pub payload_bytes: u64,
    pub path: PathBuf,
}

/// Reader: verify integrity and identity without ever loading the payload.
pub struct SliceReader;

impl SliceReader {
    /// Open and validate header, manifest checksum, and slice root checksum.
    /// The payload region is range-checked but not hashed here (bounded
    /// evidence may be hashed on demand via `verify_payload`).
    pub fn open(path: impl AsRef<Path>) -> Result<VerifiedSlice> {
        let path = path.as_ref();
        let file_bytes = fs::metadata(path)?.len();
        let mut file = File::open(path)?;
        let mut header_bytes = [0u8; SLICE_HEADER_BYTES as usize];
        file.read_exact(&mut header_bytes)?;
        let header = SliceHeader::decode(&header_bytes)?;
        let manifest_end = header.manifest_offset + header.manifest_bytes;
        let payload_end = header.payload_offset + header.payload_bytes;
        if manifest_end > file_bytes || payload_end > file_bytes {
            return Err(SliceError::Invalid(
                "slice ranges exceed file size (truncated)".into(),
            ));
        }
        if header.payload_offset % SLICE_ALIGNMENT as u64 != 0 {
            return Err(SliceError::Invalid(
                "slice payload is not 4 KiB aligned".into(),
            ));
        }
        let mut manifest_blob = vec![0u8; header.manifest_bytes as usize];
        file.seek(SeekFrom::Start(header.manifest_offset))?;
        file.read_exact(&mut manifest_blob)?;
        let manifest_hash = sha256_bytes(&manifest_blob);
        if hex::decode(&manifest_hash).unwrap().as_slice() != header.manifest_sha256.as_slice() {
            return Err(SliceError::Invalid(
                "slice manifest checksum mismatch".into(),
            ));
        }
        let manifest: SliceManifest = serde_json::from_slice(&manifest_blob)?;
        if manifest.schema != SLICE_SCHEMA || manifest.format_version != SLICE_FORMAT_VERSION {
            return Err(SliceError::Invalid(
                "slice manifest schema/version mismatch".into(),
            ));
        }
        if manifest.offsets.logical_bytes != header.payload_bytes {
            return Err(SliceError::Invalid(
                "slice manifest and header disagree on payload bytes".into(),
            ));
        }
        // Root checksum: canonical manifest + payload region (payload hashed).
        let mut payload = vec![0u8; header.payload_bytes as usize];
        file.seek(SeekFrom::Start(header.payload_offset))?;
        file.read_exact(&mut payload)?;
        let mut root_input = Vec::with_capacity(manifest_blob.len() + payload.len());
        root_input.extend_from_slice(&manifest_blob);
        root_input.extend_from_slice(&payload);
        let slice_root = sha256_bytes(&root_input);
        if hex::decode(&slice_root).unwrap().as_slice() != header.slice_root_sha256.as_slice() {
            return Err(SliceError::Invalid("slice root checksum mismatch".into()));
        }
        if sha256_bytes(&payload) != manifest.payload_checksum_sha256 {
            return Err(SliceError::Invalid(
                "slice payload checksum mismatch".into(),
            ));
        }
        Ok(VerifiedSlice {
            manifest,
            manifest_sha256: manifest_hash,
            slice_root_sha256: slice_root,
            payload_offset: header.payload_offset,
            payload_bytes: header.payload_bytes,
            path: path.to_path_buf(),
        })
    }

    /// Identity checks without touching the payload: model root, tensor
    /// identity, row range, generation, and (optionally) package root.
    pub fn validate_identity(
        slice: &VerifiedSlice,
        expected_model_root: Option<&str>,
        expected_generation: Option<u64>,
        expected_package_root: Option<&str>,
    ) -> Result<()> {
        if let Some(expected) = expected_model_root {
            if slice.manifest.source.model_root_sha256 != expected {
                return Err(SliceError::Invalid(format!(
                    "slice model root {} does not match expected {expected}",
                    slice.manifest.source.model_root_sha256
                )));
            }
        }
        if let Some(expected) = expected_generation {
            if slice.manifest.package.generation != expected {
                return Err(SliceError::Invalid(format!(
                    "slice generation {} is stale; expected {expected}",
                    slice.manifest.package.generation
                )));
            }
        }
        if let Some(expected) = expected_package_root {
            if slice.manifest.package.package_root_sha256 != expected {
                return Err(SliceError::Invalid(format!(
                    "slice package root {} does not match expected {expected}",
                    slice.manifest.package.package_root_sha256
                )));
            }
        }
        Ok(())
    }

    /// Read only the payload bytes (bounded, checksum-validated by `open`).
    pub fn read_payload(slice: &VerifiedSlice) -> Result<Vec<u8>> {
        let mut file = File::open(&slice.path)?;
        file.seek(SeekFrom::Start(slice.payload_offset))?;
        let mut payload = vec![0u8; slice.payload_bytes as usize];
        file.read_exact(&mut payload)?;
        Ok(payload)
    }
}

/// Recursively sort JSON object keys for canonical serialization.
fn sort_json_value(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut sorted = serde_json::Map::with_capacity(map.len());
            let mut keys: Vec<String> = map.keys().cloned().collect();
            keys.sort();
            for key in keys {
                let value = map.get(&key).cloned().unwrap_or(serde_json::Value::Null);
                sorted.insert(key, sort_json_value(value));
            }
            serde_json::Value::Object(sorted)
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.into_iter().map(sort_json_value).collect())
        }
        other => other,
    }
}

/// Convenience: build the source-slice identity from an inspected phenotype.
pub fn source_slice_from_phenotype(
    phenotype: &ModelPhenotype,
    descriptor: &ModelTensorDescriptor,
    row_elements: u64,
    block_elements: u64,
    block_bytes: u64,
) -> Result<SourceSlice> {
    let row_bytes = (row_elements / block_elements) * block_bytes;
    Ok(SourceSlice {
        model_root: phenotype.model_name.clone(),
        shard_basename: Path::new(&phenotype.path)
            .file_name()
            .and_then(|v| v.to_str())
            .unwrap_or("source.gguf")
            .to_owned(),
        model_root_sha256: phenotype.sha256.clone(),
        tensor_identity: descriptor.name.clone(),
        ggml_type_id: descriptor.ggml_type,
        quant_format: descriptor.quantization.clone(),
        tensor_dimensions: descriptor.dimensions.clone(),
        element_count: descriptor.element_count,
        row_elements,
        block_elements,
        block_bytes,
        row_bytes,
    })
}

#[cfg(test)]
mod tests;
