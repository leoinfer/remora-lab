//! Reader for package builder's `har.expert_sidecar.v1` aligned container.

use crate::direct_io::{DirectIoEngine, ReadRequest};
use har_residency::{
    ExpertProjection, ModelRoot, PageId, PageKind, PageSource, ResidencyError, Result, StorageSlice,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const HEADER_BYTES: usize = 4096;
const MAGIC: &[u8; 8] = b"HARSIDE1";

#[derive(Clone, Debug, Deserialize)]
struct SidecarEntry {
    source_tensor_id: String,
    layer: u32,
    #[serde(alias = "expert")]
    expert_id: u32,
    projection: String,
    #[serde(alias = "quant_type")]
    quant_format: String,
    offset: u64,
    payload_bytes: u64,
    alignment: u64,
    checksum_sha256: String,
    #[serde(default)]
    native_payload: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct SidecarIndex {
    #[serde(default)]
    model_identity: String,
    #[serde(default)]
    source_model_sha256: Option<String>,
    entries: Vec<SidecarEntry>,
}

#[derive(Clone, Debug)]
pub struct AlignedSidecarStore {
    pub path: PathBuf,
    pub model_id: String,
    pub entries: Vec<StorageSlice>,
    pub index_sha256: String,
    pub payload_offset: u64,
    pub source_model_sha256: Option<String>,
    pub native_payload_entries: usize,
    pub io: Arc<DirectIoEngine>,
}

impl AlignedSidecarStore {
    pub fn verify_source_model_sha256(&self, expected: &str) -> Result<()> {
        match &self.source_model_sha256 {
            Some(actual) if actual.eq_ignore_ascii_case(expected) => Ok(()),
            Some(actual) => Err(ResidencyError::Invalid(format!(
                "sidecar source model checksum mismatch: {actual} != {expected}"
            ))),
            None => Err(ResidencyError::Invalid(
                "sidecar does not declare source_model_sha256".into(),
            )),
        }
    }

    pub fn open(path: impl AsRef<Path>, io: Arc<DirectIoEngine>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = File::open(&path).map_err(|error| ResidencyError::Io(error.to_string()))?;
        let header = read_header(&mut file)?;
        let index_offset = u64::from_le_bytes(header[24..32].try_into().unwrap());
        let index_bytes = u64::from_le_bytes(header[32..40].try_into().unwrap());
        let payload_offset = u64::from_le_bytes(header[40..48].try_into().unwrap());
        if index_offset < HEADER_BYTES as u64 || index_bytes == 0 {
            return Err(ResidencyError::Invalid(
                "sidecar index header is invalid".into(),
            ));
        }
        file.seek(SeekFrom::Start(index_offset))
            .map_err(|error| ResidencyError::Io(error.to_string()))?;
        let mut index_data = vec![0u8; index_bytes as usize];
        file.read_exact(&mut index_data)
            .map_err(|error| ResidencyError::Io(error.to_string()))?;
        let mut index_hash = Sha256::new();
        index_hash.update(&index_data);
        let index_sha256 = format!("{:x}", index_hash.finalize());
        let header_hash = header[56..88]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        if header_hash.chars().any(|character| character != '0') && header_hash != index_sha256 {
            return Err(ResidencyError::Invalid(
                "sidecar canonical index checksum mismatch".into(),
            ));
        }
        let document: SidecarIndex = serde_json::from_slice(&index_data)
            .map_err(|error| ResidencyError::Invalid(format!("sidecar index JSON: {error}")))?;
        let source_model_sha256 = document.source_model_sha256.clone();
        let native_payload_entries = document
            .entries
            .iter()
            .filter(|entry| entry.native_payload)
            .count();
        let model_id = if document.model_identity.is_empty() {
            "sidecar-model".into()
        } else {
            document.model_identity.clone()
        };
        let root = ModelRoot::new(model_id.clone());
        let entries = document
            .entries
            .into_iter()
            .map(|entry| {
                if entry.alignment == 0
                    || entry.offset < payload_offset
                    || entry.offset % entry.alignment != 0
                    || entry.checksum_sha256.len() != 64
                {
                    return Err(ResidencyError::Invalid(
                        "sidecar entry alignment, payload region, or checksum is invalid".into(),
                    ));
                }
                let projection = match entry.projection.as_str() {
                    "gate" => ExpertProjection::Gate,
                    "up" => ExpertProjection::Up,
                    "down" => ExpertProjection::Down,
                    other => ExpertProjection::Other(other.into()),
                };
                let projection_index = match projection {
                    ExpertProjection::Gate => 0,
                    ExpertProjection::Up => 1,
                    ExpertProjection::Down => 2,
                    ExpertProjection::Other(_) => 3,
                };
                Ok(StorageSlice {
                    page_id: PageId {
                        model_root: root.clone(),
                        kind: PageKind::Weights,
                        ordinal: entry.layer as u64 * 1_000_000
                            + entry.expert_id as u64 * 4
                            + projection_index,
                    },
                    model_id: model_id.clone(),
                    source_path: path.to_string_lossy().into_owned(),
                    shard: "sidecar".into(),
                    tensor: entry.source_tensor_id,
                    offset: entry.offset,
                    payload_bytes: entry.payload_bytes,
                    alignment: entry.alignment,
                    quant_type: entry.quant_format,
                    layer: Some(entry.layer),
                    expert: Some(entry.expert_id),
                    projection,
                    checksum_sha256: Some(entry.checksum_sha256),
                    parent_offset: None,
                    parent_payload_bytes: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            path,
            model_id,
            entries,
            index_sha256,
            payload_offset,
            source_model_sha256,
            native_payload_entries,
            io,
        })
    }
}

fn read_header(file: &mut File) -> Result<Vec<u8>> {
    let mut header = vec![0u8; HEADER_BYTES];
    file.read_exact(&mut header)
        .map_err(|error| ResidencyError::Io(error.to_string()))?;
    if &header[..8] != MAGIC {
        return Err(ResidencyError::Invalid("not a HARSIDE1 sidecar".into()));
    }
    if u32::from_le_bytes(header[8..12].try_into().unwrap()) != 1 {
        return Err(ResidencyError::Unsupported("sidecar version".into()));
    }
    if u32::from_le_bytes(header[12..16].try_into().unwrap()) != HEADER_BYTES as u32 {
        return Err(ResidencyError::Invalid("sidecar header size".into()));
    }
    if u32::from_le_bytes(header[16..20].try_into().unwrap()) != 4096 {
        return Err(ResidencyError::Invalid("sidecar alignment".into()));
    }
    Ok(header)
}

impl PageSource for AlignedSidecarStore {
    fn read_slice(&self, slice: &StorageSlice) -> Result<Vec<u8>> {
        let result = self.io.read(ReadRequest {
            path: self.path.clone(),
            offset: slice.offset,
            bytes: slice.payload_bytes,
            alignment: slice.alignment,
            tensor_id: slice.tensor.clone(),
            mandatory: true,
            speculative: false,
        })?;
        if let Some(expected) = &slice.checksum_sha256 {
            let mut hash = Sha256::new();
            hash.update(&result.data);
            if format!("{:x}", hash.finalize()) != *expected {
                return Err(ResidencyError::Io(
                    "sidecar payload checksum mismatch".into(),
                ));
            }
        }
        Ok(result.data)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sidecar_magic_is_versioned() {
        assert_eq!(super::MAGIC, b"HARSIDE1");
    }
}
