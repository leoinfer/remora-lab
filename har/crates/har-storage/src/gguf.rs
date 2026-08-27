//! Dependency-light split GGUF expert directory reader.

use crate::direct_io::{DirectIoEngine, ReadRequest};
use har_residency::{
    ExpertProjection, ModelRoot, PageId, PageKind, PageSource, ResidencyError, Result, StorageSlice,
};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

#[derive(Clone, Debug)]
struct TensorInfo {
    name: String,
    dimensions: Vec<u64>,
    type_name: String,
    relative_offset: u64,
    payload_bytes: Option<u64>,
}

#[derive(Clone, Debug)]
struct ShardHeader {
    path: PathBuf,
    version: u32,
    metadata: Metadata,
    alignment: u64,
    data_start: u64,
    tensors: Vec<TensorInfo>,
}

#[derive(Clone, Debug, Default)]
struct Metadata {
    model_name: Option<String>,
    expert_count: u32,
    block_count: u32,
    split_no: u32,
}

fn read_exact(file: &mut File, bytes: usize) -> Result<Vec<u8>> {
    let mut value = vec![0u8; bytes];
    file.read_exact(&mut value)
        .map_err(|error| ResidencyError::Io(error.to_string()))?;
    Ok(value)
}
fn u32(file: &mut File) -> Result<u32> {
    Ok(u32::from_le_bytes(read_exact(file, 4)?.try_into().unwrap()))
}
fn u64(file: &mut File) -> Result<u64> {
    Ok(u64::from_le_bytes(read_exact(file, 8)?.try_into().unwrap()))
}
fn string(file: &mut File) -> Result<String> {
    let length = u64(file)? as usize;
    if length > 1 << 30 {
        return Err(ResidencyError::Invalid("GGUF string too large".into()));
    }
    Ok(String::from_utf8_lossy(&read_exact(file, length)?).into_owned())
}
fn align_up(value: u64, alignment: u64) -> u64 {
    value.saturating_add(alignment - 1) / alignment * alignment
}

fn skip_value(file: &mut File, value_type: u32) -> Result<Option<String>> {
    match value_type {
        0 | 1 | 7 => {
            file.seek(SeekFrom::Current(1))
                .map_err(|error| ResidencyError::Io(error.to_string()))?;
            Ok(None)
        }
        2 | 3 => {
            file.seek(SeekFrom::Current(2))
                .map_err(|error| ResidencyError::Io(error.to_string()))?;
            Ok(None)
        }
        4..=6 => {
            file.seek(SeekFrom::Current(4))
                .map_err(|error| ResidencyError::Io(error.to_string()))?;
            Ok(None)
        }
        8 => Ok(Some(string(file)?)),
        9 => {
            let element_type = u32(file)?;
            let count = u64(file)?;
            if count > 100_000_000 {
                return Err(ResidencyError::Invalid(
                    "GGUF metadata array too large".into(),
                ));
            }
            for _ in 0..count {
                let _ = skip_value(file, element_type)?;
            }
            Ok(None)
        }
        10 | 11 => {
            file.seek(SeekFrom::Current(8))
                .map_err(|error| ResidencyError::Io(error.to_string()))?;
            Ok(None)
        }
        12 => {
            file.seek(SeekFrom::Current(8))
                .map_err(|error| ResidencyError::Io(error.to_string()))?;
            Ok(None)
        }
        _ => Err(ResidencyError::Unsupported(format!(
            "GGUF metadata type {value_type}"
        ))),
    }
}

fn parse_shard(path: &Path) -> Result<ShardHeader> {
    let mut file = File::open(path).map_err(|error| ResidencyError::Io(error.to_string()))?;
    if read_exact(&mut file, 4)?.as_slice() != GGUF_MAGIC {
        return Err(ResidencyError::Invalid(format!(
            "not GGUF: {}",
            path.display()
        )));
    }
    let version = u32(&mut file)?;
    if !(1..=3).contains(&version) {
        return Err(ResidencyError::Unsupported(format!(
            "GGUF version {version}"
        )));
    }
    let tensor_count = u64(&mut file)?;
    let metadata_count = u64(&mut file)?;
    let mut metadata = Metadata::default();
    let mut alignment = 32u64;
    for _ in 0..metadata_count {
        let key = string(&mut file)?;
        let value_type = u32(&mut file)?;
        let value = skip_value(&mut file, value_type)?;
        match key.as_str() {
            "general.name" | "general.basename" => {
                if let Some(value) = value {
                    metadata.model_name = Some(value);
                }
            }
            "deepseek4.expert_count" | "llama.expert_count" if value_type == 4 => {
                file.seek(SeekFrom::Current(-4))
                    .map_err(|error| ResidencyError::Io(error.to_string()))?;
                metadata.expert_count = u32(&mut file)?;
            }
            "deepseek4.block_count" if value_type == 4 => {
                file.seek(SeekFrom::Current(-4))
                    .map_err(|error| ResidencyError::Io(error.to_string()))?;
                metadata.block_count = u32(&mut file)?;
            }
            "general.alignment" => {
                if value_type == 10 {
                    file.seek(SeekFrom::Current(-8))
                        .map_err(|error| ResidencyError::Io(error.to_string()))?;
                    alignment = u64(&mut file)?;
                } else if value_type == 4 {
                    file.seek(SeekFrom::Current(-4))
                        .map_err(|error| ResidencyError::Io(error.to_string()))?;
                    alignment = u32(&mut file)? as u64;
                }
            }
            "split.no" if value_type == 4 => {
                file.seek(SeekFrom::Current(-4))
                    .map_err(|error| ResidencyError::Io(error.to_string()))?;
                metadata.split_no = u32(&mut file)?;
            }
            _ => {}
        }
    }
    if alignment == 0 || alignment > 1 << 20 {
        return Err(ResidencyError::Invalid("GGUF alignment is invalid".into()));
    }
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = string(&mut file)?;
        let dimensions_count = u32(&mut file)?;
        if dimensions_count > 16 {
            return Err(ResidencyError::Unsupported("GGUF tensor rank".into()));
        }
        let mut dimensions = Vec::with_capacity(dimensions_count as usize);
        for _ in 0..dimensions_count {
            dimensions.push(u64(&mut file)?);
        }
        let type_id = u32(&mut file)?;
        let relative_offset = u64(&mut file)?;
        let (type_name, block_elements, block_bytes) = type_info(type_id);
        let elements = dimensions.iter().product::<u64>();
        let payload_bytes = block_elements.map(|blocks| elements.div_ceil(blocks) * block_bytes);
        tensors.push(TensorInfo {
            name,
            dimensions,
            type_name,
            relative_offset,
            payload_bytes,
        });
    }
    let data_start = align_up(
        file.stream_position()
            .map_err(|error| ResidencyError::Io(error.to_string()))?,
        alignment,
    );
    Ok(ShardHeader {
        path: path.to_path_buf(),
        version,
        metadata,
        alignment,
        data_start,
        tensors,
    })
}

fn type_info(type_id: u32) -> (String, Option<u64>, u64) {
    let (name, blocks, bytes) = match type_id {
        0 => ("F32", 1, 4),
        1 => ("F16", 1, 2),
        2 => ("Q4_0", 32, 18),
        3 => ("Q4_1", 32, 20),
        6 => ("Q5_0", 32, 22),
        7 => ("Q5_1", 32, 24),
        8 => ("Q8_0", 32, 34),
        9 => ("Q8_1", 32, 36),
        10 => ("Q2_K", 256, 84),
        11 => ("Q3_K", 256, 110),
        12 => ("Q4_K", 256, 144),
        13 => ("Q5_K", 256, 176),
        14 => ("Q6_K", 256, 210),
        15 => ("Q8_K", 256, 292),
        30 => ("BF16", 1, 2),
        39 => ("MXFP4", 32, 17),
        40 => ("NVFP4", 64, 36),
        _ => ("UNKNOWN", 0, 0),
    };
    (
        name.into(),
        if blocks == 0 { None } else { Some(blocks) },
        bytes,
    )
}

fn split_paths(path: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let path = path.as_ref();
    if path.is_dir() {
        let mut paths: Vec<_> = std::fs::read_dir(path)
            .map_err(|error| ResidencyError::Io(error.to_string()))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("gguf"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(ResidencyError::Invalid("no GGUF shards".into()));
        }
        return Ok(paths);
    }
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    if let Some(marker) = name.rfind("-of-") {
        if name.ends_with(".gguf") {
            let count = &name[marker + 4..name.len() - 5];
            let part_start = marker.saturating_sub(5);
            let prefix = &name[..part_start];
            let mut paths = Vec::new();
            for entry in std::fs::read_dir(path.parent().unwrap_or_else(|| Path::new(".")))
                .map_err(|error| ResidencyError::Io(error.to_string()))?
            {
                let candidate = entry
                    .map_err(|error| ResidencyError::Io(error.to_string()))?
                    .path();
                let candidate_name = candidate
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or_default();
                if candidate_name.starts_with(prefix)
                    && candidate_name.ends_with(&format!("-of-{count}.gguf"))
                {
                    paths.push(candidate);
                }
            }
            paths.sort();
            if !paths.is_empty() {
                return Ok(paths);
            }
        }
    }
    Ok(vec![path.to_path_buf()])
}

#[derive(Clone, Debug)]
pub struct ExpertIndex {
    pub model_id: String,
    pub model_root: ModelRoot,
    pub shards: Vec<PathBuf>,
    pub entries: Vec<StorageSlice>,
    pub block_count: u32,
    pub expert_count: u32,
    pub gguf_version: u32,
}

impl ExpertIndex {
    pub fn from_gguf(path: impl AsRef<Path>) -> Result<Self> {
        let shards = split_paths(path)?;
        let parsed: Vec<_> = shards
            .iter()
            .map(|path| parse_shard(path))
            .collect::<Result<_>>()?;
        let model_id = parsed
            .iter()
            .find_map(|shard| shard.metadata.model_name.clone())
            .unwrap_or_else(|| {
                shards[0]
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            });
        let model_root = ModelRoot::new(model_id.clone());
        let expert_count = parsed
            .iter()
            .find_map(|shard| {
                (shard.metadata.expert_count > 0).then_some(shard.metadata.expert_count)
            })
            .unwrap_or(0);
        let block_count = parsed
            .iter()
            .find_map(|shard| {
                (shard.metadata.block_count > 0).then_some(shard.metadata.block_count)
            })
            .unwrap_or(0);
        let mut entries = Vec::new();
        for (shard_index, shard) in parsed.iter().enumerate() {
            let mut ordered = shard.tensors.clone();
            ordered.sort_by_key(|tensor| tensor.relative_offset);
            for index in 0..ordered.len() {
                let tensor = &ordered[index];
                let projection = projection_for(&tensor.name);
                let layer = layer_for(&tensor.name);
                if projection.is_none() || layer.is_none() {
                    continue;
                }
                let projection = projection.unwrap();
                let layer = layer.unwrap();
                let experts = if expert_count > 0 {
                    expert_count
                } else {
                    *tensor.dimensions.last().unwrap_or(&0) as u32
                };
                if experts == 0 || tensor.dimensions.last().copied() != Some(experts as u64) {
                    continue;
                }
                let parent_bytes = tensor.payload_bytes.or_else(|| {
                    ordered
                        .get(index + 1)
                        .map(|next| next.relative_offset.saturating_sub(tensor.relative_offset))
                });
                let parent_bytes = match parent_bytes {
                    Some(value) if value > 0 && value % experts as u64 == 0 => value,
                    _ => continue,
                };
                let expert_bytes = parent_bytes / experts as u64;
                let parent_offset = shard.data_start + tensor.relative_offset;
                for expert in 0..experts {
                    let projection_index = match projection {
                        ExpertProjection::Gate => 0,
                        ExpertProjection::Up => 1,
                        ExpertProjection::Down => 2,
                        ExpertProjection::Other(_) => 3,
                    };
                    let ordinal =
                        (layer as u64) * 1_000_000 + (expert as u64) * 4 + projection_index;
                    entries.push(StorageSlice {
                        page_id: PageId {
                            model_root: model_root.clone(),
                            kind: PageKind::Weights,
                            ordinal,
                        },
                        model_id: model_id.clone(),
                        source_path: shard.path.to_string_lossy().into_owned(),
                        shard: shard_index.to_string(),
                        tensor: tensor.name.clone(),
                        offset: parent_offset + expert as u64 * expert_bytes,
                        payload_bytes: expert_bytes,
                        alignment: shard.alignment,
                        quant_type: tensor.type_name.clone(),
                        layer: Some(layer),
                        expert: Some(expert),
                        projection: projection.clone(),
                        checksum_sha256: None,
                        parent_offset: Some(parent_offset),
                        parent_payload_bytes: Some(parent_bytes),
                    });
                }
            }
        }
        Ok(Self {
            model_id,
            model_root,
            shards,
            entries,
            block_count,
            expert_count,
            gguf_version: parsed.first().map(|shard| shard.version).unwrap_or(0),
        })
    }

    pub fn lookup(
        &self,
        layer: u32,
        expert: u32,
        projection: ExpertProjection,
    ) -> Option<&StorageSlice> {
        self.entries.iter().find(|entry| {
            entry.layer == Some(layer)
                && entry.expert == Some(expert)
                && entry.projection == projection
        })
    }

    pub fn by_expert(&self, layer: u32, expert: u32) -> Vec<&StorageSlice> {
        self.entries
            .iter()
            .filter(|entry| entry.layer == Some(layer) && entry.expert == Some(expert))
            .collect()
    }

    pub fn checksum(&self, slice: &StorageSlice, io: &DirectIoEngine) -> Result<String> {
        let result = io.read(ReadRequest {
            path: PathBuf::from(&slice.source_path),
            offset: slice.offset,
            bytes: slice.payload_bytes,
            alignment: io.alignment(),
            tensor_id: format!(
                "l{}e{}{}",
                slice.layer.unwrap_or(0),
                slice.expert.unwrap_or(0),
                slice.projection.as_str()
            ),
            mandatory: true,
            speculative: false,
        })?;
        let mut hasher = Sha256::new();
        hasher.update(result.data);
        Ok(format!("{:x}", hasher.finalize()))
    }
}

fn projection_for(name: &str) -> Option<ExpertProjection> {
    if name.contains("ffn_gate_exps") {
        Some(ExpertProjection::Gate)
    } else if name.contains("ffn_up_exps") {
        Some(ExpertProjection::Up)
    } else if name.contains("ffn_down_exps") {
        Some(ExpertProjection::Down)
    } else {
        None
    }
}
fn layer_for(name: &str) -> Option<u32> {
    let marker = name.find("blk.")? + 4;
    let digits: String = name[marker..]
        .chars()
        .take_while(|character| character.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

#[derive(Clone, Debug)]
pub struct OriginalGgufStore {
    pub io: Arc<DirectIoEngine>,
}
impl OriginalGgufStore {
    pub fn new(io: Arc<DirectIoEngine>) -> Self {
        Self { io }
    }
}
impl PageSource for OriginalGgufStore {
    fn read_slice(&self, slice: &StorageSlice) -> Result<Vec<u8>> {
        let result = self.io.read(ReadRequest {
            path: PathBuf::from(&slice.source_path),
            offset: slice.offset,
            bytes: slice.payload_bytes,
            alignment: self.io.alignment(),
            tensor_id: slice.tensor.to_string(),
            mandatory: true,
            speculative: false,
        })?;
        if let Some(expected) = &slice.checksum_sha256 {
            let mut hasher = Sha256::new();
            hasher.update(&result.data);
            if format!("{:x}", hasher.finalize()) != *expected {
                return Err(ResidencyError::Io("expert slice checksum mismatch".into()));
            }
        }
        Ok(result.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn projection_and_layer_parsing() {
        assert_eq!(
            projection_for("blk.7.ffn_gate_exps.weight"),
            Some(ExpertProjection::Gate)
        );
        assert_eq!(layer_for("blk.7.ffn_gate_exps.weight"), Some(7));
    }
}
