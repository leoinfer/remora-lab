//! Model inspection and GGUF source-span access.
//!
//! The loader reads the GGUF header and tensor directory with native Rust.
//! Payloads remain owned by the source file until a selected native backend
//! explicitly copies or maps a bounded span.

use har_core::{canonical_sha256, sha256_file, HarError, QuantFormat, Result, TensorRole};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub const MODEL_INTERFACE: &str = "har.model_phenotype.v1";
pub const TENSOR_INTERFACE: &str = "har.tensor_descriptor.v1";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorDescriptor {
    pub interface: String,
    pub ordinal: u64,
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub quantization: QuantFormat,
    pub element_count: u64,
    pub payload_bytes: u64,
    pub raw_span_bytes: u64,
    pub file_offset: u64,
    pub alignment_bytes: u64,
    pub layer: Option<u32>,
    pub role: TensorRole,
    pub hotness: u32,
    pub is_weight: bool,
    pub is_mtp: bool,
}
impl TensorDescriptor {
    pub fn elements(&self) -> u64 {
        self.element_count
    }
    pub fn is_quantized(&self) -> bool {
        self.quantization.is_quantized()
    }
    pub fn row_elements(&self) -> Option<u64> {
        self.dimensions.first().copied()
    }
    pub fn rows(&self) -> Option<u64> {
        self.dimensions.get(1).copied()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelPhenotype {
    pub interface: String,
    pub path: String,
    pub sha256: Option<String>,
    pub file_bytes: u64,
    pub gguf_version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
    pub data_offset: u64,
    pub tensor_payload_bytes: u64,
    pub tensor_padding_bytes: u64,
    pub architecture: String,
    pub model_name: String,
    pub block_count: u32,
    pub embedding_length: u32,
    pub attention_heads: u32,
    pub kv_heads: u32,
    pub key_length: u32,
    pub value_length: u32,
    pub nextn_predict_layers: u32,
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub kv_geometry: String,
    pub metadata_summary: BTreeMap<String, String>,
    pub quantization_bytes: BTreeMap<String, u64>,
    pub quantization_tensor_counts: BTreeMap<String, u64>,
    pub tensors: Vec<TensorDescriptor>,
    pub tokenizer: Option<GgufTokenizer>,
}
impl ModelPhenotype {
    pub fn tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.iter().find(|item| item.name == name)
    }
    pub fn identity_hash(&self) -> Result<String> {
        canonical_sha256(self)
    }
    pub fn source_identity(&self) -> &str {
        self.sha256.as_deref().unwrap_or("")
    }
}

/// The GGUF tokenizer metadata (captured for the native tokenizer).
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct GgufTokenizer {
    pub model: Option<String>,
    pub tokens: Vec<String>,
    pub merges: Vec<String>,
    pub scores: Vec<f32>,
    pub token_types: Vec<u32>,
    pub added_tokens: Vec<String>,
    pub eos_token_id: Option<u32>,
    pub bos_token_id: Option<u32>,
    pub chat_template: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
    pub data_offset: u64,
    pub alignment: u64,
}

pub struct GgufReader {
    path: PathBuf,
}
impl GgufReader {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn inspect(&self, with_hash: bool) -> Result<ModelPhenotype> {
        let file_bytes = std::fs::metadata(&self.path)?.len();
        let file = File::open(&self.path)?;
        let mut input = BufReader::new(file);
        let (header, metadata, raw, tokenizer) = read_directory(&mut input, file_bytes)?;
        let sha256 = if with_hash {
            Some(sha256_file(&self.path)?)
        } else {
            None
        };
        let mut descriptors = Vec::with_capacity(raw.len());
        let mut quantization_bytes: BTreeMap<String, u64> = BTreeMap::new();
        let mut quantization_tensor_counts: BTreeMap<String, u64> = BTreeMap::new();
        let mut raw_by_offset: Vec<(u64, usize)> = raw
            .iter()
            .enumerate()
            .map(|(i, item)| (item.offset, i))
            .collect();
        raw_by_offset.sort_unstable();
        let mut next_span = vec![None; raw.len()];
        for pair in raw_by_offset.windows(2) {
            next_span[pair[0].1] = Some(pair[1].0.saturating_sub(pair[0].0));
        }
        for (index, item) in raw.iter().enumerate() {
            let element_count = item.dimensions.iter().product::<u64>();
            let known = tensor_payload_bytes(item.ggml_type, element_count);
            let span = next_span[index].unwrap_or_else(|| {
                file_bytes.saturating_sub(header.data_offset.saturating_add(item.offset))
            });
            let payload = if known != 0 { known } else { span };
            let layer = parse_layer(&item.name);
            let role = classify_role(&item.name);
            let quantization = QuantFormat::from_ggml_type(item.ggml_type);
            *quantization_bytes
                .entry(quantization.as_str().to_string())
                .or_default() += payload;
            *quantization_tensor_counts
                .entry(quantization.as_str().to_string())
                .or_default() += 1;
            descriptors.push(TensorDescriptor {
                interface: TENSOR_INTERFACE.into(),
                ordinal: index as u64,
                name: item.name.clone(),
                dimensions: item.dimensions.clone(),
                ggml_type: item.ggml_type,
                quantization,
                element_count,
                payload_bytes: payload,
                raw_span_bytes: span.max(payload),
                file_offset: header.data_offset + item.offset,
                alignment_bytes: header.alignment,
                layer,
                role,
                hotness: if item.name.contains("router") { 100 } else { 1 },
                is_weight: true,
                is_mtp: item.name.to_ascii_lowercase().contains("mtp")
                    || item.name.to_ascii_lowercase().contains("nextn"),
            });
        }
        let architecture = metadata
            .get("general.architecture")
            .cloned()
            .unwrap_or_else(|| "unknown".into());
        let model_name = metadata.get("general.name").cloned().unwrap_or_else(|| {
            self.path
                .file_stem()
                .and_then(|x| x.to_str())
                .unwrap_or("model")
                .into()
        });
        let block_count = metadata_number(&metadata, &format!("{architecture}.block_count"))
            .or_else(|| metadata_number(&metadata, "block_count"))
            .unwrap_or(0) as u32;
        let embedding_length =
            metadata_number(&metadata, &format!("{architecture}.embedding_length")).unwrap_or(0)
                as u32;
        let attention_heads = metadata_number(
            &metadata,
            &format!("{architecture}.attention.head_count"),
        )
        .unwrap_or(0) as u32;
        let kv_heads = metadata_number(
            &metadata,
            &format!("{architecture}.attention.head_count_kv"),
        )
        .unwrap_or(0) as u32;
        let key_length = metadata_number(&metadata, &format!("{architecture}.attention.key_length"))
            .unwrap_or(0) as u32;
        let value_length =
            metadata_number(&metadata, &format!("{architecture}.attention.value_length"))
                .unwrap_or(0) as u32;
        let nextn_predict_layers = metadata_number(
            &metadata,
            &format!("{architecture}.nextn_predict_layers"),
        )
        .unwrap_or(0) as u32;
        let expert_count =
            metadata_number(&metadata, &format!("{architecture}.expert_count")).unwrap_or(0) as u32;
        let expert_used_count =
            metadata_number(&metadata, &format!("{architecture}.expert_used_count")).unwrap_or(0)
                as u32;
        let tensor_payload_bytes = descriptors.iter().map(|x| x.payload_bytes).sum();
        let tokenizer: Option<GgufTokenizer> =
            (!tokenizer.tokens.is_empty() || tokenizer.model.is_some()).then_some(tokenizer);
        Ok(ModelPhenotype {
            interface: MODEL_INTERFACE.into(),
            path: self.path.display().to_string(),
            sha256,
            file_bytes,
            gguf_version: header.version,
            tensor_count: header.tensor_count,
            metadata_count: header.metadata_count,
            data_offset: header.data_offset,
            tensor_payload_bytes,
            tensor_padding_bytes: file_bytes
                .saturating_sub(header.data_offset)
                .saturating_sub(tensor_payload_bytes),
            architecture,
            model_name,
            block_count,
            embedding_length,
            attention_heads,
            kv_heads,
            key_length,
            value_length,
            nextn_predict_layers,
            expert_count,
            expert_used_count,
            kv_geometry: String::new(),
            metadata_summary: metadata,
            quantization_bytes,
            quantization_tensor_counts,
            tensors: descriptors,
            tokenizer,
        })
    }

    pub fn read_tensor_range(
        &self,
        tensor: &TensorDescriptor,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>> {
        if offset.checked_add(length).is_none() || offset + length > tensor.payload_bytes {
            return Err(HarError::Invalid {
                kind: "tensor range",
                message: format!("range {offset}+{length} exceeds {}", tensor.payload_bytes),
            });
        }
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(tensor.file_offset + offset))?;
        let mut bytes = vec![0u8; length as usize];
        file.read_exact(&mut bytes)?;
        Ok(bytes)
    }
    pub fn read_tensor(&self, tensor: &TensorDescriptor) -> Result<Vec<u8>> {
        self.read_tensor_range(tensor, 0, tensor.payload_bytes)
    }

    pub fn tensor_type_name(type_id: u32) -> &'static str {
        tensor_type_name(type_id)
    }
    pub fn tensor_block_size(type_id: u32) -> u64 {
        tensor_block_size(type_id)
    }
    pub fn tensor_type_size(type_id: u32) -> u64 {
        tensor_type_size(type_id)
    }
    pub fn tensor_payload_bytes(type_id: u32, elements: u64) -> u64 {
        tensor_payload_bytes(type_id, elements)
    }
}

#[derive(Clone, Debug)]
struct RawTensor {
    name: String,
    dimensions: Vec<u64>,
    ggml_type: u32,
    offset: u64,
}

#[allow(clippy::type_complexity)]
fn read_directory(
    input: &mut BufReader<File>,
    file_bytes: u64,
) -> Result<(
    GgufHeader,
    BTreeMap<String, String>,
    Vec<RawTensor>,
    GgufTokenizer,
)> {
    let magic = read_exact(input, 4)?;
    if magic.as_slice() != b"GGUF" {
        return Err(HarError::Invalid {
            kind: "GGUF",
            message: "bad magic".into(),
        });
    }
    let version = read_u32(input)?;
    if !(1..=3).contains(&version) {
        return Err(HarError::Unsupported {
            kind: "GGUF version",
            message: version.to_string(),
        });
    }
    let tensor_count = read_u64(input)?;
    let metadata_count = read_u64(input)?;
    if tensor_count > 10_000_000 || metadata_count > 10_000_000 {
        return Err(HarError::Invalid {
            kind: "GGUF counts",
            message: "count exceeds safety bound".into(),
        });
    }
    let mut metadata = BTreeMap::new();
    let mut tokenizer = GgufTokenizer::default();
    for _ in 0..metadata_count {
        let key = read_string(input)?;
        let value_type = read_u32(input)?;
        if key.starts_with("tokenizer.ggml.") {
            // capture_tokenizer consumes the value itself (all branches).
            capture_tokenizer(input, value_type, &key, &mut tokenizer)?;
        } else {
            let value = read_metadata(input, value_type)?;
            metadata.insert(key, value);
        }
    }
    let alignment = metadata_number(&metadata, "general.alignment")
        .unwrap_or(32)
        .max(1);
    let mut raw = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_string(input)?;
        let rank = read_u32(input)?;
        if rank > 16 {
            return Err(HarError::Invalid {
                kind: "GGUF rank",
                message: rank.to_string(),
            });
        }
        let mut dimensions = Vec::with_capacity(rank as usize);
        for _ in 0..rank {
            dimensions.push(read_u64(input)?);
        }
        let ggml_type = read_u32(input)?;
        let offset = read_u64(input)?;
        raw.push(RawTensor {
            name,
            dimensions,
            ggml_type,
            offset,
        });
    }
    let position = input.stream_position()?;
    let data_offset = align(position, alignment);
    if data_offset > file_bytes {
        return Err(HarError::Invalid {
            kind: "GGUF data offset",
            message: data_offset.to_string(),
        });
    }
    Ok((
        GgufHeader {
            version,
            tensor_count,
            metadata_count,
            data_offset,
            alignment,
        },
        metadata,
        raw,
        tokenizer,
    ))
}

/// Capture one tokenizer metadata value into the structured tokenizer.
fn capture_tokenizer(
    input: &mut BufReader<File>,
    value_type: u32,
    key: &str,
    tokenizer: &mut GgufTokenizer,
) -> Result<()> {
    match key {
        "tokenizer.ggml.model" => tokenizer.model = Some(read_string(input)?),
        "tokenizer.ggml.tokens" => tokenizer.tokens = read_string_array(input, value_type)?,
        "tokenizer.ggml.merges" => tokenizer.merges = read_string_array(input, value_type)?,
        "tokenizer.ggml.scores" => {
            if value_type == 9 {
                let element_type = read_u32(input)?;
                let count = read_u64(input)?;
                if count > 1_000_000 {
                    return Err(HarError::Invalid {
                        kind: "GGUF metadata array",
                        message: count.to_string(),
                    });
                }
                if element_type == 6 {
                    let mut values = Vec::with_capacity(count as usize);
                    for _ in 0..count {
                        values.push(f32::from_le_bytes(
                            read_exact(input, 4)?.try_into().unwrap(),
                        ));
                    }
                    tokenizer.scores = values;
                } else {
                    for _ in 0..count {
                        let _ = read_metadata(input, element_type)?;
                    }
                }
            }
        }
        "tokenizer.ggml.token_type" => {
            if value_type == 9 {
                let element_type = read_u32(input)?;
                let count = read_u64(input)?;
                if count > 1_000_000 {
                    return Err(HarError::Invalid {
                        kind: "GGUF metadata array",
                        message: count.to_string(),
                    });
                }
                if element_type == 4 {
                    let mut values = Vec::with_capacity(count as usize);
                    for _ in 0..count {
                        values.push(read_u32(input)?);
                    }
                    tokenizer.token_types = values;
                } else {
                    for _ in 0..count {
                        let _ = read_metadata(input, element_type)?;
                    }
                }
            }
        }
        "tokenizer.ggml.added_tokens" => {
            tokenizer.added_tokens = read_string_array(input, value_type)?
        }
        "tokenizer.ggml.eos_token_id" => {
            if value_type == 4 {
                tokenizer.eos_token_id = Some(read_u32(input)?);
            }
        }
        "tokenizer.ggml.bos_token_id" => {
            if value_type == 4 {
                tokenizer.bos_token_id = Some(read_u32(input)?);
            }
        }
        "tokenizer.ggml.chat_template" => tokenizer.chat_template = Some(read_string(input)?),
        _ => {
            let _ = read_metadata(input, value_type)?;
        }
    }
    Ok(())
}

/// Read an array of strings (either a direct string or an array of strings).
fn read_string_array(input: &mut BufReader<File>, value_type: u32) -> Result<Vec<String>> {
    if value_type != 9 {
        // Not an array: consume as a plain value.
        let _ = read_metadata(input, value_type)?;
        return Ok(Vec::new());
    }
    let element_type = read_u32(input)?;
    let count = read_u64(input)?;
    if count > 1_000_000 {
        return Err(HarError::Invalid {
            kind: "GGUF metadata array",
            message: count.to_string(),
        });
    }
    let mut values = Vec::with_capacity(count as usize);
    for _ in 0..count {
        if element_type == 8 {
            values.push(read_string(input)?);
        } else {
            let _ = read_metadata(input, element_type)?;
        }
    }
    Ok(values)
}

fn read_metadata(input: &mut BufReader<File>, value_type: u32) -> Result<String> {
    match value_type {
        0 => Ok(read_u8(input)?.to_string()),
        1 => Ok((read_u8(input)? as i8).to_string()),
        2 => Ok(read_u16(input)?.to_string()),
        3 => Ok((read_u16(input)? as i16).to_string()),
        4 => Ok(read_u32(input)?.to_string()),
        5 => Ok((read_u32(input)? as i32).to_string()),
        6 => Ok(f32::from_le_bytes(read_exact(input, 4)?.try_into().unwrap()).to_string()),
        7 => Ok((read_u8(input)? != 0).to_string()),
        8 => read_string(input),
        9 => {
            let element_type = read_u32(input)?;
            let count = read_u64(input)?;
            if count > 1_000_000 {
                return Err(HarError::Invalid {
                    kind: "GGUF metadata array",
                    message: count.to_string(),
                });
            }
            for _ in 0..count {
                let _ = read_metadata(input, element_type)?;
            }
            Ok(format!("array[{count}]"))
        }
        10 => Ok(read_u64(input)?.to_string()),
        11 => Ok((read_u64(input)? as i64).to_string()),
        12 => Ok(f64::from_le_bytes(read_exact(input, 8)?.try_into().unwrap()).to_string()),
        other => Err(HarError::Unsupported {
            kind: "GGUF metadata type",
            message: other.to_string(),
        }),
    }
}
fn metadata_number(map: &BTreeMap<String, String>, key: &str) -> Option<u64> {
    map.get(key).and_then(|value| value.parse().ok())
}
fn read_exact(input: &mut BufReader<File>, count: usize) -> Result<Vec<u8>> {
    let mut value = vec![0u8; count];
    input.read_exact(&mut value)?;
    Ok(value)
}
fn read_u8(input: &mut BufReader<File>) -> Result<u8> {
    Ok(read_exact(input, 1)?[0])
}
fn read_u16(input: &mut BufReader<File>) -> Result<u16> {
    Ok(u16::from_le_bytes(
        read_exact(input, 2)?.try_into().unwrap(),
    ))
}
fn read_u32(input: &mut BufReader<File>) -> Result<u32> {
    Ok(u32::from_le_bytes(
        read_exact(input, 4)?.try_into().unwrap(),
    ))
}
fn read_u64(input: &mut BufReader<File>) -> Result<u64> {
    Ok(u64::from_le_bytes(
        read_exact(input, 8)?.try_into().unwrap(),
    ))
}
fn read_string(input: &mut BufReader<File>) -> Result<String> {
    let length = read_u64(input)?;
    if length > 1 << 30 {
        return Err(HarError::Invalid {
            kind: "GGUF string",
            message: length.to_string(),
        });
    }
    Ok(String::from_utf8_lossy(&read_exact(input, length as usize)?).into_owned())
}
fn align(value: u64, alignment: u64) -> u64 {
    value.saturating_add(alignment - 1) / alignment * alignment
}

pub fn tensor_type_name(type_id: u32) -> &'static str {
    match type_id {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        30 => "BF16",
        36 => "R4X_D32A",
        39 => "MXFP4",
        other => {
            let _ = other;
            "UNKNOWN"
        }
    }
}
pub fn tensor_block_size(type_id: u32) -> u64 {
    match type_id {
        0 | 1 | 30 => 1,
        2 | 3 | 6 | 7 | 8 => 32,
        10..=15 | 36 | 39 => 256,
        _ => 0,
    }
}
pub fn tensor_type_size(type_id: u32) -> u64 {
    match type_id {
        0 => 4,
        1 | 30 => 2,
        2 => 18,
        3 => 20,
        6 => 22,
        7 => 24,
        8 => 34,
        10 => 84,
        11 => 110,
        12 => 144,
        13 => 176,
        14 => 210,
        15 => 292,
        36 => 144,
        39 => 256,
        _ => 0,
    }
}
pub fn tensor_payload_bytes(type_id: u32, elements: u64) -> u64 {
    let block = tensor_block_size(type_id);
    let size = tensor_type_size(type_id);
    if block == 0 || size == 0 {
        0
    } else {
        elements.saturating_add(block - 1) / block * size
    }
}
fn parse_layer(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("blk.")?;
    rest.split('.').next()?.parse().ok()
}
fn classify_role(name: &str) -> TensorRole {
    TensorRole::classify(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn q4_geometry_is_explicit() {
        assert_eq!(tensor_payload_bytes(12, 5120), 2880);
        assert_eq!(tensor_type_size(12), 144);
    }
    #[test]
    fn role_classification_is_deterministic() {
        assert_eq!(
            classify_role("blk.0.ffn_gate.weight"),
            TensorRole::FeedForward
        );
        assert_eq!(
            classify_role("blk.0.attn_qkv.weight"),
            TensorRole::Attention
        );
    }
    #[test]
    fn typed_fields_keep_legacy_wire_form() {
        let descriptor = TensorDescriptor {
            interface: TENSOR_INTERFACE.into(),
            ordinal: 0,
            name: "blk.0.ffn_gate.weight".into(),
            dimensions: vec![5120, 32],
            ggml_type: 12,
            quantization: QuantFormat::Q4K,
            element_count: 163840,
            payload_bytes: 92160,
            raw_span_bytes: 92160,
            file_offset: 1024,
            alignment_bytes: 256,
            layer: Some(0),
            role: TensorRole::FeedForward,
            hotness: 1,
            is_weight: true,
            is_mtp: false,
        };
        let value = serde_json::to_value(&descriptor).unwrap();
        assert_eq!(value["role"], serde_json::json!("FEED_FORWARD"));
        assert_eq!(value["quantization"], serde_json::json!("Q4_K"));
    }
}
