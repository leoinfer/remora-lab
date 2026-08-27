use std::collections::BTreeMap;
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CodecPayload {
    pub codec_id: String,
    pub version: String,
    pub bytes: Vec<u8>,
    pub source_bytes: u64,
    pub metadata_bytes: u64,
    pub dictionary_bytes: u64,
    pub scratch_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CodecError {
    InvalidPayload,
    Unsupported(String),
    SizeOverflow,
}

impl fmt::Display for CodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPayload => formatter.write_str("invalid codec payload"),
            Self::Unsupported(name) => write!(
                formatter,
                "codec unsupported in Rust orchestration layer: {name}"
            ),
            Self::SizeOverflow => formatter.write_str("codec size overflow"),
        }
    }
}
impl std::error::Error for CodecError {}

/// Byte-level codec contract. Numeric reconstruction kernels are deliberately
/// outside this crate; the trait lets native-kernel registry/residency layer bind a native decoder.
pub trait KVCodec: Send + Sync {
    fn id(&self) -> &str;
    fn version(&self) -> &str;
    fn encode(&self, source: &[u8]) -> Result<CodecPayload, CodecError>;
    fn decode(&self, payload: &CodecPayload) -> Result<Vec<u8>, CodecError>;
}

/// Lossless, dependency-free control codec used for Rust identity/store tests.
/// It delta-XORs adjacent bytes, then emits zero runs and literal runs.  It is
/// not a replacement for the measured zlib Q8 control in offline evidence.
#[derive(Clone, Debug, Default)]
pub struct LosslessXorRleCodec;

impl LosslessXorRleCodec {
    fn delta_encode(source: &[u8]) -> Vec<u8> {
        let mut delta = Vec::with_capacity(source.len());
        let mut previous = 0u8;
        for &byte in source {
            delta.push(byte ^ previous);
            previous = byte;
        }
        delta
    }

    fn delta_decode(delta: &[u8]) -> Vec<u8> {
        let mut source = Vec::with_capacity(delta.len());
        let mut previous = 0u8;
        for &byte in delta {
            let value = byte ^ previous;
            source.push(value);
            previous = value;
        }
        source
    }

    fn rle_encode(bytes: &[u8]) -> Vec<u8> {
        // Record: tag=0 zero run, tag=1 literal run; run length is u16 LE.
        let mut result = Vec::new();
        let mut index = 0usize;
        while index < bytes.len() {
            if bytes[index] == 0 {
                let start = index;
                while index < bytes.len() && bytes[index] == 0 && index - start < u16::MAX as usize
                {
                    index += 1;
                }
                result.push(0);
                result.extend_from_slice(&((index - start) as u16).to_le_bytes());
            } else {
                let start = index;
                while index < bytes.len()
                    && (bytes[index] != 0 || index + 1 >= bytes.len() || bytes[index + 1] != 0)
                    && index - start < u16::MAX as usize
                {
                    index += 1;
                }
                if index == start {
                    index += 1;
                }
                result.push(1);
                result.extend_from_slice(&((index - start) as u16).to_le_bytes());
                result.extend_from_slice(&bytes[start..index]);
            }
        }
        result
    }

    fn rle_decode(bytes: &[u8]) -> Result<Vec<u8>, CodecError> {
        let mut result = Vec::new();
        let mut index = 0usize;
        while index < bytes.len() {
            if index + 3 > bytes.len() {
                return Err(CodecError::InvalidPayload);
            }
            let tag = bytes[index];
            let length = u16::from_le_bytes([bytes[index + 1], bytes[index + 2]]) as usize;
            index += 3;
            match tag {
                0 => result.resize(
                    result
                        .len()
                        .checked_add(length)
                        .ok_or(CodecError::SizeOverflow)?,
                    0,
                ),
                1 => {
                    if index + length > bytes.len() {
                        return Err(CodecError::InvalidPayload);
                    }
                    result.extend_from_slice(&bytes[index..index + length]);
                    index += length;
                }
                _ => return Err(CodecError::InvalidPayload),
            }
        }
        Ok(result)
    }
}

impl KVCodec for LosslessXorRleCodec {
    fn id(&self) -> &str {
        "lossless_q8_xor_rle"
    }
    fn version(&self) -> &str {
        "v1"
    }
    fn encode(&self, source: &[u8]) -> Result<CodecPayload, CodecError> {
        let delta = Self::delta_encode(source);
        let bytes = Self::rle_encode(&delta);
        Ok(CodecPayload {
            codec_id: self.id().to_string(),
            version: self.version().to_string(),
            bytes,
            source_bytes: source.len() as u64,
            metadata_bytes: 0,
            dictionary_bytes: 0,
            scratch_bytes: source.len() as u64,
        })
    }
    fn decode(&self, payload: &CodecPayload) -> Result<Vec<u8>, CodecError> {
        if payload.codec_id != self.id() || payload.version != self.version() {
            return Err(CodecError::InvalidPayload);
        }
        let decoded = Self::delta_decode(&Self::rle_decode(&payload.bytes)?);
        if decoded.len() != payload.source_bytes as usize {
            return Err(CodecError::InvalidPayload);
        }
        Ok(decoded)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredCodec {
    pub id: String,
    pub field: String,
    pub persistent_bytes: u64,
    pub reconstruction_bytes: u64,
    pub encode_time_us: Option<u64>,
    pub decode_time_us: Option<u64>,
    pub attention_score_error: Option<f64>,
    pub attention_distribution_divergence: Option<f64>,
    pub attention_output_error: Option<f64>,
    pub exact_tail_retrieval: Option<f64>,
    pub fallback_rate: Option<f64>,
}

/// Measured table imported from the existing real-capture CSV.  The Rust
/// runtime uses this as a selection/evidence catalog; it does not reinterpret
/// numerical quality as a certificate.
#[derive(Clone, Debug, Default)]
pub struct MeasuredCodecCatalog {
    profiles: BTreeMap<(String, String), MeasuredCodec>,
}

impl MeasuredCodecCatalog {
    pub fn insert(&mut self, profile: MeasuredCodec) {
        self.profiles
            .insert((profile.field.clone(), profile.id.clone()), profile);
    }
    pub fn get(&self, field: &str, codec: &str) -> Option<&MeasuredCodec> {
        self.profiles.get(&(field.to_string(), codec.to_string()))
    }
    pub fn profiles(&self) -> impl Iterator<Item = &MeasuredCodec> {
        self.profiles.values()
    }
    pub fn from_csv(csv: &str) -> Result<Self, CodecError> {
        let mut lines = csv.lines();
        let header = lines.next().ok_or(CodecError::InvalidPayload)?;
        let columns: Vec<&str> = header.split(',').collect();
        let index = |name: &str| {
            columns
                .iter()
                .position(|column| *column == name)
                .ok_or(CodecError::InvalidPayload)
        };
        let field_i = index("field")?;
        let codec_i = index("codec")?;
        let persistent_i = index("persistent_bytes")?;
        let reconstruction_i = index("scratch_bytes").or_else(|_| index("reconstruction_bytes"))?;
        let score_i = index("attention_score_mse").ok();
        let jsd_i = index("attention_distribution_jsd").ok();
        let output_i = index("attention_output_normalized_mse").ok();
        let mut catalog = Self::default();
        for line in lines.filter(|line| !line.trim().is_empty()) {
            let values: Vec<&str> = line.split(',').collect();
            if values.len() < columns.len() {
                continue;
            }
            let parse_u64 = |value: &str| {
                value
                    .parse::<f64>()
                    .ok()
                    .map(|x| x.max(0.0) as u64)
                    .unwrap_or(0)
            };
            let parse_f64 = |maybe: Option<usize>| {
                maybe
                    .and_then(|i| values.get(i))
                    .and_then(|v| v.parse::<f64>().ok())
            };
            catalog.insert(MeasuredCodec {
                id: values[codec_i].to_string(),
                field: values[field_i].to_string(),
                persistent_bytes: parse_u64(values[persistent_i]),
                reconstruction_bytes: parse_u64(values[reconstruction_i]),
                encode_time_us: None,
                decode_time_us: None,
                attention_score_error: parse_f64(score_i),
                attention_distribution_divergence: parse_f64(jsd_i),
                attention_output_error: parse_f64(output_i),
                exact_tail_retrieval: None,
                fallback_rate: None,
            });
        }
        Ok(catalog)
    }
}

impl MeasuredCodec {
    pub fn q8_quality_wall(
        &self,
        native_score: Option<f64>,
        native_jsd: Option<f64>,
        native_output: Option<f64>,
    ) -> bool {
        let worse = |candidate: Option<f64>, control: Option<f64>| match (candidate, control) {
            (Some(a), Some(b)) => a > b * 1.25 + 1e-12,
            _ => true,
        };
        worse(self.attention_score_error, native_score)
            || worse(self.attention_distribution_divergence, native_jsd)
            || worse(self.attention_output_error, native_output)
    }
}
