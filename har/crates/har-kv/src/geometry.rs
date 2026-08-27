use std::fmt;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KVNativeFormat {
    F16,
    Q8_0,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KVGeometry {
    pub model_id: String,
    pub attention_layer_ids: Vec<u32>,
    pub kv_heads: u32,
    pub query_heads: u32,
    pub head_dim_k: u32,
    pub head_dim_v: u32,
    pub page_tokens: u32,
    pub q8_block_elements: u32,
    pub q8_scale_bytes: u32,
    pub q8_value_bytes: u32,
    pub row_alignment_bytes: u64,
    pub allocator_alignment_bytes: u64,
    pub context_length: Option<u32>,
    pub model_sha256: Option<String>,
    pub metadata_sha256: Option<String>,
    pub architecture: Option<String>,
    pub weight_format: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ByteProjection {
    pub tokens: u64,
    pub k_bytes: u64,
    pub v_bytes: u64,
    pub total_bytes: u64,
}

impl KVGeometry {
    pub fn qwen36_27b(model_sha256: impl Into<String>, metadata_sha256: impl Into<String>) -> Self {
        Self {
            model_id: "synthetic-hybrid-model".to_string(),
            attention_layer_ids: (0..64).filter(|layer| (layer + 1) % 4 == 0).collect(),
            kv_heads: 4,
            query_heads: 24,
            head_dim_k: 256,
            head_dim_v: 256,
            page_tokens: 32,
            q8_block_elements: 32,
            q8_scale_bytes: 2,
            q8_value_bytes: 1,
            row_alignment_bytes: 1,
            allocator_alignment_bytes: 256,
            context_length: Some(262_144),
            model_sha256: Some(model_sha256.into()),
            metadata_sha256: Some(metadata_sha256.into()),
            architecture: Some("qwen35".to_string()),
            weight_format: "Q4".to_string(),
        }
    }

    pub fn layer_count(&self) -> u64 {
        self.attention_layer_ids.len() as u64
    }

    pub fn k_elements_per_layer(&self) -> u64 {
        self.kv_heads as u64 * self.head_dim_k as u64
    }

    pub fn v_elements_per_layer(&self) -> u64 {
        self.kv_heads as u64 * self.head_dim_v as u64
    }

    pub fn align_up(value: u64, alignment: u64) -> u64 {
        if alignment <= 1 {
            value
        } else {
            value.div_ceil(alignment) * alignment
        }
    }

    pub fn q8_row_bytes(&self, key: bool) -> u64 {
        let elements = if key {
            self.k_elements_per_layer()
        } else {
            self.v_elements_per_layer()
        };
        let blocks = elements.div_ceil(self.q8_block_elements as u64);
        Self::align_up(
            blocks
                * (self.q8_scale_bytes as u64
                    + self.q8_block_elements as u64 * self.q8_value_bytes as u64),
            self.row_alignment_bytes,
        )
    }

    pub fn f16_row_bytes(&self, key: bool) -> u64 {
        let elements = if key {
            self.k_elements_per_layer()
        } else {
            self.v_elements_per_layer()
        };
        Self::align_up(elements * 2, self.row_alignment_bytes)
    }

    pub fn row_bytes(&self, key: bool, format: KVNativeFormat) -> u64 {
        match format {
            KVNativeFormat::F16 => self.f16_row_bytes(key),
            KVNativeFormat::Q8_0 => self.q8_row_bytes(key),
        }
    }

    pub fn bytes_per_token(&self, k: KVNativeFormat, v: KVNativeFormat) -> ByteProjection {
        let k_bytes = self.layer_count() * self.row_bytes(true, k);
        let v_bytes = self.layer_count() * self.row_bytes(false, v);
        ByteProjection {
            tokens: 1,
            k_bytes,
            v_bytes,
            total_bytes: k_bytes + v_bytes,
        }
    }

    pub fn project(&self, tokens: u64, k: KVNativeFormat, v: KVNativeFormat) -> ByteProjection {
        let per = self.bytes_per_token(k, v);
        ByteProjection {
            tokens,
            k_bytes: per.k_bytes * tokens,
            v_bytes: per.v_bytes * tokens,
            total_bytes: per.total_bytes * tokens,
        }
    }

    pub fn q8_scales_per_row(&self, key: bool) -> u64 {
        let elements = if key {
            self.k_elements_per_layer()
        } else {
            self.v_elements_per_layer()
        };
        elements.div_ceil(self.q8_block_elements as u64) * self.q8_scale_bytes as u64
    }

    pub fn validate(&self) -> Result<(), GeometryError> {
        if self.model_id.is_empty() || self.attention_layer_ids.is_empty() {
            return Err(GeometryError("model/layer identity is empty"));
        }
        if self.kv_heads == 0
            || self.query_heads == 0
            || self.head_dim_k == 0
            || self.head_dim_v == 0
        {
            return Err(GeometryError("head geometry must be non-zero"));
        }
        if self.q8_block_elements == 0 || self.row_alignment_bytes == 0 {
            return Err(GeometryError("Q8 block/alignment must be non-zero"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeometryError(pub &'static str);

impl fmt::Display for GeometryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for GeometryError {}
