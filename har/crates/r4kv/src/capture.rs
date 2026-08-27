//! Authoritative attention-capture schema + fail-closed validator.
//!
//! Permanent infrastructure for R4KV/R4X/DFlash/QAT/long-context research.
//! Every captured tensor MUST carry a CaptureManifest; downstream tools must
//! reject captures failing `validate()` (directive: no "probably correct").

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    pub tensor_name: String, // e.g. "Qcur", "Kcur", "Vcur", "attn_probs"
    pub layer: u32,
    pub dtype: String,           // "f32" | "f16" | "bf16"
    pub logical_shape: Vec<u64>, // semantics-documented dims
    pub physical_shape: Vec<u64>,
    pub strides_bytes: Vec<u64>,
    pub element_count: u64,
    pub byte_count: u64,
    pub sha256: String,
}

#[derive(Debug, Clone)]
pub struct PositionDescriptor {
    /// absolute token position in sequence (0-based)
    pub absolute_pos: u32,
    /// prefix(prompt) vs generated
    pub origin: Origin,
    /// R4KV page containing this position
    pub page_id: Option<u64>,
    pub offset_in_page: Option<u32>,
    pub page_epoch: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Origin {
    PromptPrefix,
    Generated,
}

#[derive(Debug, Clone)]
pub struct GqaMap {
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    /// q_head index -> kv_head index (must be complete + uniform groups)
    pub q_to_kv: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct RopeState {
    pub mode: String, // "mrope_interleaved"
    pub theta: f64,
    pub partial_rotary_factor: f64, // qwen3_5: 0.25
    pub mrope_sections: Option<Vec<u32>>,
    pub applied_to_captured_tensors: bool,
    pub position_ids_documented: bool,
}

#[derive(Debug, Clone)]
pub struct CaptureManifest {
    pub capture_version: u32,
    pub capture_id: String, // shared by Q/K/V/probs of one event
    pub model_sha256: String,
    pub tokenizer_identity: String,
    pub quant_identity: String,
    pub runtime_commit: String,
    pub layer: u32,
    pub sequence_id: String,
    pub prompt_prefix_positions: u32,
    pub generated_query_positions: u32,
    pub total_kv_positions: u32,
    pub tensors: Vec<TensorDescriptor>,
    pub gqa: GqaMap,
    pub rope: RopeState,
    pub causal_mask_semantics: String, // "standard_triangular" | documented variant
    pub positions: Vec<PositionDescriptor>,
    pub page_size_tokens: u32,
    pub includes_post_softmax_probs: bool,
}

impl CaptureManifest {
    pub fn validate(&self, payloads: &HashMap<String, usize>) -> Result<(), String> {
        if self.capture_id.is_empty() {
            return Err("capture_id empty".into());
        }
        if self.gqa.q_heads == 0 || self.gqa.kv_heads == 0 || self.gqa.head_dim == 0 {
            return Err("zero geometry".into());
        }
        if self.gqa.q_heads % self.gqa.kv_heads != 0 {
            return Err("q_heads not divisible by kv_heads".into());
        }
        if self.gqa.q_to_kv.len() as u32 != self.gqa.q_heads {
            return Err("GQA map incomplete".into());
        }
        for (qi, &kv) in self.gqa.q_to_kv.iter().enumerate() {
            if kv >= self.gqa.kv_heads {
                return Err(format!("q_head {qi} maps out of range"));
            }
            if qi as u32 / (self.gqa.q_heads / self.gqa.kv_heads) != kv {
                return Err("GQA map not contiguous-uniform".into());
            }
        }
        if self.total_kv_positions != self.prompt_prefix_positions + self.generated_query_positions
        {
            return Err("total_kv != prefix + generated".into());
        }
        if self.positions.len() as u32 != self.total_kv_positions {
            return Err("position descriptors incomplete".into());
        }
        let mut prefix_seen = false;
        for p in &self.positions {
            if p.origin == Origin::PromptPrefix {
                prefix_seen = true;
            }
            if p.page_id.is_none() {
                return Err("missing page mapping".into());
            }
        }
        if !prefix_seen {
            return Err("no prompt-prefix position recorded".into());
        }
        if !self.rope.applied_to_captured_tensors {
            return Err("captured tensors must be execution-equivalent (post-RoPE)".into());
        }
        if !self.rope.position_ids_documented {
            return Err("rope position ids undocumented".into());
        }
        if self.causal_mask_semantics.is_empty() {
            return Err("causal mask semantics required".into());
        }
        if self.page_size_tokens == 0 {
            return Err("page size required".into());
        }
        for t in &self.tensors {
            let prod: u64 = t.logical_shape.iter().product();
            if prod != t.element_count {
                return Err(format!(
                    "{} logical_shape product != element_count",
                    t.tensor_name
                ));
            }
            let expect_bytes = match t.dtype.as_str() {
                "f32" => t.element_count * 4,
                "f16" | "bf16" => t.element_count * 2,
                _ => return Err(format!("{} unknown dtype", t.tensor_name)),
            };
            if t.byte_count != expect_bytes {
                return Err(format!("{} byte_count mismatch", t.tensor_name));
            }
            if t.strides_bytes.len() != t.physical_shape.len() {
                return Err(format!("{} stride rank mismatch", t.tensor_name));
            }
            match payloads.get(&t.sha256) {
                Some(sz) if *sz as u64 == t.byte_count => {}
                _ => return Err(format!("{} payload missing/size mismatch", t.tensor_name)),
            }
        }
        // Q/K/V identity coherence
        let names: Vec<&str> = self
            .tensors
            .iter()
            .map(|t| t.tensor_name.as_str())
            .collect();
        for req in ["Qcur", "Kcur", "Vcur"] {
            if !names.iter().any(|n| n.starts_with(req)) && !self.includes_post_softmax_probs
                || (req == "Vcur" && !names.iter().any(|n| n.starts_with(req)))
            {
                // direct-probability captures may omit V only when explicitly probs-only
                if !(self.includes_post_softmax_probs && req == "Vcur")
                    && !names.iter().any(|n| n.starts_with(req))
                {
                    return Err(format!("missing required tensor {}", req));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> CaptureManifest {
        CaptureManifest {
            capture_version: 2,
            capture_id: "cap-test-001".into(),
            model_sha256: "synthetic-model".into(),
            tokenizer_identity: "synthetic-tokenizer".into(),
            quant_identity: "r4x-q4std".into(),
            runtime_commit: "synthetic-runtime".into(),
            layer: 3,
            sequence_id: "synthetic-sequence".into(),
            prompt_prefix_positions: 7,
            generated_query_positions: 8,
            total_kv_positions: 15,
            tensors: vec![
                TensorDescriptor {
                    tensor_name: "Vcur".into(),
                    layer: 3,
                    dtype: "f32".into(),
                    logical_shape: vec![15, 4, 256],
                    physical_shape: vec![15, 4, 256],
                    strides_bytes: vec![4096, 1024, 4],
                    element_count: 15 * 4 * 256,
                    byte_count: 15 * 4 * 256 * 4,
                    sha256: "cccc".into(),
                },
                TensorDescriptor {
                    tensor_name: "Kcur".into(),
                    layer: 3,
                    dtype: "f32".into(),
                    logical_shape: vec![15, 4, 256],
                    physical_shape: vec![15, 4, 256],
                    strides_bytes: vec![4096, 1024, 4],
                    element_count: 15 * 4 * 256,
                    byte_count: 15 * 4 * 256 * 4,
                    sha256: "bbbb".into(),
                },
                TensorDescriptor {
                    tensor_name: "Qcur".into(),
                    layer: 3,
                    dtype: "f32".into(),
                    logical_shape: vec![8, 24, 256],
                    physical_shape: vec![8, 24, 256],
                    strides_bytes: vec![24576, 1024, 4],
                    element_count: 8 * 24 * 256,
                    byte_count: 8 * 24 * 256 * 4,
                    sha256: "aaaa".into(),
                },
            ],
            gqa: GqaMap {
                q_heads: 24,
                kv_heads: 4,
                head_dim: 256,
                q_to_kv: (0..24).map(|i| i / 6).collect(),
            },
            rope: RopeState {
                mode: "mrope_interleaved".into(),
                theta: 1e7,
                partial_rotary_factor: 0.25,
                mrope_sections: Some(vec![11, 11, 10]),
                applied_to_captured_tensors: true,
                position_ids_documented: true,
            },
            causal_mask_semantics: "standard_triangular".into(),
            positions: (0..15)
                .map(|p| PositionDescriptor {
                    absolute_pos: p,
                    origin: if p < 7 {
                        Origin::PromptPrefix
                    } else {
                        Origin::Generated
                    },
                    page_id: Some((p / 8) as u64),
                    offset_in_page: Some(p % 8),
                    page_epoch: Some(0),
                })
                .collect(),
            page_size_tokens: 8,
            includes_post_softmax_probs: false,
        }
    }

    fn payloads_ok(m: &CaptureManifest) -> HashMap<String, usize> {
        m.tensors
            .iter()
            .map(|t| (t.sha256.clone(), t.byte_count as usize))
            .collect()
    }

    #[test]
    fn valid_manifest_passes() {
        assert!(base().validate(&payloads_ok(&base())).is_ok());
    }

    #[test]
    fn bad_gqa_rejected() {
        let mut m = base();
        m.gqa.q_to_kv[0] = 9;
        assert!(m.validate(&payloads_ok(&m)).is_err());
    }

    #[test]
    fn position_mismatch_rejected() {
        let mut m = base();
        m.total_kv_positions = 99;
        assert!(m.validate(&payloads_ok(&m)).is_err());
    }

    #[test]
    fn pre_rope_capture_rejected() {
        let mut m = base();
        m.rope.applied_to_captured_tensors = false;
        assert!(m.validate(&payloads_ok(&m)).is_err());
    }

    #[test]
    fn missing_page_mapping_rejected() {
        let mut m = base();
        m.positions[3].page_id = None;
        assert!(m.validate(&payloads_ok(&m)).is_err());
    }
}
