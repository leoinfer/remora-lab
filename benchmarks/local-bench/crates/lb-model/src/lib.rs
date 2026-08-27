//! # lb-model
//!
//! Header-only GGUF model inspection. Reads the GGUF header + tensor info
//! section and produces the metadata a synthetic performance simulation
//! needs — **never touches weight data** (light speed by construction).
//!
//! Output [`ModelMeta`] is serde-serializable so simulations, search and
//! calibration all share one model description.
//!
//! ## No-cheat guarantees
//!
//! 1. **No weight reads**: only the header, metadata KV and tensor-info
//!    section are parsed; tensor payload offsets are skipped.
//! 2. **Byte accounting is exact**: per-tensor bytes come from the GGUF
//!    tensor size * block-size table (the same table ggml uses), summed per
//!    quant class — this makes byte accounting auditable for any input.
//! 3. **Geometry is derived from tensor names** (`blk.N.<class>`), not from
//!    architecture heuristics, so any model family works.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

// ---------------------------------------------------------------------------
// GGUF primitives
// ---------------------------------------------------------------------------

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF"

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ValueType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Bool,
    String,
    Array,
    Uint64,
    Int64,
    Float64,
}

impl ValueType {
    fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::Uint8,
            1 => Self::Int8,
            2 => Self::Uint16,
            3 => Self::Int16,
            4 => Self::Uint32,
            5 => Self::Int32,
            6 => Self::Float32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::Uint64,
            11 => Self::Int64,
            12 => Self::Float64,
            _ => return None,
        })
    }
}

// ggml type traits: (type_size_bytes, block_size_elements)
// authoritative mapping from gguf.constants.GGML_QUANT_SIZES (2026 enum:
// BF16=30, Q8_K=15, integer types 24-28, MXFP4=39, NVFP4=40, Q1_0=41)
fn quant_block(type_id: u32) -> Option<(u64, u64)> {
    Some(match type_id {
        0 => (4, 1),      // F32
        1 => (2, 1),      // F16
        2 => (18, 32),    // Q4_0
        3 => (20, 32),    // Q4_1
        6 => (22, 32),    // Q5_0
        7 => (24, 32),    // Q5_1
        8 => (34, 32),    // Q8_0
        9 => (40, 32),    // Q8_1
        10 => (84, 256),  // Q2_K
        11 => (110, 256), // Q3_K
        12 => (144, 256), // Q4_K
        13 => (176, 256), // Q5_K
        14 => (210, 256), // Q6_K
        15 => (292, 256), // Q8_K
        16 => (66, 256),  // IQ2_XXS
        17 => (74, 256),  // IQ2_XS
        18 => (98, 256),  // IQ3_XXS
        19 => (50, 256),  // IQ1_S
        20 => (18, 32),   // IQ4_NL
        21 => (110, 256), // IQ3_S
        22 => (82, 256),  // IQ2_S
        23 => (136, 256), // IQ4_XS
        24 => (1, 1),     // I8
        25 => (2, 1),     // I16
        26 => (4, 1),     // I32
        27 => (8, 1),     // I64
        28 => (8, 1),     // F64
        29 => (56, 256),  // IQ1_M
        30 => (2, 1),     // BF16
        34 => (54, 256),  // TQ1_0
        35 => (66, 256),  // TQ2_0
        39 => (17, 32),   // MXFP4
        40 => (36, 64),   // NVFP4
        41 => (18, 128),  // Q1_0
        _ => return None,
    })
}

fn quant_name(type_id: u32) -> String {
    const NAMES: [&str; 42] = [
        "F32", "F16", "Q4_0", "Q4_1", "?4", "?5", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K", "Q3_K",
        "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S",
        "IQ2_S", "IQ4_XS", "I8", "I16", "I32", "I64", "F64", "IQ1_M", "BF16", "?31", "?32", "?33",
        "TQ1_0", "TQ2_0", "?36", "?37", "?38", "MXFP4", "NVFP4", "Q1_0",
    ];
    NAMES.get(type_id as usize).unwrap_or(&"?").to_string()
}

// ---------------------------------------------------------------------------
// Parsed metadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuantMix {
    pub tensor_count: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMeta {
    pub path: String,
    pub file_bytes: u64,
    pub tensor_count: usize,
    pub arch: String,
    pub name: String,
    pub quant_mix: BTreeMap<String, QuantMix>,
    pub total_tensor_bytes: u64,
    pub total_elements: u64,
    pub non_layer_bytes: u64,
    pub layer_bytes: u64,
    pub n_layers: usize,
    pub n_attention_layers: usize,
    pub kv_heads: u64,
    pub key_length: u64,
    pub value_length: u64,
    pub embedding_length: u64,
    pub head_count: u64,
    pub has_mtp: bool,
    pub mtp_bytes: u64,
    pub metadata: BTreeMap<String, String>,
}

impl ModelMeta {
    /// Exact KV cache bytes per token for a given K/V element size.
    /// Attention KV only — recurrent state is fixed-size and not context-bound.
    pub fn kv_bytes_per_token(&self, k_elem: u64, v_elem: u64) -> u64 {
        self.n_attention_layers as u64
            * self.kv_heads
            * (self.key_length * k_elem + self.value_length * v_elem)
    }

    /// Approximate bytes per layer (total minus non-layer, divided by layers).
    pub fn avg_layer_bytes(&self) -> f64 {
        if self.n_layers == 0 {
            0.0
        } else {
            self.layer_bytes as f64 / self.n_layers as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

pub fn inspect(path: &Path) -> Result<ModelMeta, String> {
    let mut f = BufReader::new(File::open(path).map_err(|e| e.to_string())?);
    let file_bytes = f.get_ref().metadata().map_err(|e| e.to_string())?.len();

    // header
    let magic = read_u32(&mut f)?;
    if magic != GGUF_MAGIC {
        return Err(format!("not a GGUF file (magic 0x{magic:08x})"));
    }
    let _version = read_u32(&mut f)?;
    let tensor_count = read_u64(&mut f)?;

    // metadata KV — walk values, keeping the fields we care about
    let mut metadata: BTreeMap<String, String> = BTreeMap::new();
    let kv_count = read_u64(&mut f)?;
    for _ in 0..kv_count {
        let key = read_string(&mut f)?;
        let vtype = read_u32(&mut f)?;
        let vt = ValueType::from_u32(vtype).ok_or_else(|| format!("unknown value type {vtype}"))?;
        let value = read_value(&mut f, vt)?;
        metadata.insert(key, value);
    }

    // tensor info section: names, dims, types, offsets — no payload reads
    let mut quant_mix: BTreeMap<String, QuantMix> = BTreeMap::new();
    let mut total_tensor_bytes: u64 = 0;
    let mut total_elements: u64 = 0;
    let mut non_layer_bytes: u64 = 0;
    let mut layer_bytes: u64 = 0;
    let mut n_layers: usize = 0;
    let mut has_mtp = false;
    let mut mtp_bytes: u64 = 0;

    let mut attn_layers: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
    for _ in 0..tensor_count {
        let name = read_string(&mut f)?;
        let n_dims = read_u32(&mut f)?;
        let mut nelems: u64 = 1;
        for _ in 0..n_dims {
            let d = read_u64(&mut f)?;
            nelems = nelems.saturating_mul(d);
        }
        let type_id = read_u32(&mut f)?;
        let _offset = read_u64(&mut f)?;

        let (tsz, bsz) = quant_block(type_id)
            .ok_or_else(|| format!("tensor {name}: unsupported type {type_id}"))?;
        let bytes = nelems.div_ceil(bsz).saturating_mul(tsz);
        total_tensor_bytes += bytes;
        total_elements += nelems;

        let entry = quant_mix.entry(quant_name(type_id)).or_default();
        entry.tensor_count += 1;
        entry.bytes += bytes;

        if let Some(rest) = name.strip_prefix("blk.") {
            let layer: usize = rest.split('.').next().unwrap_or("").parse().unwrap_or(0);
            n_layers = n_layers.max(layer + 1);
            layer_bytes += bytes;
            // attention layers with real context KV: only layers carrying
            // attn_k/attn_v/attn_kv tensors. Fused attn_qkv on hybrid arches
            // (gated-delta path) has fixed-size state, NOT context-scaled KV.
            let lc = rest.to_ascii_lowercase();
            if lc.contains("attn_k.weight")
                || lc.contains("attn_v.weight")
                || lc.contains("attn_kv.weight")
            {
                attn_layers.insert(layer);
            }
        } else {
            non_layer_bytes += bytes;
        }
        let lower = name.to_ascii_lowercase();
        if lower.contains("nextn") || lower.contains("mtp") {
            has_mtp = true;
            mtp_bytes += bytes;
        }
    }

    let g = |k: &str| metadata.get(k).cloned().unwrap_or_default();
    let p = |k: &str| g(k).parse().unwrap_or(0u64);
    let arch = g("general.architecture");
    let name = g("general.name");
    let arch_owned = arch.clone();

    Ok(ModelMeta {
        path: path.display().to_string(),
        file_bytes,
        tensor_count: tensor_count as usize,
        arch,
        name,
        quant_mix,
        total_tensor_bytes,
        total_elements,
        non_layer_bytes,
        layer_bytes,
        n_layers,
        n_attention_layers: attn_layers.len(),
        kv_heads: p("qwen35.attention.head_count_kv")
            .max(p(&format!("{arch_owned}.attention.head_count_kv")))
            .max(p(&format!("{arch_owned}.attention.key_length")).min(1)),
        key_length: p(&format!("{arch_owned}.attention.key_length")),
        value_length: p(&format!("{arch_owned}.attention.value_length")),
        embedding_length: p(&format!("{arch_owned}.embedding_length")),
        head_count: p(&format!("{arch_owned}.attention.head_count")),
        has_mtp,
        mtp_bytes,
        metadata,
    })
}

// ---------------------------------------------------------------------------
// Primitive readers
// ---------------------------------------------------------------------------

fn read_u8(f: &mut impl Read) -> Result<u8, String> {
    let mut b = [0u8; 1];
    f.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(b[0])
}

fn read_u16(f: &mut impl Read) -> Result<u16, String> {
    let mut b = [0u8; 2];
    f.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32(f: &mut impl Read) -> Result<u32, String> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(f: &mut impl Read) -> Result<u64, String> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u64::from_le_bytes(b))
}

fn read_i64(f: &mut impl Read) -> Result<i64, String> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(i64::from_le_bytes(b))
}

fn read_f32(f: &mut impl Read) -> Result<f32, String> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(f32::from_le_bytes(b))
}

fn read_f64(f: &mut impl Read) -> Result<f64, String> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(f64::from_le_bytes(b))
}

fn read_string(f: &mut impl Read) -> Result<String, String> {
    let len = read_u64(f)?;
    let mut buf = vec![0u8; len as usize];
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn read_value(f: &mut impl Read, vt: ValueType) -> Result<String, String> {
    Ok(match vt {
        ValueType::Uint8 => read_u8(f)?.to_string(),
        ValueType::Int8 => (read_u8(f)? as i8).to_string(),
        ValueType::Uint16 => read_u16(f)?.to_string(),
        ValueType::Int16 => (read_u16(f)? as i16).to_string(),
        ValueType::Uint32 => read_u32(f)?.to_string(),
        ValueType::Int32 => (read_u32(f)? as i32).to_string(),
        ValueType::Float32 => read_f32(f)?.to_string(),
        ValueType::Bool => (read_u8(f)? != 0).to_string(),
        ValueType::String => read_string(f)?,
        ValueType::Uint64 => read_u64(f)?.to_string(),
        ValueType::Int64 => read_i64(f)?.to_string(),
        ValueType::Float64 => read_f64(f)?.to_string(),
        ValueType::Array => {
            let elem_type = read_u32(f)?;
            let n = read_u64(f)?;
            let et = ValueType::from_u32(elem_type).unwrap_or(ValueType::Uint8);
            let mut parts = Vec::with_capacity(n.min(16) as usize);
            for i in 0..n {
                let v = read_value(f, et)?;
                if i < 16 {
                    parts.push(v);
                }
            }
            format!(
                "[{}]{}",
                parts.join(","),
                if n > 16 {
                    format!("...(+{})", n - 16)
                } else {
                    String::new()
                }
            )
        }
    })
}

/// Convenience: parse from a path string.
pub fn inspect_path(path: &str) -> Result<ModelMeta, String> {
    inspect(Path::new(path))
}

// ---------------------------------------------------------------------------
// Tests: build a tiny synthetic GGUF in memory and round-trip it
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    fn write_u32(f: &mut Vec<u8>, v: u32) {
        f.extend_from_slice(&v.to_le_bytes());
    }
    fn write_u64(f: &mut Vec<u8>, v: u64) {
        f.extend_from_slice(&v.to_le_bytes());
    }
    fn write_str(f: &mut Vec<u8>, s: &str) {
        write_u64(f, s.len() as u64);
        f.extend_from_slice(s.as_bytes());
    }

    fn write_kv(f: &mut Vec<u8>, key: &str, vtype: u32, val: &[u8]) {
        write_str(f, key);
        write_u32(f, vtype);
        f.extend_from_slice(val);
    }

    fn synth_gguf() -> Vec<u8> {
        let mut f = Vec::new();
        write_u32(&mut f, GGUF_MAGIC);
        write_u32(&mut f, 3);
        write_u64(&mut f, 2); // 2 tensors
        write_u64(&mut f, 3); // 3 kv entries
        // general.architecture = "qwen35"
        let mut v = Vec::new();
        write_str(&mut v, "qwen35");
        write_kv(&mut f, "general.architecture", 8, &v);
        // qwen35.attention.head_count_kv = 4
        write_kv(
            &mut f,
            "qwen35.attention.head_count_kv",
            10,
            &4u64.to_le_bytes(),
        );
        // qwen35.attention.key_length = 256
        write_kv(
            &mut f,
            "qwen35.attention.key_length",
            10,
            &256u64.to_le_bytes(),
        );
        // tensor 1: blk.0.attn_k.weight, 1 dim, 5120 elems, Q6_K (14)
        write_str(&mut f, "blk.0.attn_k.weight");
        write_u32(&mut f, 1);
        write_u64(&mut f, 5120);
        write_u32(&mut f, 14);
        write_u64(&mut f, 0);
        // tensor 2: token_embd.weight, 2 dims 5120x1024, F16 (1)
        write_str(&mut f, "token_embd.weight");
        write_u32(&mut f, 2);
        write_u64(&mut f, 5120);
        write_u64(&mut f, 1024);
        write_u32(&mut f, 1);
        write_u64(&mut f, 0);
        f
    }

    #[test]
    fn parses_synthetic_gguf() {
        let data = synth_gguf();
        let path = std::env::temp_dir().join("lb_model_test.gguf");
        std::fs::write(&path, &data).unwrap();
        let m = inspect(&path).unwrap();
        assert_eq!(m.arch, "qwen35");
        assert_eq!(m.kv_heads, 4);
        assert_eq!(m.key_length, 256);
        assert_eq!(m.n_layers, 1);
        assert_eq!(m.n_attention_layers, 1);
        // Q6_K: 5120 elems / 256 * 210 = 20 * 210 = 4200
        assert_eq!(m.quant_mix["Q6_K"].bytes, 4200);
        assert_eq!(m.total_elements, 5120 + 5120 * 1024);
        // F16: 5120*1024 * 2
        assert_eq!(m.quant_mix["F16"].bytes, 5120 * 1024 * 2);
        assert_eq!(m.non_layer_bytes, 5120 * 1024 * 2);
        assert!(!m.has_mtp);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn rejects_non_gguf() {
        let path = std::env::temp_dir().join("lb_not_gguf.bin");
        std::fs::write(&path, b"not a gguf file at all!!").unwrap();
        assert!(inspect(&path).is_err());
        std::fs::remove_file(path).ok();
    }
}
