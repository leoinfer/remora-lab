//! Native GGUF tensor backend — load real quantized weights from a
//! GGUF file into the native CPU model backends (`--model model.gguf`).
//!
//! Uses `har-model`'s native Rust reader: the
//! tensor directory is parsed, a bounded row span is copied, and the
//! bytes become `Q40Model` (the Q4_0 format) or `Q4KModel`
//! weights. Only the requested rows are read from the caller's model file
//! costs a header parse + a few KB of payload.

use crate::adapter::{BatchStepModel, Hidden, StepOutcome};
use crate::q40::{Q40Model, Q40_BLOCK_BYTES, Q40_BLOCK_VALUES};
use crate::q4k::{Q4KModel, Q4K_BLOCK_BYTES, Q4K_BLOCK_VALUES};
use har_model::GgufReader;
use std::path::Path;

/// GGUF tensor type ids used by the native backends.
///
/// NOTE: `Q4_0 = 2`, not 7 — 7 is `Q5_1`.  The original constant (7) was
/// self-consistent with the synthetic test builder but wrong against
/// external GGUF files; a model may expose `token_embd.weight` as Q4_0
/// exposed it the first time the ignored integration test actually ran.
pub const GGML_TYPE_Q4_0: u32 = 2;
pub const GGML_TYPE_Q4_K: u32 = 12;

/// HAR production migration S1: R4X-D32A (format type id 36). Block layout:
/// 256 values per 144 B block — 8 fp16 scales
/// (d[8]) + 128 B of packed int4 nibbles (qs[8][16]).
pub const GGML_TYPE_R4X_D32A: u32 = 36;
pub const R4X_D32A_BLOCK_VALUES: usize = 256;
pub const R4X_D32A_BLOCK_BYTES: usize = 144;

/// The concrete backend built from a loaded tensor.
#[derive(Clone)]
pub enum LoadedBackend {
    Q40(Q40Model),
    Q4K(Q4KModel),
}

impl BatchStepModel for LoadedBackend {
    fn batch_step(&self, inputs: &[(Hidden, u32)]) -> Vec<StepOutcome> {
        match self {
            Self::Q40(m) => m.batch_step(inputs),
            Self::Q4K(m) => m.batch_step(inputs),
        }
    }
    fn initial_hidden(&self) -> Hidden {
        match self {
            Self::Q40(m) => m.initial_hidden(),
            Self::Q4K(m) => m.initial_hidden(),
        }
    }
    fn eos(&self) -> u32 {
        match self {
            Self::Q40(m) => m.eos(),
            Self::Q4K(m) => m.eos(),
        }
    }
    fn weight_bytes_per_row(&self) -> u64 {
        match self {
            Self::Q40(m) => m.weight_bytes_per_row(),
            Self::Q4K(m) => m.weight_bytes_per_row(),
        }
    }
}

/// One loaded quantized tensor, ready to serve.
pub struct LoadedModel {
    pub tensor_name: String,
    pub ggml_type: u32,
    /// Rows loaded (== the model's vocab).
    pub rows: usize,
    /// Row length in the source tensor (elements).
    pub row_elements: u64,
    /// Bytes read from the file.
    pub payload_bytes: u64,
    pub backend: LoadedBackend,
}

/// Byte layout of one row of the source tensor: Q4_0 = `row_elements/32`
/// blocks × 18 B; Q4_K = `row_elements/256` blocks × 144 B.  The row
/// width is the tensor's own (`row_elements` — the caller's
/// `token_embd.weight` rows are 5120 wide), not a hardcoded 256.
fn window_row_bytes(ggml_type: u32, row_elements: u64) -> Result<usize, String> {
    match ggml_type {
        GGML_TYPE_Q4_0 => {
            assert_eq!(
                row_elements % Q40_BLOCK_VALUES as u64,
                0,
                "Q4_0 row width must be 32-aligned"
            );
            Ok((row_elements as usize / Q40_BLOCK_VALUES) * Q40_BLOCK_BYTES)
        }
        GGML_TYPE_Q4_K => {
            assert_eq!(
                row_elements % Q4K_BLOCK_VALUES as u64,
                0,
                "Q4_K row width must be 256-aligned"
            );
            Ok((row_elements as usize / Q4K_BLOCK_VALUES) * Q4K_BLOCK_BYTES)
        }
        GGML_TYPE_R4X_D32A => {
            // Row-size contract: ceil(n / block_size) blocks. Caller models
            // vocab widths are not always 256-aligned (e.g. 151936 -> 594).
            let blocks = row_elements.div_ceil(R4X_D32A_BLOCK_VALUES as u64);
            Ok(blocks as usize * R4X_D32A_BLOCK_BYTES)
        }
        other => Err(format!(
            "unsupported format type {other} (need Q4_0=2, Q4_K=12, or R4X_D32A=36)"
        )),
    }
}

/// Load the first `rows` rows of a tensor (the tensor's own row width).
pub fn load_rows(
    path: impl AsRef<Path>,
    tensor_name: &str,
    rows: usize,
    seed: u64,
) -> Result<LoadedModel, String> {
    let reader = GgufReader::new(path.as_ref().to_path_buf());
    let phenotype = reader
        .inspect(false)
        .map_err(|e| format!("GGUF inspect {}: {e}", path.as_ref().display()))?;
    let tensor = phenotype
        .tensor(tensor_name)
        .ok_or_else(|| format!("tensor {tensor_name} not in {}", path.as_ref().display()))?;
    let row_elements = tensor.row_elements().unwrap_or(0);
    let row_bytes = window_row_bytes(tensor.ggml_type, row_elements)?;

    if rows == 0 {
        return Err("row count must be non-zero".into());
    }

    let payload_bytes = (rows as u64) * row_bytes as u64;
    let data = reader
        .read_tensor_range(tensor, 0, payload_bytes)
        .map_err(|e| format!("read {rows} rows of {tensor_name}: {e}"))?;
    assert_eq!(data.len() as u64, payload_bytes, "bounded read");

    let eos = (rows - 1) as u32;
    let backend = match tensor.ggml_type {
        GGML_TYPE_Q4_0 => {
            assert_eq!(
                data.len(),
                rows * (row_elements as usize / Q40_BLOCK_VALUES) * Q40_BLOCK_BYTES,
                "Q4_0 payload geometry"
            );
            LoadedBackend::Q40(Q40Model::from_blocks_dim(
                &data,
                rows,
                row_elements as usize,
                eos,
                seed,
            ))
        }
        GGML_TYPE_Q4_K => LoadedBackend::Q4K(Q4KModel::from_blocks_dim(
            &data,
            rows,
            row_elements as usize,
            eos,
            seed,
        )),
        _ => unreachable!("window_row_bytes validated the type"),
    };

    Ok(LoadedModel {
        tensor_name: tensor.name.clone(),
        ggml_type: tensor.ggml_type,
        rows,
        row_elements,
        payload_bytes,
        backend,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::q40::q40_dequant;
    use std::io::Write;

    /// Minimal synthetic GGUF with one Q4_0 tensor
    /// `token_embd.weight` of shape [256, rows] (rows × 144 bytes).
    fn write_synthetic_gguf(path: &Path, rows: usize, blocks: &[u8]) {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&1u64.to_le_bytes()); // metadata_count
                                                    // metadata: general.alignment = 32 (u32)
        let key = b"general.alignment";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&4u32.to_le_bytes()); // GGUF_VALUE_TYPE_UINT32
        buf.extend_from_slice(&32u32.to_le_bytes());
        // tensor info: token_embd.weight, rank 2, dims [256, rows], Q4_0
        let name = b"token_embd.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name);
        buf.extend_from_slice(&2u32.to_le_bytes()); // rank
        buf.extend_from_slice(&256u64.to_le_bytes());
        buf.extend_from_slice(&(rows as u64).to_le_bytes());
        buf.extend_from_slice(&GGML_TYPE_Q4_0.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset
                                                    // pad to 32-byte alignment
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(blocks);
        let mut f = std::fs::File::create(path).expect("create gguf");
        f.write_all(&buf).expect("write gguf");
    }

    #[test]
    fn loads_q40_rows_from_synthetic_gguf() {
        let dir = std::env::temp_dir().join("har-serve-gguf-test");
        std::fs::create_dir_all(&dir).expect("tmpdir");
        let path = dir.join("synthetic-q40.gguf");
        let rows = 8usize;
        // d = 1.0, code bytes with a known pattern: value v -> (v % 16) - 8
        let mut blocks = Vec::new();
        for _ in 0..rows * 8 {
            blocks.push(0x00);
            blocks.push(0x3c); // d = 1.0
            for v in 0..16u8 {
                blocks.push(v); // codes: value 2v -> v-8, value 2v+1 -> v-8
            }
        }
        write_synthetic_gguf(&path, rows, &blocks);

        let loaded = load_rows(&path, "token_embd.weight", rows, 0xABCD).expect("load");
        assert_eq!(loaded.ggml_type, GGML_TYPE_Q4_0);
        assert_eq!(loaded.rows, rows);
        assert_eq!(loaded.payload_bytes as usize, rows * 144);

        // Verify the dequant against the known bytes: row 0, value 0 = -8.
        let LoadedBackend::Q40(model) = &loaded.backend else {
            panic!("expected Q40 backend");
        };
        let row = &model.weights[..144];
        let b0: [u8; Q40_BLOCK_BYTES] = row[..Q40_BLOCK_BYTES].try_into().unwrap();
        assert_eq!(q40_dequant(&b0, 0), -8.0);
        assert_eq!(q40_dequant(&b0, 1), -8.0);
        assert_eq!(q40_dequant(&b0, 14), -1.0, "value 14 -> code v=7 -> -1");
        assert_eq!(q40_dequant(&b0, 28), 6.0, "value 28 -> code v=14 -> +6");

        // The model runs and is deterministic.
        let h = model.initial_hidden();
        let a = model.batch_step(&[(h.clone(), 3)]);
        let b = model.batch_step(&[(h, 3)]);
        assert_eq!(a[0].logits, b[0].logits);
        std::fs::remove_file(&path).ok();
    }

    /// Optional integration: load 8 rows of `token_embd.weight` from a
    /// caller-provided GGUF. Ignored by default; set `HAR_TEST_GGUF` and run
    /// `cargo test -- --ignored` when the file is available locally.
    #[test]
    #[ignore = "requires a caller-provided GGUF"]
    fn loads_caller_supplied_rows() {
        let Some(path) = std::env::var_os("HAR_TEST_GGUF") else {
            eprintln!("HAR_TEST_GGUF is not set; skipping");
            return;
        };
        let path = std::path::PathBuf::from(path);
        if !path.exists() {
            eprintln!("HAR_TEST_GGUF does not exist; skipping");
            return;
        }
        let loaded = load_rows(&path, "token_embd.weight", 8, 0x5EED).expect("load");
        assert_eq!(loaded.ggml_type, GGML_TYPE_Q4_0);
        assert_eq!(loaded.rows, 8);
        assert_eq!(
            loaded.row_elements, 5120,
            "Qwen3-27B token_embd row width (hidden dim)"
        );
        let h = loaded.backend.initial_hidden();
        assert_eq!(h.len(), 5120, "hidden width follows the tensor rows");
        let out = loaded.backend.batch_step(&[(h, 3u32)]);
        assert_eq!(out[0].logits.len(), 8, "one logit per loaded row");
        println!(
            "caller-provided rows OK: tensor={} rows={} row_elements={} payload={}B",
            loaded.tensor_name, loaded.rows, loaded.row_elements, loaded.payload_bytes
        );
    }

    #[test]
    fn captures_tokenizer_metadata_from_synthetic_gguf() {
        // Extend the synthetic builder with tokenizer arrays.
        let dir = std::env::temp_dir().join("har-serve-gguf-test");
        std::fs::create_dir_all(&dir).expect("tmpdir");
        let path = dir.join("synthetic-tok.gguf");
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&5u64.to_le_bytes()); // 5 metadata keys
        let mut put = |key: &str, value_type: u32, payload: &[u8]| {
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            buf.extend_from_slice(&value_type.to_le_bytes());
            buf.extend_from_slice(payload);
        };
        put("general.alignment", 4, &32u32.to_le_bytes());
        put("tokenizer.ggml.model", 8, &{
            let s = b"gpt2";
            let mut v = (s.len() as u64).to_le_bytes().to_vec();
            v.extend_from_slice(s);
            v
        });
        // tokens: array of strings ["h", "e", "hello"]
        let tokens = [b"h".as_slice(), b"e", b"he", b"hello"];
        let mut payload = 8u32.to_le_bytes().to_vec();
        payload.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
        for t in tokens {
            payload.extend_from_slice(&(t.len() as u64).to_le_bytes());
            payload.extend_from_slice(t);
        }
        put("tokenizer.ggml.tokens", 9, &payload);
        // merges: array of strings ["h e"]
        let mut payload = 8u32.to_le_bytes().to_vec();
        payload.extend_from_slice(&1u64.to_le_bytes());
        let m = b"h e";
        payload.extend_from_slice(&(m.len() as u64).to_le_bytes());
        payload.extend_from_slice(m);
        put("tokenizer.ggml.merges", 9, &payload);
        put("tokenizer.ggml.eos_token_id", 4, &0u32.to_le_bytes());
        // tensor info
        let name = b"token_embd.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name);
        buf.extend_from_slice(&2u32.to_le_bytes()); // rank
        buf.extend_from_slice(&256u64.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&2u32.to_le_bytes()); // Q4_0 (format id 2)
        buf.extend_from_slice(&0u64.to_le_bytes());
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(&[0u8; 4 * 144]);
        std::fs::write(&path, &buf).expect("write");

        let reader = har_model::GgufReader::new(path.clone());
        let phenotype = reader.inspect(false).expect("inspect");
        let tok = phenotype.tokenizer.expect("tokenizer captured");
        assert_eq!(tok.model.as_deref(), Some("gpt2"));
        assert_eq!(tok.tokens, vec!["h", "e", "he", "hello"]);
        assert_eq!(tok.merges, vec!["h e"]);
        assert_eq!(tok.eos_token_id, Some(0));

        // And the Tokenizer can encode with it.
        let t = crate::tokenizer::Tokenizer::from_gguf(&tok).expect("tokenizer");
        let ids = t.encode("he").expect("encode");
        assert_eq!(ids.len(), 1, "h e merge applied");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_unknown_tensor_and_type() {
        let dir = std::env::temp_dir().join("har-serve-gguf-test");
        std::fs::create_dir_all(&dir).expect("tmpdir");
        let path = dir.join("synthetic-q4k.gguf");
        // Build a Q4_K-typed tensor to hit the type rejection.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        let key = b"general.alignment";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key);
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&32u32.to_le_bytes());
        let name = b"token_embd.weight";
        buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
        buf.extend_from_slice(name);
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&256u64.to_le_bytes());
        buf.extend_from_slice(&8u64.to_le_bytes());
        buf.extend_from_slice(&GGML_TYPE_Q4_K.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        while buf.len() % 32 != 0 {
            buf.push(0);
        }
        buf.extend_from_slice(&[0u8; 8 * 144]);
        let mut f = std::fs::File::create(&path).expect("create");
        f.write_all(&buf).expect("write");

        assert!(load_rows(&path, "missing.weight", 8, 1).is_err());
        let loaded = load_rows(&path, "token_embd.weight", 8, 1).expect("load Q4_K");
        assert_eq!(loaded.ggml_type, GGML_TYPE_Q4_K);
        assert!(matches!(loaded.backend, LoadedBackend::Q4K(_)));
        std::fs::remove_file(&path).ok();
    }
}

#[cfg(test)]
mod r4x_s1_tests {
    use super::*;

    #[test]
    fn r4x_d32a_row_window_math() {
        // 5120-wide projection row: 20 superblocks x 144 B
        assert_eq!(window_row_bytes(GGML_TYPE_R4X_D32A, 5120).unwrap(), 2880);
        // real vocab width: 593.5 blocks -> ceil = 594 (row-size contract)
        assert_eq!(
            window_row_bytes(GGML_TYPE_R4X_D32A, 151936).unwrap(),
            151936usize.div_ceil(256) * 144
        );
        // partial trailing superblock
        assert_eq!(
            window_row_bytes(GGML_TYPE_R4X_D32A, 151936 + 128).unwrap(),
            (151936usize + 128).div_ceil(256) * 144
        );
    }

    #[test]
    fn r4x_d32a_real_artifact_inspect() {
        // Optional integration: parse a caller-provided transformed artifact
        // and check the R4X row geometry. CI remains offline-safe.
        let Some(p) = std::env::var_os("HAR_TEST_R4X_GGUF") else {
            return;
        };
        let p = std::path::PathBuf::from(p);
        if !p.exists() {
            return;
        }
        let reader = GgufReader::new(p);
        let ph = reader.inspect(false).expect("inspect transformed_v3");
        let mut n36 = 0usize;
        for name in [
            "blk.0.attn_q.weight",
            "blk.0.ffn_gate.weight",
            "output.weight",
        ] {
            if let Some(t) = ph.tensor(name) {
                if t.ggml_type == GGML_TYPE_R4X_D32A {
                    n36 += 1;
                    let re = t.row_elements().unwrap_or(0);
                    assert_eq!(re % 256, 0, "{name} rows must be 256-aligned");
                }
            }
        }
        assert!(n36 > 0, "expected d32a weight tensors in transformed_v3");
    }
}
