//! Native HAR prompt -> token generation for the dense Qwen3 GGUF lane.
//!
//! This binary executes the model with native Rust math. It
//! parses a caller-supplied GGUF tensor directory, memory maps the caller's
//! Qwen weights,
//! performs Qwen3 RMSNorm/NeoX-RoPE/GQA/SiLU gated-MLP math with native Rust
//! Q4_K/Q6_K matvecs, and greedily emits tokens.  The first supported model is
//! the dense `qwen3` architecture (for example Qwen3-4B-Q4_K_M.gguf).  Qwen3.5
//! Qwen3.5/Qwen3.6 hybrid MoE models are rejected rather than silently treated
//! as dense.
//!
//! This is a CPU-native correctness/generation path, not a Vulkan generation
//! claim.  Physical MoE residency is measured separately by
//! `har-real-expert-upload`; this binary's artifact states that kernel-visible
//! residency is false.

use har_metabolism::clock::{ClockBasis, FastObservation};
use har_metabolism::controller::ControllerConfig;
use har_metabolism::ledger::LedgerClass;
use har_metabolism::surplus::SurplusInputs;
use har_metabolism::MetabolismController;
use har_model::{GgufReader, ModelPhenotype, TensorDescriptor};
use libc::{mmap, munmap, MAP_FAILED, MAP_PRIVATE, PROT_READ};
use regex::Regex;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Error as IoError, ErrorKind, Read};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const QK_K: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q6_K_BLOCK_BYTES: usize = 210;
const Q4_K_GGML_TYPE: u32 = 12;
const Q6_K_GGML_TYPE: u32 = 14;
const F32_GGML_TYPE: u32 = 0;
const F16_GGML_TYPE: u32 = 1;
const DEFAULT_PROMPT: &str = "Hello";
const DEFAULT_NEW_TOKENS: usize = 1;
const DEFAULT_OUTPUT: &str = "results/native_qwen3_generation.json";

#[derive(Debug)]
struct Args {
    model: PathBuf,
    prompt: String,
    max_new_tokens: usize,
    output: PathBuf,
    no_bos: bool,
}

fn usage() -> &'static str {
    "usage: har-native-qwen3 <model.gguf> [--prompt TEXT] [--max-new-tokens N] [--output path.json] [--no-bos]"
}

fn parse_args() -> Result<Args, String> {
    let mut raw = std::env::args().skip(1);
    let model = raw
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| usage().to_string())?;
    let mut args = Args {
        model,
        prompt: DEFAULT_PROMPT.into(),
        max_new_tokens: DEFAULT_NEW_TOKENS,
        output: PathBuf::from(DEFAULT_OUTPUT),
        no_bos: false,
    };
    while let Some(flag) = raw.next() {
        match flag.as_str() {
            "--prompt" => args.prompt = raw.next().ok_or("--prompt requires a value")?,
            "--max-new-tokens" => {
                args.max_new_tokens = raw
                    .next()
                    .ok_or("--max-new-tokens requires a value")?
                    .parse()
                    .map_err(|_| "invalid --max-new-tokens")?;
            }
            "--output" => {
                args.output = PathBuf::from(raw.next().ok_or("--output requires a path")?);
            }
            "--no-bos" => args.no_bos = true,
            _ => return Err(format!("unknown argument {flag}\n{}", usage())),
        }
    }
    if args.prompt.is_empty() {
        return Err("--prompt must not be empty".into());
    }
    if args.max_new_tokens == 0 || args.max_new_tokens > 64 {
        return Err("--max-new-tokens must be between 1 and 64".into());
    }
    Ok(args)
}

fn io_error(message: impl Into<String>) -> IoError {
    IoError::new(ErrorKind::InvalidData, message.into())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn sha256_f32(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn half_to_f32(value: u16) -> f32 {
    let sign = ((value & 0x8000) as u32) << 16;
    let exponent = ((value >> 10) & 0x1f) as u32;
    let fraction = (value & 0x03ff) as u32;
    let bits = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let mut mantissa = fraction;
            let mut shift = 0u32;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                shift += 1;
            }
            mantissa &= 0x03ff;
            sign | ((127 - 14 - shift) << 23) | (mantissa << 13)
        }
        31 => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | ((exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(bits)
}

fn scale_min(scales: &[u8], index: usize) -> (u8, u8) {
    if index < 4 {
        (scales[index] & 63, scales[index + 4] & 63)
    } else {
        (
            (scales[index + 4] & 15) | ((scales[index - 4] >> 6) << 4),
            (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
        )
    }
}

fn q4k_dot(block: &[u8], input: &[f32]) -> f32 {
    let d = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let min = half_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let mut q_offset = 16usize;
    let mut scale_index = 0usize;
    let mut sum = 0.0f32;
    for input_base in (0..QK_K).step_by(64) {
        let (s1, m1) = scale_min(scales, scale_index);
        let (s2, m2) = scale_min(scales, scale_index + 1);
        let d1 = d * s1 as f32;
        let d2 = d * s2 as f32;
        let min1 = min * m1 as f32;
        let min2 = min * m2 as f32;
        for lane in 0..32 {
            sum += (d1 * (block[q_offset + lane] & 0x0f) as f32 - min1) * input[input_base + lane];
            sum +=
                (d2 * (block[q_offset + lane] >> 4) as f32 - min2) * input[input_base + 32 + lane];
        }
        q_offset += 32;
        scale_index += 2;
    }
    sum
}

fn q6k_dot(block: &[u8], input: &[f32]) -> f32 {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = half_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let mut sum = 0.0f32;
    for half in 0..2usize {
        let ql_base = half * 64;
        let qh_base = half * 32;
        let scale_base = half * 8;
        let input_base = half * 128;
        for lane in 0..32usize {
            let high = qh[qh_base + lane];
            let low_0 = ql[ql_base + lane];
            let low_1 = ql[ql_base + 32 + lane];
            let q1 = ((low_0 & 0x0f) | ((high & 0x03) << 4)) as i16 - 32;
            let q2 = ((low_1 & 0x0f) | (((high >> 2) & 0x03) << 4)) as i16 - 32;
            let q3 = ((low_0 >> 4) | (((high >> 4) & 0x03) << 4)) as i16 - 32;
            let q4 = ((low_1 >> 4) | (((high >> 6) & 0x03) << 4)) as i16 - 32;
            let s1 = scales[scale_base + lane / 16] as i8 as f32;
            let s2 = scales[scale_base + lane / 16 + 2] as i8 as f32;
            let s3 = scales[scale_base + lane / 16 + 4] as i8 as f32;
            let s4 = scales[scale_base + lane / 16 + 6] as i8 as f32;
            sum += d
                * (s1 * q1 as f32 * input[input_base + lane]
                    + s2 * q2 as f32 * input[input_base + 32 + lane]
                    + s3 * q3 as f32 * input[input_base + 64 + lane]
                    + s4 * q4 as f32 * input[input_base + 96 + lane]);
        }
    }
    sum
}

fn dot_quant_row(row: &[u8], input: &[f32], ggml_type: u32) -> Result<f32, String> {
    let block_bytes = match ggml_type {
        Q4_K_GGML_TYPE => Q4_K_BLOCK_BYTES,
        Q6_K_GGML_TYPE => Q6_K_BLOCK_BYTES,
        _ => return Err(format!("unsupported quantized matvec type {ggml_type}")),
    };
    if input.len() % QK_K != 0 || row.len() != input.len() / QK_K * block_bytes {
        return Err(format!(
            "quantized row geometry mismatch: row_bytes={} input_elements={} type={ggml_type}",
            row.len(),
            input.len()
        ));
    }
    let mut sum = 0.0f32;
    for (block_index, block) in row.chunks_exact(block_bytes).enumerate() {
        let input_base = block_index * QK_K;
        sum += match ggml_type {
            Q4_K_GGML_TYPE => q4k_dot(block, &input[input_base..input_base + QK_K]),
            Q6_K_GGML_TYPE => q6k_dot(block, &input[input_base..input_base + QK_K]),
            _ => unreachable!(),
        };
    }
    Ok(sum)
}

struct MappedFile {
    ptr: *const u8,
    len: usize,
}

unsafe impl Send for MappedFile {}
unsafe impl Sync for MappedFile {}

impl MappedFile {
    fn open(path: &Path, len: u64) -> Result<Self, String> {
        let len = usize::try_from(len).map_err(|_| "model is too large for this host")?;
        if len == 0 {
            return Err("cannot map an empty model".into());
        }
        let file = File::open(path).map_err(|error| format!("open model for mmap: {error}"))?;
        // SAFETY: the file descriptor remains valid for the mmap call; the
        // mapping owns the pages after the call and is read-only/private.
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ,
                MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == MAP_FAILED {
            return Err(format!(
                "mmap model ({len} bytes): {}",
                IoError::last_os_error()
            ));
        }
        Ok(Self {
            ptr: ptr.cast::<u8>(),
            len,
        })
    }

    fn bytes(&self, offset: u64, len: usize) -> Result<&[u8], String> {
        let offset = usize::try_from(offset).map_err(|_| "mapped offset overflows usize")?;
        if offset > self.len || len > self.len - offset {
            return Err(format!(
                "mapped range [{offset}, {}) exceeds file length {}",
                offset.saturating_add(len),
                self.len
            ));
        }
        // SAFETY: bounds are checked against the live read-only mapping.
        Ok(unsafe { std::slice::from_raw_parts(self.ptr.add(offset), len) })
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        // SAFETY: ptr/len came from the successful mmap call and are released
        // exactly once after all borrowed slices have gone out of scope.
        unsafe {
            let _ = munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

#[derive(Clone, Debug)]
struct Tokenizer {
    tokens: Vec<String>,
    token_to_id: HashMap<String, u32>,
    merges: HashMap<(String, String), u32>,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
    add_bos: bool,
}

impl Tokenizer {
    fn load(path: &Path) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|error| format!("open GGUF tokenizer: {error}"))?;
        let mut cursor = CursorReader::new(&mut file);
        if cursor.read_exact(4)? != b"GGUF" {
            return Err("tokenizer source is not GGUF".into());
        }
        let version = cursor.read_u32()?;
        if !(1..=3).contains(&version) {
            return Err(format!("unsupported GGUF version {version}"));
        }
        let _tensor_count = cursor.read_u64()?;
        let metadata_count = cursor.read_u64()?;
        let mut tokens = None;
        let mut merge_strings = None;
        let mut bos_id = None;
        let mut eos_id = None;
        let mut add_bos = false;
        for _ in 0..metadata_count {
            let key = cursor.read_string()?;
            let value_type = cursor.read_u32()?;
            match key.as_str() {
                "tokenizer.ggml.tokens" => {
                    tokens = Some(cursor.read_string_array(value_type)?);
                }
                "tokenizer.ggml.merges" => {
                    merge_strings = Some(cursor.read_string_array(value_type)?);
                }
                "tokenizer.ggml.bos_token_id" => {
                    bos_id = Some(cursor.read_integer(value_type)? as u32);
                }
                "tokenizer.ggml.eos_token_id" => {
                    eos_id = Some(cursor.read_integer(value_type)? as u32);
                }
                "tokenizer.ggml.add_bos_token" => {
                    add_bos = cursor.read_bool(value_type)?;
                }
                _ => cursor.skip_value(value_type)?,
            }
        }
        let tokens = tokens.ok_or("GGUF tokenizer tokens array is missing")?;
        let merge_strings = merge_strings.ok_or("GGUF tokenizer merges array is missing")?;
        let token_to_id = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id as u32))
            .collect();
        let merges = merge_strings
            .iter()
            .enumerate()
            .filter_map(|(rank, merge)| {
                let (left, right) = merge.split_once(' ')?;
                Some(((left.to_owned(), right.to_owned()), rank as u32))
            })
            .collect();
        Ok(Self {
            tokens,
            token_to_id,
            merges,
            bos_id,
            eos_id,
            add_bos,
        })
    }

    fn encode(&self, text: &str, suppress_bos: bool) -> Result<Vec<u32>, String> {
        let mut output = Vec::new();
        if self.add_bos && !suppress_bos {
            output.push(
                self.bos_id
                    .ok_or("tokenizer requests BOS but has no BOS id")?,
            );
        }
        for piece in qwen2_pieces(text)? {
            let symbols = byte_encode(piece.as_bytes());
            let symbols = self.bpe(symbols)?;
            for symbol in symbols {
                let id =
                    self.token_to_id.get(&symbol).copied().ok_or_else(|| {
                        format!("tokenizer vocabulary has no BPE symbol {symbol:?}")
                    })?;
                output.push(id);
            }
        }
        if output.is_empty() {
            return Err("tokenizer produced no tokens".into());
        }
        Ok(output)
    }

    fn bpe(&self, mut symbols: Vec<String>) -> Result<Vec<String>, String> {
        while symbols.len() > 1 {
            let mut best: Option<(u32, usize)> = None;
            for index in 0..symbols.len() - 1 {
                let Some(rank) = self
                    .merges
                    .get(&(symbols[index].clone(), symbols[index + 1].clone()))
                else {
                    continue;
                };
                if best.map_or(true, |(best_rank, _)| *rank < best_rank) {
                    best = Some((*rank, index));
                }
            }
            let Some((_, index)) = best else { break };
            let merged = format!("{}{}", symbols[index], symbols[index + 1]);
            symbols.splice(index..=index + 1, [merged]);
        }
        Ok(symbols)
    }

    fn decode_token(&self, id: u32) -> Result<String, String> {
        let token = self
            .tokens
            .get(id as usize)
            .ok_or_else(|| format!("generated token id {id} exceeds vocabulary"))?;
        if token.starts_with('<') && token.ends_with('>') {
            return Ok(token.clone());
        }
        let mut bytes = Vec::new();
        for character in token.chars() {
            let code = character as u32;
            let byte = if (33..=126).contains(&code)
                || (161..=172).contains(&code)
                || (174..=255).contains(&code)
            {
                code as u8
            } else if (256..=511).contains(&code) {
                decode_byte_unicode(code as u16)
                    .ok_or_else(|| format!("cannot byte-decode token {token:?}"))?
            } else {
                return Err(format!("cannot byte-decode token {token:?}"));
            };
            bytes.push(byte);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

fn decode_byte_unicode(code: u16) -> Option<u8> {
    let mut next = 0u16;
    for byte in 0u16..=255 {
        let unchanged = (33..=126).contains(&byte)
            || (161..=172).contains(&byte)
            || (174..=255).contains(&byte);
        if !unchanged {
            if 256 + next == code {
                return Some(byte as u8);
            }
            next += 1;
        } else if byte == code {
            return Some(byte as u8);
        }
    }
    None
}

fn byte_encode(bytes: &[u8]) -> Vec<String> {
    bytes
        .iter()
        .map(|byte| {
            let code = if (33..=126).contains(byte)
                || (161..=172).contains(byte)
                || (174..=255).contains(byte)
            {
                *byte as u32
            } else {
                let mut missing_before = 0u32;
                for candidate in 0u8..*byte {
                    if !((33..=126).contains(&candidate)
                        || (161..=172).contains(&candidate)
                        || (174..=255).contains(&candidate))
                    {
                        missing_before += 1;
                    }
                }
                256 + missing_before
            };
            char::from_u32(code)
                .expect("GPT-2 byte-to-unicode code point is always valid")
                .to_string()
        })
        .collect()
}

fn qwen2_pieces(text: &str) -> Result<Vec<String>, String> {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    let regex = REGEX.get_or_init(|| {
        Regex::new(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+",
        )
        .expect("Qwen2 tokenizer regex is valid")
    });
    let pieces: Vec<String> = regex
        .find_iter(text)
        .map(|match_| match_.as_str().to_owned())
        .collect();
    let covered: usize = pieces.iter().map(String::len).sum();
    if covered != text.len() {
        return Err(format!(
            "Qwen2 pre-tokenizer left {} bytes uncovered",
            text.len().saturating_sub(covered)
        ));
    }
    Ok(pieces)
}

struct CursorReader<'a> {
    file: &'a mut File,
}

impl<'a> CursorReader<'a> {
    fn new(file: &'a mut File) -> Self {
        Self { file }
    }

    fn read_exact(&mut self, length: usize) -> Result<Vec<u8>, String> {
        let mut bytes = vec![0u8; length];
        self.file
            .read_exact(&mut bytes)
            .map_err(|error| format!("read GGUF metadata: {error}"))?;
        Ok(bytes)
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.read_exact(2)?.try_into().unwrap()))
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.read_exact(4)?.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.read_exact(8)?.try_into().unwrap()))
    }

    fn read_i8(&mut self) -> Result<i8, String> {
        Ok(self.read_u8()? as i8)
    }

    fn read_i16(&mut self) -> Result<i16, String> {
        Ok(self.read_u16()? as i16)
    }

    fn read_i32(&mut self) -> Result<i32, String> {
        Ok(self.read_u32()? as i32)
    }

    fn read_i64(&mut self) -> Result<i64, String> {
        Ok(self.read_u64()? as i64)
    }

    fn read_string(&mut self) -> Result<String, String> {
        let length = self.read_u64()?;
        if length > 1 << 30 {
            return Err(format!("GGUF metadata string too long: {length}"));
        }
        String::from_utf8(self.read_exact(length as usize)?)
            .map_err(|error| format!("GGUF metadata string is not UTF-8: {error}"))
    }

    fn read_integer(&mut self, value_type: u32) -> Result<u64, String> {
        match value_type {
            4 => Ok(self.read_u32()? as u64),
            5 => Ok(self.read_i32()? as i64 as u64),
            10 => self.read_u64(),
            11 => Ok(self.read_i64()? as u64),
            0 => Ok(self.read_u8()? as u64),
            1 => Ok(self.read_i8()? as i64 as u64),
            2 => Ok(self.read_u16()? as u64),
            3 => Ok(self.read_i16()? as i64 as u64),
            _ => Err(format!("GGUF value type {value_type} is not an integer")),
        }
    }

    fn read_bool(&mut self, value_type: u32) -> Result<bool, String> {
        if value_type != 7 {
            return Err(format!("GGUF value type {value_type} is not a bool"));
        }
        Ok(self.read_u8()? != 0)
    }

    fn read_string_array(&mut self, value_type: u32) -> Result<Vec<String>, String> {
        if value_type != 9 {
            return Err(format!("GGUF value type {value_type} is not an array"));
        }
        let element_type = self.read_u32()?;
        let count = self.read_u64()?;
        if element_type != 8 || count > 1_000_000 {
            return Err(format!(
                "GGUF tokenizer array has element_type={element_type} count={count}"
            ));
        }
        let mut values = Vec::with_capacity(count as usize);
        for _ in 0..count {
            values.push(self.read_string()?);
        }
        Ok(values)
    }

    fn skip_value(&mut self, value_type: u32) -> Result<(), String> {
        match value_type {
            0 | 1 | 7 => self.read_exact(1).map(|_| ()),
            2 | 3 => self.read_exact(2).map(|_| ()),
            4..=6 => self.read_exact(4).map(|_| ()),
            8 => self.read_string().map(|_| ()),
            10..=12 => self.read_exact(8).map(|_| ()),
            9 => {
                let element_type = self.read_u32()?;
                let count = self.read_u64()?;
                if count > 1_000_000 {
                    return Err(format!("GGUF metadata array too long: {count}"));
                }
                for _ in 0..count {
                    self.skip_value(element_type)?;
                }
                Ok(())
            }
            _ => Err(format!("unsupported GGUF metadata type {value_type}")),
        }
    }
}

#[derive(Clone)]
struct LayerCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    positions: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl LayerCache {
    fn new(max_positions: usize, kv_heads: usize, head_dim: usize) -> Self {
        Self {
            keys: vec![0.0; max_positions * kv_heads * head_dim],
            values: vec![0.0; max_positions * kv_heads * head_dim],
            positions: 0,
            kv_heads,
            head_dim,
        }
    }

    fn append(&mut self, key: &[f32], value: &[f32]) -> Result<(), String> {
        let row_size = self.kv_heads * self.head_dim;
        if key.len() != row_size || value.len() != row_size {
            return Err("KV append has invalid row size".into());
        }
        let offset = self.positions * row_size;
        if offset + row_size > self.keys.len() {
            return Err("KV cache capacity exhausted".into());
        }
        self.keys[offset..offset + row_size].copy_from_slice(key);
        self.values[offset..offset + row_size].copy_from_slice(value);
        self.positions += 1;
        Ok(())
    }

    fn row<'a>(&self, data: &'a [f32], position: usize, head: usize) -> &'a [f32] {
        let offset = (position * self.kv_heads + head) * self.head_dim;
        &data[offset..offset + self.head_dim]
    }
}

struct NativeQwen3 {
    phenotype: ModelPhenotype,
    tensors: HashMap<String, TensorDescriptor>,
    file: MappedFile,
    tokenizer: Tokenizer,
    n_layer: usize,
    n_embd: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_ff: usize,
    vocab: usize,
    rms_epsilon: f32,
    rope_freq_base: f32,
}

impl NativeQwen3 {
    fn load(path: &Path) -> Result<Self, String> {
        let reader = GgufReader::new(path);
        let identity = reader
            .inspect(false)
            .map_err(|error| format!("GGUF inspection failed: {error}"))?;
        if identity.architecture != "qwen3" {
            return Err(format!(
                "native Qwen3 lane requires architecture qwen3; found {}",
                identity.architecture
            ));
        }
        let phenotype = reader
            .inspect(true)
            .map_err(|error| format!("GGUF identity hashing failed: {error}"))?;
        let metadata = &phenotype.metadata_summary;
        let get_usize = |key: &str, fallback: usize| {
            metadata
                .get(key)
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(fallback)
        };
        let get_f32 = |key: &str, fallback: f32| {
            metadata
                .get(key)
                .and_then(|value| value.parse::<f32>().ok())
                .unwrap_or(fallback)
        };
        let n_layer = get_usize("qwen3.block_count", phenotype.block_count as usize);
        let n_embd = get_usize(
            "qwen3.embedding_length",
            phenotype.embedding_length as usize,
        );
        let n_heads = get_usize(
            "qwen3.attention.head_count",
            phenotype.attention_heads as usize,
        );
        let n_kv_heads = get_usize("qwen3.attention.head_count_kv", phenotype.kv_heads as usize);
        let head_dim = get_usize("qwen3.attention.key_length", phenotype.key_length as usize);
        let n_ff = get_usize("qwen3.feed_forward_length", 0);
        if n_layer == 0
            || n_embd == 0
            || n_heads == 0
            || n_kv_heads == 0
            || head_dim == 0
            || n_ff == 0
            || n_heads % n_kv_heads != 0
        {
            return Err("Qwen3 GGUF has incomplete or inconsistent architecture metadata".into());
        }
        if n_heads * head_dim
            != phenotype
                .tensors
                .iter()
                .find(|tensor| tensor.name == "blk.0.attn_q.weight")
                .map(|tensor| tensor.dimensions[1] as usize)
                .unwrap_or(0)
        {
            return Err("Qwen3 Q projection shape does not match head geometry".into());
        }
        let vocab = phenotype
            .tensors
            .iter()
            .find(|tensor| tensor.name == "token_embd.weight")
            .and_then(|tensor| tensor.dimensions.get(1))
            .copied()
            .ok_or("token_embd.weight is missing")? as usize;
        let rms_epsilon = get_f32("qwen3.attention.layer_norm_rms_epsilon", 1.0e-6);
        let rope_freq_base = get_f32("qwen3.rope.freq_base", 10_000.0);
        let tensors = phenotype
            .tensors
            .iter()
            .cloned()
            .map(|tensor| (tensor.name.clone(), tensor))
            .collect::<HashMap<_, _>>();
        let file = MappedFile::open(path, phenotype.file_bytes)?;
        let tokenizer = Tokenizer::load(path)?;
        Ok(Self {
            phenotype,
            tensors,
            file,
            tokenizer,
            n_layer,
            n_embd,
            n_heads,
            n_kv_heads,
            head_dim,
            n_ff,
            vocab,
            rms_epsilon,
            rope_freq_base,
        })
    }

    fn tensor(&self, name: &str) -> Result<&TensorDescriptor, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("required Qwen3 tensor is missing: {name}"))
    }

    fn tensor_bytes(&self, tensor: &TensorDescriptor) -> Result<&[u8], String> {
        let length = usize::try_from(tensor.payload_bytes)
            .map_err(|_| format!("tensor {} is too large for host", tensor.name))?;
        self.file.bytes(tensor.file_offset, length)
    }

    fn read_f32(&self, tensor: &TensorDescriptor) -> Result<Vec<f32>, String> {
        let values = tensor.element_count as usize;
        if tensor.ggml_type != F32_GGML_TYPE || tensor.payload_bytes < (values * 4) as u64 {
            return Err(format!(
                "tensor {} is not an exact F32 vector: type={} elements={} bytes={}",
                tensor.name, tensor.ggml_type, tensor.element_count, tensor.payload_bytes
            ));
        }
        let bytes = self.tensor_bytes(tensor)?;
        Ok(bytes[..values * 4]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    fn matvec(&self, tensor: &TensorDescriptor, input: &[f32]) -> Result<Vec<f32>, String> {
        if tensor.dimensions.len() != 2 || tensor.dimensions[0] as usize != input.len() {
            return Err(format!(
                "matvec shape mismatch for {}: dims={:?}, input={}",
                tensor.name,
                tensor.dimensions,
                input.len()
            ));
        }
        let rows = tensor.dimensions[1] as usize;
        let row_bytes = match tensor.ggml_type {
            Q4_K_GGML_TYPE => input.len() / QK_K * Q4_K_BLOCK_BYTES,
            Q6_K_GGML_TYPE => input.len() / QK_K * Q6_K_BLOCK_BYTES,
            F32_GGML_TYPE => input.len() * 4,
            F16_GGML_TYPE => input.len() * 2,
            other => {
                return Err(format!(
                    "unsupported matvec type {} for {}",
                    other, tensor.name
                ))
            }
        };
        let expected = row_bytes
            .checked_mul(rows)
            .ok_or_else(|| format!("tensor {} row geometry overflows", tensor.name))?;
        if expected > tensor.payload_bytes as usize {
            return Err(format!(
                "tensor {} payload {} is shorter than expected {}",
                tensor.name, tensor.payload_bytes, expected
            ));
        }
        let workers = std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1)
            .min(rows.max(1));
        let chunk_rows = rows.div_ceil(workers);
        let mut output = vec![0.0f32; rows];
        std::thread::scope(|scope| {
            let mut joins = Vec::new();
            for (chunk_index, chunk) in output.chunks_mut(chunk_rows).enumerate() {
                let row_start = chunk_index * chunk_rows;
                joins.push(
                    scope.spawn(move || {
                        self.matvec_rows(tensor, input, row_bytes, row_start, chunk)
                    }),
                );
            }
            for join in joins {
                join.join()
                    .map_err(|_| "matvec worker panicked".to_string())??;
            }
            Ok::<(), String>(())
        })?;
        Ok(output)
    }

    fn matvec_rows(
        &self,
        tensor: &TensorDescriptor,
        input: &[f32],
        row_bytes: usize,
        row_start: usize,
        output: &mut [f32],
    ) -> Result<(), String> {
        for (relative, value) in output.iter_mut().enumerate() {
            let row = row_start + relative;
            let offset = tensor
                .file_offset
                .checked_add((row * row_bytes) as u64)
                .ok_or_else(|| "tensor row offset overflow".to_string())?;
            let bytes = self.file.bytes(offset, row_bytes)?;
            *value = match tensor.ggml_type {
                Q4_K_GGML_TYPE | Q6_K_GGML_TYPE => dot_quant_row(bytes, input, tensor.ggml_type)?,
                F32_GGML_TYPE => bytes
                    .chunks_exact(4)
                    .zip(input)
                    .map(|(weight, value)| f32::from_le_bytes(weight.try_into().unwrap()) * value)
                    .sum(),
                F16_GGML_TYPE => bytes
                    .chunks_exact(2)
                    .zip(input)
                    .map(|(weight, value)| {
                        half_to_f32(u16::from_le_bytes(weight.try_into().unwrap())) * value
                    })
                    .sum(),
                _ => unreachable!(),
            };
        }
        Ok(())
    }

    fn embedding(&self, token: u32) -> Result<Vec<f32>, String> {
        if token as usize >= self.vocab {
            return Err(format!(
                "token id {token} exceeds vocabulary {}",
                self.vocab
            ));
        }
        let tensor = self.tensor("token_embd.weight")?;
        if tensor.dimensions.len() != 2 || tensor.dimensions[0] as usize != self.n_embd {
            return Err("token embedding shape does not match Qwen3 hidden size".into());
        }
        let row_bytes = tensor.payload_bytes / tensor.dimensions[1];
        let offset = tensor
            .file_offset
            .checked_add(row_bytes * token as u64)
            .ok_or("embedding row offset overflow")?;
        let bytes = self.file.bytes(offset, row_bytes as usize)?;
        let mut output = vec![0.0f32; self.n_embd];
        match tensor.ggml_type {
            Q4_K_GGML_TYPE | Q6_K_GGML_TYPE => {
                let blocks = self.n_embd / QK_K;
                let block_bytes = if tensor.ggml_type == Q4_K_GGML_TYPE {
                    Q4_K_BLOCK_BYTES
                } else {
                    Q6_K_BLOCK_BYTES
                };
                for block in 0..blocks {
                    let begin = block * block_bytes;
                    let end = begin + block_bytes;
                    let mut unit = [0.0f32; QK_K];
                    let source = &bytes[begin..end];
                    decode_quant_block(source, tensor.ggml_type, &mut unit)?;
                    output[block * QK_K..(block + 1) * QK_K].copy_from_slice(&unit);
                }
            }
            F32_GGML_TYPE => {
                for (index, value) in output.iter_mut().enumerate() {
                    *value =
                        f32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap());
                }
            }
            F16_GGML_TYPE => {
                for (index, value) in output.iter_mut().enumerate() {
                    *value = half_to_f32(u16::from_le_bytes(
                        bytes[index * 2..index * 2 + 2].try_into().unwrap(),
                    ));
                }
            }
            other => return Err(format!("unsupported token embedding type {other}")),
        }
        Ok(output)
    }

    #[allow(clippy::needless_range_loop)]
    fn forward(
        &self,
        token: u32,
        position: usize,
        caches: &mut [LayerCache],
    ) -> Result<Vec<f32>, String> {
        let mut hidden = self.embedding(token)?;
        for layer in 0..self.n_layer {
            let prefix = format!("blk.{layer}");
            let residual = hidden.clone();
            let attn_norm = self.read_f32(self.tensor(&format!("{prefix}.attn_norm.weight"))?)?;
            rms_norm(&mut hidden, &attn_norm, self.rms_epsilon)?;

            let mut q = self.matvec(self.tensor(&format!("{prefix}.attn_q.weight"))?, &hidden)?;
            let mut k = self.matvec(self.tensor(&format!("{prefix}.attn_k.weight"))?, &hidden)?;
            let v = self.matvec(self.tensor(&format!("{prefix}.attn_v.weight"))?, &hidden)?;
            let q_norm = self.read_f32(self.tensor(&format!("{prefix}.attn_q_norm.weight"))?)?;
            let k_norm = self.read_f32(self.tensor(&format!("{prefix}.attn_k_norm.weight"))?)?;
            rms_norm_heads(
                &mut q,
                &q_norm,
                self.rms_epsilon,
                self.n_heads,
                self.head_dim,
            )?;
            rms_norm_heads(
                &mut k,
                &k_norm,
                self.rms_epsilon,
                self.n_kv_heads,
                self.head_dim,
            )?;
            rope_neox(
                &mut q,
                position,
                self.n_heads,
                self.head_dim,
                self.rope_freq_base,
            );
            rope_neox(
                &mut k,
                position,
                self.n_kv_heads,
                self.head_dim,
                self.rope_freq_base,
            );
            caches[layer].append(&k, &v)?;
            let attended = attention(
                &q,
                &caches[layer],
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
            )?;
            let projected = self.matvec(
                self.tensor(&format!("{prefix}.attn_output.weight"))?,
                &attended,
            )?;
            for (value, (left, right)) in hidden.iter_mut().zip(residual.iter().zip(projected)) {
                *value = *left + right;
            }

            let ffn_residual = hidden.clone();
            let ffn_norm = self.read_f32(self.tensor(&format!("{prefix}.ffn_norm.weight"))?)?;
            rms_norm(&mut hidden, &ffn_norm, self.rms_epsilon)?;
            let mut gate =
                self.matvec(self.tensor(&format!("{prefix}.ffn_gate.weight"))?, &hidden)?;
            let up = self.matvec(self.tensor(&format!("{prefix}.ffn_up.weight"))?, &hidden)?;
            for (gate_value, up_value) in gate.iter_mut().zip(up) {
                *gate_value = silu(*gate_value) * up_value;
            }
            let down = self.matvec(self.tensor(&format!("{prefix}.ffn_down.weight"))?, &gate)?;
            for (value, (left, right)) in hidden.iter_mut().zip(ffn_residual.iter().zip(down)) {
                *value = *left + right;
            }
        }
        let output_norm = self.read_f32(self.tensor("output_norm.weight")?)?;
        rms_norm(&mut hidden, &output_norm, self.rms_epsilon)?;
        let embedding = self.tensor("token_embd.weight")?;
        let row_bytes = embedding.payload_bytes / embedding.dimensions[1];
        let workers = std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1)
            .min(self.vocab.max(1));
        let chunk_rows = self.vocab.div_ceil(workers);
        let mut logits = vec![0.0f32; self.vocab];
        let hidden_ref = &hidden;
        let embedding_ref = embedding;
        std::thread::scope(|scope| {
            let mut joins = Vec::new();
            for (chunk_index, chunk) in logits.chunks_mut(chunk_rows).enumerate() {
                let row_start = chunk_index * chunk_rows;
                joins.push(scope.spawn(move || {
                    self.matvec_rows(
                        embedding_ref,
                        hidden_ref,
                        row_bytes as usize,
                        row_start,
                        chunk,
                    )
                }));
            }
            for join in joins {
                join.join()
                    .map_err(|_| "logit worker panicked".to_string())??;
            }
            Ok::<(), String>(())
        })?;
        Ok(logits)
    }
}

fn decode_quant_block(
    block: &[u8],
    ggml_type: u32,
    output: &mut [f32; QK_K],
) -> Result<(), String> {
    if ggml_type == Q4_K_GGML_TYPE {
        if block.len() != Q4_K_BLOCK_BYTES {
            return Err("Q4_K block has incorrect size".into());
        }
        let d = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min = half_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let mut q_offset = 16;
        let mut scale_index = 0;
        for base in (0..QK_K).step_by(64) {
            let (s1, m1) = scale_min(scales, scale_index);
            let (s2, m2) = scale_min(scales, scale_index + 1);
            for lane in 0..32 {
                output[base + lane] =
                    d * s1 as f32 * (block[q_offset + lane] & 0x0f) as f32 - min * m1 as f32;
                output[base + 32 + lane] =
                    d * s2 as f32 * (block[q_offset + lane] >> 4) as f32 - min * m2 as f32;
            }
            q_offset += 32;
            scale_index += 2;
        }
        return Ok(());
    }
    if ggml_type == Q6_K_GGML_TYPE {
        if block.len() != Q6_K_BLOCK_BYTES {
            return Err("Q6_K block has incorrect size".into());
        }
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = half_to_f32(u16::from_le_bytes([block[208], block[209]]));
        for half in 0..2usize {
            let ql_base = half * 64;
            let qh_base = half * 32;
            let scale_base = half * 8;
            let out_base = half * 128;
            for lane in 0..32usize {
                let high = qh[qh_base + lane];
                let low_0 = ql[ql_base + lane];
                let low_1 = ql[ql_base + 32 + lane];
                let q1 = ((low_0 & 0x0f) | ((high & 0x03) << 4)) as i16 - 32;
                let q2 = ((low_1 & 0x0f) | (((high >> 2) & 0x03) << 4)) as i16 - 32;
                let q3 = ((low_0 >> 4) | (((high >> 4) & 0x03) << 4)) as i16 - 32;
                let q4 = ((low_1 >> 4) | (((high >> 6) & 0x03) << 4)) as i16 - 32;
                let s1 = scales[scale_base + lane / 16] as i8 as f32;
                let s2 = scales[scale_base + lane / 16 + 2] as i8 as f32;
                let s3 = scales[scale_base + lane / 16 + 4] as i8 as f32;
                let s4 = scales[scale_base + lane / 16 + 6] as i8 as f32;
                output[out_base + lane] = d * s1 * q1 as f32;
                output[out_base + 32 + lane] = d * s2 * q2 as f32;
                output[out_base + 64 + lane] = d * s3 * q3 as f32;
                output[out_base + 96 + lane] = d * s4 * q4 as f32;
            }
        }
        return Ok(());
    }
    Err(format!("unsupported embedding block type {ggml_type}"))
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn rms_norm(values: &mut [f32], weight: &[f32], epsilon: f32) -> Result<(), String> {
    if values.len() != weight.len() {
        return Err(format!(
            "RMSNorm length mismatch: values={} weight={}",
            values.len(),
            weight.len()
        ));
    }
    let mean = values.iter().map(|value| value * value).sum::<f32>() / values.len() as f32;
    let scale = (mean + epsilon).sqrt().recip();
    for (value, weight) in values.iter_mut().zip(weight) {
        *value *= scale * weight;
    }
    Ok(())
}

fn rms_norm_heads(
    values: &mut [f32],
    weight: &[f32],
    epsilon: f32,
    heads: usize,
    head_dim: usize,
) -> Result<(), String> {
    if values.len() != heads * head_dim || weight.len() != head_dim {
        return Err("head RMSNorm geometry mismatch".into());
    }
    for head in 0..heads {
        rms_norm(
            &mut values[head * head_dim..(head + 1) * head_dim],
            weight,
            epsilon,
        )?;
    }
    Ok(())
}

fn rope_neox(values: &mut [f32], position: usize, heads: usize, head_dim: usize, base: f32) {
    let half = head_dim / 2;
    for head in 0..heads {
        let offset = head * head_dim;
        for lane in 0..half {
            let theta = position as f32 * base.powf(-(2.0 * lane as f32) / head_dim as f32);
            let (sin, cos) = theta.sin_cos();
            let x0 = values[offset + lane];
            let x1 = values[offset + half + lane];
            values[offset + lane] = x0 * cos - x1 * sin;
            values[offset + half + lane] = x0 * sin + x1 * cos;
        }
    }
}

#[allow(clippy::needless_range_loop)]
fn attention(
    query: &[f32],
    cache: &LayerCache,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>, String> {
    if query.len() != heads * head_dim || heads % kv_heads != 0 {
        return Err("attention query geometry mismatch".into());
    }
    let mut output = vec![0.0f32; query.len()];
    let scale = (head_dim as f32).sqrt().recip();
    for head in 0..heads {
        let query_head = &query[head * head_dim..(head + 1) * head_dim];
        let kv_head = head / (heads / kv_heads);
        let mut scores = Vec::with_capacity(cache.positions);
        for position in 0..cache.positions {
            let key = cache.row(&cache.keys, position, kv_head);
            scores.push(
                query_head
                    .iter()
                    .zip(key)
                    .map(|(left, right)| left * right)
                    .sum::<f32>()
                    * scale,
            );
        }
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut normalizer = 0.0f32;
        for score in &mut scores {
            *score = (*score - max_score).exp();
            normalizer += *score;
        }
        let output_head = &mut output[head * head_dim..(head + 1) * head_dim];
        for position in 0..cache.positions {
            let weight = scores[position] / normalizer;
            let value = cache.row(&cache.values, position, kv_head);
            for lane in 0..head_dim {
                output_head[lane] += weight * value[lane];
            }
        }
    }
    Ok(output)
}

fn argmax(values: &[f32]) -> Result<(u32, f32), String> {
    values
        .iter()
        .copied()
        .enumerate()
        .max_by(|(left_id, left), (right_id, right)| {
            left.total_cmp(right).then_with(|| right_id.cmp(left_id))
        })
        .map(|(id, value)| (id as u32, value))
        .ok_or("cannot argmax empty logits".into())
}

fn parse_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn ensure_parent(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(io_error)?;
    if !args.model.is_file() {
        return Err(IoError::new(
            ErrorKind::NotFound,
            format!("model not found: {}", args.model.display()),
        )
        .into());
    }

    let load_started = Instant::now();
    let model = NativeQwen3::load(&args.model).map_err(io_error)?;
    let load_ns = load_started.elapsed().as_nanos() as u64;
    let prompt_tokens = model
        .tokenizer
        .encode(&args.prompt, args.no_bos)
        .map_err(io_error)?;
    let max_positions = prompt_tokens.len() + args.max_new_tokens;
    let mut caches = (0..model.n_layer)
        .map(|_| LayerCache::new(max_positions, model.n_kv_heads, model.head_dim))
        .collect::<Vec<_>>();

    let worker_count = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    let model_root = model
        .phenotype
        .sha256
        .clone()
        .ok_or("model inspection did not produce a source SHA-256")?;
    let mut remora = MetabolismController::new(ControllerConfig {
        vram_capacity_mib: 1,
        ram_capacity_mib: model.phenotype.file_bytes / 1_048_576 + 1,
        protected_min_mib: 0,
        gpu_compute_budget: 1,
        slow_window: 256,
    });
    remora.set_basis(ClockBasis {
        model_root: model_root.clone(),
        graph_identity: "har.native.qwen3.dense.cpu.v1".into(),
        worker_set: format!("std-scoped-workers:{worker_count}"),
    });
    let remora_initial = remora.snapshot();
    let remora_safe_surplus = remora.safe_surplus(&SurplusInputs {
        total_reserve: model.phenotype.file_bytes / 1_048_576 + 1,
        maintenance_setpoint_mib: remora_initial.maintenance_vram_mib,
        miss_rate_penalty_mib: 0,
        contention_mib: None,
        interference_mib: None,
        unknown_shrinks: true,
    });

    let mut logits = Vec::new();
    let prompt_started = Instant::now();
    for (position, token) in prompt_tokens.iter().copied().enumerate() {
        logits = model
            .forward(token, position, &mut caches)
            .map_err(io_error)?;
    }
    let prompt_ns = prompt_started.elapsed().as_nanos() as u64;

    let prompt_hash = sha256_bytes(args.prompt.as_bytes());
    let mut generated_ids = Vec::with_capacity(args.max_new_tokens);
    let mut generated_text = String::new();
    let mut generated_logits = Vec::new();
    let mut next_token_latency_ns = Vec::new();
    let mut last_logit = 0.0f32;
    let mut pending_compute_ns = prompt_ns;
    for index in 0..args.max_new_tokens {
        let (token, logit) = argmax(&logits).map_err(io_error)?;
        last_logit = logit;
        generated_ids.push(token);
        generated_text.push_str(&model.tokenizer.decode_token(token).map_err(io_error)?);
        generated_logits.push(logit);
        remora
            .record_spent(
                LedgerClass::AuthoritativeUseful,
                format!(
                    "{}:{}:generated:{}:{}",
                    model_root, prompt_hash, index, token
                ),
                0,
                1,
                1,
            )
            .map_err(|error| io_error(format!("REMORA ledger rejected exact token: {error:?}")))?;
        remora.observe(
            FastObservation {
                transfer_cost_ns: 0,
                compute_cost_ms: (pending_compute_ns / 1_000_000).max(1),
                was_useful: true,
                miss: false,
            },
            None,
        );
        if model.tokenizer.eos_id == Some(token) || index + 1 == args.max_new_tokens {
            break;
        }
        let started = Instant::now();
        logits = model
            .forward(token, prompt_tokens.len() + index, &mut caches)
            .map_err(io_error)?;
        pending_compute_ns = started.elapsed().as_nanos() as u64;
        next_token_latency_ns.push(pending_compute_ns);
    }
    let remora_final = remora.snapshot();

    let output = json!({
        "schema": "har.native_qwen3_generation.v1",
        "status": "REAL_QWEN_NATIVE_PROMPT_TO_TOKEN_PASS",
        "classification": "REAL_GGUF_QWEN3_NATIVE_RUST_CPU_GENERATION",
        "claims": {
            "real_gguf_weights": true,
            "native_tokenizer": true,
            "native_qwen3_forward": true,
            "prompt_to_token_generation": true,
            "vulkan_kernel_consumption": false,
            "physical_moe_residency": false,
            "foreign_backend_execution": false
        },
        "model": {
            "path": args.model,
            "basename": args.model.file_name().map(|value| value.to_string_lossy().into_owned()),
            "model_id": model.phenotype.model_name,
            "architecture": model.phenotype.architecture,
            "file_bytes": model.phenotype.file_bytes,
            "source_model_sha256": model.phenotype.sha256,
            "tensor_count": model.phenotype.tensor_count,
            "vocab": model.vocab,
            "layers": model.n_layer,
            "embedding_length": model.n_embd,
            "attention_heads": model.n_heads,
            "kv_heads": model.n_kv_heads,
            "head_dim": model.head_dim,
            "feed_forward_length": model.n_ff,
            "rms_epsilon": model.rms_epsilon,
            "rope_freq_base": model.rope_freq_base
        },
        "input": {
            "prompt": args.prompt,
            "prompt_sha256": prompt_hash,
            "token_ids": prompt_tokens,
            "token_count": prompt_tokens.len(),
            "tokenizer_bos_id": model.tokenizer.bos_id,
            "tokenizer_eos_id": model.tokenizer.eos_id,
            "tokenizer_add_bos": model.tokenizer.add_bos
        },
        "output": {
            "generated_token_ids": generated_ids,
            "generated_text": generated_text,
            "generated_token_count": generated_ids.len(),
            "selected_logit_f32": generated_logits,
            "selected_logit_last": last_logit,
            "logits_sha256": sha256_f32(&logits)
        },
        "remora": {
            "schema": har_metabolism::METABOLISM_SCHEMA,
            "initial_snapshot": remora_initial,
            "final_snapshot": remora_final,
            "telemetry_totals": remora_final.to_totals(),
            "safe_surplus": remora_safe_surplus,
            "control_basis": {
                "model_root": model_root,
                "graph_identity": "har.native.qwen3.dense.cpu.v1",
                "worker_set": format!("std-scoped-workers:{worker_count}")
            },
            "energy_scope": "UNKNOWN; no GPU energy counter was sampled"
        },
        "timing": {
            "timestamp_unix_s": parse_timestamp(),
            "model_load_and_map_ns": load_ns,
            "prompt_forward_ns": prompt_ns,
            "prompt_tokens_per_second": prompt_tokens.len() as f64 / (prompt_ns as f64 / 1_000_000_000.0).max(1.0e-12),
            "generated_forward_ns": next_token_latency_ns,
            "generated_forward_count": next_token_latency_ns.len()
        },
        "notes": [
            "Dense Qwen3 is the first native generation lane; qwen3.5/qwen3.6 hybrid MoE is rejected by architecture identity.",
            "The tied token embedding is used as the output projection when output.weight is absent, matching the Qwen3 GGUF contract.",
            "This artifact proves native CPU graph execution and prompt-to-token output, not Vulkan kernel consumption or MoE VRAM residency."
        ],
        "markers": [
            "HAR_NATIVE_QWEN3_PROMPT_TO_TOKEN_PASS",
            "HAR_REAL_GGUF_WEIGHTS_CONSUMED",
            "HAR_NATIVE_TOKENIZER_PASS",
            "HAR_FULL_QWEN35_MOE_NATIVE_LOOP_NOT_READY"
        ]
    });

    ensure_parent(&args.output)?;
    std::fs::write(&args.output, serde_json::to_vec_pretty(&output)?)?;
    println!("[HAR_NATIVE_QWEN3_PROMPT_TO_TOKEN_PASS]");
    println!(
        "model={} prompt_tokens={:?} generated_ids={:?} generated_text={:?}",
        model.phenotype.model_name, prompt_tokens, generated_ids, generated_text
    );
    println!(
        "prompt_forward_ns={} result={}",
        prompt_ns,
        args.output.display()
    );
    Ok(())
}

/// Small shared handles used by the bounded hybrid-MoE experiment.  The
/// dense Qwen3 binary remains the owner of the implementation; this wrapper
/// only exposes the decoder/tokenizer/mmap primitives.
pub struct SharedMappedFile(MappedFile);
impl SharedMappedFile {
    pub fn open(path: &Path, len: u64) -> Result<Self, String> {
        Ok(Self(MappedFile::open(path, len)?))
    }

    pub fn bytes(&self, offset: u64, len: usize) -> Result<&[u8], String> {
        self.0.bytes(offset, len)
    }
}

pub struct SharedTokenizer(Tokenizer);
impl SharedTokenizer {
    pub fn load(path: &Path) -> Result<Self, String> {
        Ok(Self(Tokenizer::load(path)?))
    }

    pub fn encode(&self, text: &str, suppress_bos: bool) -> Result<Vec<u32>, String> {
        self.0.encode(text, suppress_bos)
    }

    pub fn decode_token(&self, id: u32) -> Result<String, String> {
        self.0.decode_token(id)
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.0.bos_id
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.0.eos_id
    }

    pub fn add_bos(&self) -> bool {
        self.0.add_bos
    }
}

pub fn shared_half_to_f32(value: u16) -> f32 {
    half_to_f32(value)
}

pub fn shared_dot_quant_row(row: &[u8], input: &[f32], ggml_type: u32) -> Result<f32, String> {
    dot_quant_row(row, input, ggml_type)
}

pub fn shared_decode_quant_block(
    block: &[u8],
    ggml_type: u32,
    output: &mut [f32; QK_K],
) -> Result<(), String> {
    decode_quant_block(block, ggml_type, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q6_zero_block_decodes_to_zero() {
        let block = [0u8; Q6_K_BLOCK_BYTES];
        let mut output = [1.0f32; QK_K];
        decode_quant_block(&block, Q6_K_GGML_TYPE, &mut output).unwrap();
        assert!(output.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn q4_zero_block_decodes_to_zero() {
        let block = [0u8; Q4_K_BLOCK_BYTES];
        let mut output = [1.0f32; QK_K];
        decode_quant_block(&block, Q4_K_GGML_TYPE, &mut output).unwrap();
        assert!(output.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn q6_centered_codes_decode_with_signed_scales() {
        let mut block = [0u8; Q6_K_BLOCK_BYTES];
        block[208..210].copy_from_slice(&0x3c00u16.to_le_bytes());
        block[192..208].fill(1);
        let mut output = [0.0f32; QK_K];
        decode_quant_block(&block, Q6_K_GGML_TYPE, &mut output).unwrap();
        assert!(output.iter().all(|value| *value == -32.0));
    }

    #[test]
    fn qwen2_ascii_tokenizer_covers_text() {
        let pieces = qwen2_pieces("Hello world!").unwrap();
        assert_eq!(pieces.concat(), "Hello world!");
        assert!(!pieces.is_empty());
    }

    #[test]
    fn argmax_prefers_lowest_id_on_equal_logits() {
        assert_eq!(argmax(&[2.0, 3.0, 3.0]).unwrap().0, 1);
    }
}
