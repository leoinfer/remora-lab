//! Native Rust CPU execution for a caller-supplied Qwen3.6/Qwen3.5 hybrid-MoE GGUF.
//!
//! This lane is intentionally separate from the dense `har-native-qwen3`
//! binary.  It implements the Qwen35MoE layer equations directly from GGUF
//! bytes: recurrent gated-delta-net layers, full-attention layers, routed
//! top-8 experts, the shared expert, and the tied/output projection. It never
//! calls a foreign inference backend. The Vulkan Q6_K stable-slot witness remains a
//! separate bounded consumer test; this binary is CPU-native and does not
//! claim Vulkan execution.

#[allow(dead_code)]
#[path = "har-native-qwen3.rs"]
mod dense_primitives;

use dense_primitives::{
    shared_decode_quant_block, shared_dot_quant_row, shared_half_to_f32, SharedMappedFile,
    SharedTokenizer,
};
use har_metabolism::clock::{ClockBasis, FastObservation};
use har_metabolism::controller::ControllerConfig;
use har_metabolism::ledger::LedgerClass;
use har_metabolism::surplus::SurplusInputs;
use har_metabolism::MetabolismController;
use har_model::{GgufReader, ModelPhenotype, TensorDescriptor};
use serde_json::json;

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{Error as IoError, ErrorKind};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const QK_K: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q6_K_BLOCK_BYTES: usize = 210;
const Q8_0_BLOCK_BYTES: usize = 34;
const Q4_K_GGML_TYPE: u32 = 12;
const Q6_K_GGML_TYPE: u32 = 14;
const Q8_0_GGML_TYPE: u32 = 8;
const F32_GGML_TYPE: u32 = 0;
const F16_GGML_TYPE: u32 = 1;
const DEFAULT_PROMPT: &str = "Hello";
const DEFAULT_NEW_TOKENS: usize = 1;
const DEFAULT_OUTPUT: &str = "results/native_qwen35moe_generation.json";
const TOP_K: usize = 8;
const ROUTER_EPS: f32 = 1.0e-9;

#[derive(Debug)]
struct Args {
    model: PathBuf,
    prompt: String,
    max_new_tokens: usize,
    output: PathBuf,
    no_bos: bool,
}

fn usage() -> &'static str {
    "usage: har-native-qwen35moe <model.gguf> [--prompt TEXT] [--max-new-tokens N] [--output path.json] [--no-bos]"
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
    if args.max_new_tokens == 0 || args.max_new_tokens > 16 {
        return Err("--max-new-tokens must be between 1 and 16".into());
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
    shared_half_to_f32(value)
}

fn q8_0_dot(row: &[u8], input: &[f32]) -> Result<f32, String> {
    if row.len() != input.len() / 32 * Q8_0_BLOCK_BYTES || input.len() % 32 != 0 {
        return Err(format!(
            "Q8_0 row geometry mismatch: row_bytes={} input_elements={}",
            row.len(),
            input.len()
        ));
    }
    let mut sum = 0.0f32;
    for (block_index, block) in row.chunks_exact(Q8_0_BLOCK_BYTES).enumerate() {
        let d = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let input_base = block_index * 32;
        for lane in 0..32 {
            sum += d * block[2 + lane] as i8 as f32 * input[input_base + lane];
        }
    }
    Ok(sum)
}

fn q8_0_decode_row(row: &[u8], output: &mut [f32]) -> Result<(), String> {
    if row.len() != output.len() / 32 * Q8_0_BLOCK_BYTES || output.len() % 32 != 0 {
        return Err("Q8_0 decode geometry mismatch".into());
    }
    for (block_index, block) in row.chunks_exact(Q8_0_BLOCK_BYTES).enumerate() {
        let d = half_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for lane in 0..32 {
            output[block_index * 32 + lane] = d * block[2 + lane] as i8 as f32;
        }
    }
    Ok(())
}

fn dot_f32_row(row: &[u8], input: &[f32]) -> Result<f32, String> {
    if row.len() != input.len() * 4 {
        return Err("F32 row geometry mismatch".into());
    }
    Ok(row
        .chunks_exact(4)
        .zip(input)
        .map(|(weight, value)| f32::from_le_bytes(weight.try_into().unwrap()) * value)
        .sum())
}

fn dot_f16_row(row: &[u8], input: &[f32]) -> Result<f32, String> {
    if row.len() != input.len() * 2 {
        return Err("F16 row geometry mismatch".into());
    }
    Ok(row
        .chunks_exact(2)
        .zip(input)
        .map(|(weight, value)| half_to_f32(u16::from_le_bytes(weight.try_into().unwrap())) * value)
        .sum())
}

fn row_bytes_for_type(ggml_type: u32, elements: usize) -> Result<usize, String> {
    let bytes = match ggml_type {
        Q4_K_GGML_TYPE => {
            if elements % QK_K != 0 {
                return Err("Q4_K input length is not divisible by 256".into());
            }
            elements / QK_K * Q4_K_BLOCK_BYTES
        }
        Q6_K_GGML_TYPE => {
            if elements % QK_K != 0 {
                return Err("Q6_K input length is not divisible by 256".into());
            }
            elements / QK_K * Q6_K_BLOCK_BYTES
        }
        Q8_0_GGML_TYPE => {
            if elements % 32 != 0 {
                return Err("Q8_0 input length is not divisible by 32".into());
            }
            elements / 32 * Q8_0_BLOCK_BYTES
        }
        F32_GGML_TYPE => elements.checked_mul(4).ok_or("F32 row size overflow")?,
        F16_GGML_TYPE => elements.checked_mul(2).ok_or("F16 row size overflow")?,
        other => return Err(format!("unsupported GGML type {other}")),
    };
    Ok(bytes)
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        (1.0 + value.exp()).ln()
    }
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

fn l2_norm(values: &mut [f32], epsilon: f32) {
    let scale = values
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(epsilon)
        .recip();
    for value in values {
        *value *= scale;
    }
}

fn rms_norm_heads(
    values: &mut [f32],
    weight: &[f32],
    epsilon: f32,
    heads: usize,
    dim: usize,
) -> Result<(), String> {
    if values.len() != heads * dim || weight.len() != dim {
        return Err("head RMSNorm geometry mismatch".into());
    }
    for head in 0..heads {
        rms_norm(&mut values[head * dim..(head + 1) * dim], weight, epsilon)?;
    }
    Ok(())
}

fn rope_partial(
    values: &mut [f32],
    position: usize,
    heads: usize,
    head_dim: usize,
    rotate_dim: usize,
    base: f32,
) {
    let half = rotate_dim / 2;
    for head in 0..heads {
        let offset = head * head_dim;
        for lane in 0..half {
            let theta = position as f32 * base.powf(-(2.0 * lane as f32) / rotate_dim as f32);
            let (sin, cos) = theta.sin_cos();
            let x0 = values[offset + lane];
            let x1 = values[offset + half + lane];
            values[offset + lane] = x0 * cos - x1 * sin;
            values[offset + half + lane] = x0 * sin + x1 * cos;
        }
    }
}

#[derive(Clone, Debug, serde::Serialize)]
struct FullKvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    positions: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl FullKvCache {
    fn new(kv_heads: usize, head_dim: usize) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            positions: 0,
            kv_heads,
            head_dim,
        }
    }

    fn append(&mut self, key: &[f32], value: &[f32]) -> Result<(), String> {
        let size = self.kv_heads * self.head_dim;
        if key.len() != size || value.len() != size {
            return Err("full-attention KV append geometry mismatch".into());
        }
        self.keys.extend_from_slice(key);
        self.values.extend_from_slice(value);
        self.positions += 1;
        Ok(())
    }

    fn row<'a>(&self, data: &'a [f32], position: usize, head: usize) -> &'a [f32] {
        let offset = (position * self.kv_heads + head) * self.head_dim;
        &data[offset..offset + self.head_dim]
    }
}

#[derive(Clone, Debug, serde::Serialize)]
struct RecurrentState {
    conv: Vec<f32>,
    state: Vec<f32>,
}

impl RecurrentState {
    fn new(
        conv_channels: usize,
        conv_kernel: usize,
        state_size: usize,
        value_heads: usize,
    ) -> Self {
        Self {
            conv: vec![0.0; conv_channels * (conv_kernel - 1)],
            state: vec![0.0; state_size * state_size * value_heads],
        }
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct RouteRecord {
    pub layer: usize,
    pub experts: Vec<u32>,
    pub weights: Vec<f32>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct LayerTrace {
    pub layer: usize,
    pub input: Vec<f32>,
    pub ffn_residual: Vec<f32>,
    pub ffn_input: Vec<f32>,
    pub output: Vec<f32>,
    pub route: RouteRecord,
}

pub struct NativeQwen35Moe {
    phenotype: ModelPhenotype,
    tensors: HashMap<String, TensorDescriptor>,
    file: SharedMappedFile,
    tokenizer: SharedTokenizer,
    caches: Vec<FullKvCache>,
    recurrent: Vec<RecurrentState>,
    n_layer: usize,
    n_embd: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    vocab: usize,
    experts: usize,
    experts_used: usize,
    expert_ff: usize,
    shared_ff: usize,
    ssm_inner: usize,
    ssm_state: usize,
    ssm_groups: usize,
    ssm_rank: usize,
    conv_kernel: usize,
    full_attention_interval: usize,
    rope_dim: usize,
    rms_epsilon: f32,
    rope_freq_base: f32,
}

impl NativeQwen35Moe {
    fn metadata_usize(
        metadata: &std::collections::BTreeMap<String, String>,
        key: &str,
    ) -> Result<usize, String> {
        metadata
            .get(key)
            .ok_or_else(|| format!("required GGUF metadata is missing: {key}"))?
            .parse::<usize>()
            .map_err(|_| format!("GGUF metadata is not an integer: {key}"))
    }

    fn metadata_f32(
        metadata: &std::collections::BTreeMap<String, String>,
        key: &str,
    ) -> Result<f32, String> {
        metadata
            .get(key)
            .ok_or_else(|| format!("required GGUF metadata is missing: {key}"))?
            .parse::<f32>()
            .map_err(|_| format!("GGUF metadata is not f32: {key}"))
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let reader = GgufReader::new(path);
        let identity = reader
            .inspect(false)
            .map_err(|error| format!("GGUF inspection failed: {error}"))?;
        if identity.architecture != "qwen35moe" {
            return Err(format!(
                "native Qwen35MoE lane requires architecture qwen35moe; found {}",
                identity.architecture
            ));
        }
        let phenotype = reader
            .inspect(true)
            .map_err(|error| format!("GGUF identity hashing failed: {error}"))?;
        let metadata = &phenotype.metadata_summary;
        let n_layer = Self::metadata_usize(metadata, "qwen35moe.block_count")?;
        let n_embd = Self::metadata_usize(metadata, "qwen35moe.embedding_length")?;
        let n_heads = Self::metadata_usize(metadata, "qwen35moe.attention.head_count")?;
        let n_kv_heads = Self::metadata_usize(metadata, "qwen35moe.attention.head_count_kv")?;
        let head_dim = Self::metadata_usize(metadata, "qwen35moe.attention.key_length")?;
        let experts = Self::metadata_usize(metadata, "qwen35moe.expert_count")?;
        let experts_used = Self::metadata_usize(metadata, "qwen35moe.expert_used_count")?;
        let expert_ff = Self::metadata_usize(metadata, "qwen35moe.expert_feed_forward_length")?;
        let shared_ff =
            Self::metadata_usize(metadata, "qwen35moe.expert_shared_feed_forward_length")?;
        let conv_kernel = Self::metadata_usize(metadata, "qwen35moe.ssm.conv_kernel")?;
        let ssm_state = Self::metadata_usize(metadata, "qwen35moe.ssm.state_size")?;
        let ssm_groups = Self::metadata_usize(metadata, "qwen35moe.ssm.group_count")?;
        let ssm_rank = Self::metadata_usize(metadata, "qwen35moe.ssm.time_step_rank")?;
        let ssm_inner = Self::metadata_usize(metadata, "qwen35moe.ssm.inner_size")?;
        let full_attention_interval =
            Self::metadata_usize(metadata, "qwen35moe.full_attention_interval")?;
        let rope_dim = Self::metadata_usize(metadata, "qwen35moe.rope.dimension_count")?;
        let rms_epsilon =
            Self::metadata_f32(metadata, "qwen35moe.attention.layer_norm_rms_epsilon")?;
        let rope_freq_base = Self::metadata_f32(metadata, "qwen35moe.rope.freq_base")?;
        if n_layer == 0
            || n_embd == 0
            || n_heads == 0
            || n_kv_heads == 0
            || head_dim == 0
            || experts == 0
            || experts_used == 0
            || experts_used > experts
            || expert_ff == 0
            || ssm_inner == 0
            || ssm_state == 0
            || ssm_groups == 0
            || ssm_rank == 0
            || conv_kernel < 2
            || full_attention_interval == 0
            || n_heads * head_dim != n_embd * 2
            || ssm_inner % ssm_rank != 0
            || ssm_inner % ssm_groups != 0
        {
            return Err("Qwen35MoE metadata geometry is incomplete or inconsistent".into());
        }
        let vocab = phenotype
            .tensor("token_embd.weight")
            .and_then(|tensor| tensor.dimensions.get(1).copied())
            .ok_or("token_embd.weight is missing")? as usize;
        let tensors = phenotype
            .tensors
            .iter()
            .cloned()
            .map(|tensor| (tensor.name.clone(), tensor))
            .collect::<HashMap<_, _>>();
        let file = SharedMappedFile::open(path, phenotype.file_bytes)?;
        let tokenizer = SharedTokenizer::load(path)?;
        let value_heads = ssm_rank;
        let conv_channels = ssm_inner + 2 * ssm_groups * ssm_state;
        let caches = (0..n_layer)
            .map(|_| FullKvCache::new(n_kv_heads, head_dim))
            .collect();
        let recurrent = (0..n_layer)
            .map(|_| RecurrentState::new(conv_channels, conv_kernel, ssm_state, value_heads))
            .collect();
        Ok(Self {
            phenotype,
            tensors,
            file,
            tokenizer,
            caches,
            recurrent,
            n_layer,
            n_embd,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab,
            experts,
            experts_used,
            expert_ff,
            shared_ff,
            ssm_inner,
            ssm_state,
            ssm_groups,
            ssm_rank,
            conv_kernel,
            full_attention_interval,
            rope_dim,
            rms_epsilon,
            rope_freq_base,
        })
    }

    fn tensor(&self, name: &str) -> Result<&TensorDescriptor, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("required Qwen35MoE tensor is missing: {name}"))
    }

    fn tensor_bytes(&self, tensor: &TensorDescriptor) -> Result<&[u8], String> {
        let length = usize::try_from(tensor.payload_bytes)
            .map_err(|_| format!("tensor {} is too large for this host", tensor.name))?;
        self.file.bytes(tensor.file_offset, length)
    }

    fn read_f32(&self, tensor: &TensorDescriptor) -> Result<Vec<f32>, String> {
        let count = tensor.element_count as usize;
        if tensor.ggml_type != F32_GGML_TYPE || tensor.payload_bytes < (count * 4) as u64 {
            return Err(format!(
                "tensor {} is not an exact F32 vector: type={} elements={} bytes={}",
                tensor.name, tensor.ggml_type, tensor.element_count, tensor.payload_bytes
            ));
        }
        let bytes = self.tensor_bytes(tensor)?;
        Ok(bytes[..count * 4]
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect())
    }

    fn dot_row(&self, row: &[u8], input: &[f32], ggml_type: u32) -> Result<f32, String> {
        match ggml_type {
            Q4_K_GGML_TYPE | Q6_K_GGML_TYPE => shared_dot_quant_row(row, input, ggml_type),
            Q8_0_GGML_TYPE => q8_0_dot(row, input),
            F32_GGML_TYPE => dot_f32_row(row, input),
            F16_GGML_TYPE => dot_f16_row(row, input),
            other => Err(format!("unsupported matvec GGML type {other}")),
        }
    }

    fn matvec_rows(
        &self,
        tensor: &TensorDescriptor,
        input: &[f32],
        row_start: usize,
        output: &mut [f32],
        extra_offset: u64,
    ) -> Result<(), String> {
        let row_bytes = row_bytes_for_type(tensor.ggml_type, input.len())?;
        for (relative, value) in output.iter_mut().enumerate() {
            let row = row_start + relative;
            let offset = tensor
                .file_offset
                .checked_add(extra_offset)
                .and_then(|base| base.checked_add((row * row_bytes) as u64))
                .ok_or_else(|| format!("tensor {} row offset overflow", tensor.name))?;
            let bytes = self.file.bytes(offset, row_bytes)?;
            *value = self.dot_row(bytes, input, tensor.ggml_type)?;
        }
        Ok(())
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
        let row_bytes = row_bytes_for_type(tensor.ggml_type, input.len())?;
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
                    scope.spawn(move || self.matvec_rows(tensor, input, row_start, chunk, 0)),
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

    fn expert_matvec(
        &self,
        tensor: &TensorDescriptor,
        expert: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if tensor.dimensions.len() != 3
            || tensor.dimensions[0] as usize != input.len()
            || tensor.dimensions[2] as usize != self.experts
        {
            return Err(format!(
                "expert matvec shape mismatch for {}: dims={:?}, expert={} input={}",
                tensor.name,
                tensor.dimensions,
                expert,
                input.len()
            ));
        }
        if expert >= self.experts {
            return Err(format!(
                "expert {expert} exceeds model expert count {}",
                self.experts
            ));
        }
        let rows = tensor.dimensions[1] as usize;
        let row_bytes = row_bytes_for_type(tensor.ggml_type, input.len())?;
        let expert_bytes = row_bytes
            .checked_mul(rows)
            .ok_or_else(|| format!("expert row geometry overflows for {}", tensor.name))?;
        let expected = expert_bytes
            .checked_mul(self.experts)
            .ok_or_else(|| format!("expert tensor geometry overflows for {}", tensor.name))?;
        if expected > tensor.payload_bytes as usize {
            return Err(format!(
                "expert tensor {} payload {} is shorter than expected {}",
                tensor.name, tensor.payload_bytes, expected
            ));
        }
        let extra_offset = expert_bytes
            .checked_mul(expert)
            .ok_or_else(|| "expert offset overflows".to_string())?
            as u64;
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
                joins.push(scope.spawn(move || {
                    self.matvec_rows(tensor, input, row_start, chunk, extra_offset)
                }));
            }
            for join in joins {
                join.join()
                    .map_err(|_| "expert matvec worker panicked".to_string())??;
            }
            Ok::<(), String>(())
        })?;
        Ok(output)
    }

    fn decode_row(&self, row: &[u8], elements: usize, ggml_type: u32) -> Result<Vec<f32>, String> {
        let mut output = vec![0.0f32; elements];
        match ggml_type {
            Q4_K_GGML_TYPE | Q6_K_GGML_TYPE => {
                let block_bytes = if ggml_type == Q4_K_GGML_TYPE {
                    Q4_K_BLOCK_BYTES
                } else {
                    Q6_K_BLOCK_BYTES
                };
                if elements % QK_K != 0 || row.len() != elements / QK_K * block_bytes {
                    return Err("quantized embedding row geometry mismatch".into());
                }
                for (block_index, block) in row.chunks_exact(block_bytes).enumerate() {
                    let mut decoded = [0.0f32; QK_K];
                    shared_decode_quant_block(block, ggml_type, &mut decoded)?;
                    output[block_index * QK_K..(block_index + 1) * QK_K].copy_from_slice(&decoded);
                }
            }
            Q8_0_GGML_TYPE => q8_0_decode_row(row, &mut output)?,
            F32_GGML_TYPE => {
                if row.len() != elements * 4 {
                    return Err("F32 embedding row geometry mismatch".into());
                }
                for (index, value) in output.iter_mut().enumerate() {
                    *value = f32::from_le_bytes(row[index * 4..index * 4 + 4].try_into().unwrap());
                }
            }
            F16_GGML_TYPE => {
                if row.len() != elements * 2 {
                    return Err("F16 embedding row geometry mismatch".into());
                }
                for (index, value) in output.iter_mut().enumerate() {
                    *value = half_to_f32(u16::from_le_bytes(
                        row[index * 2..index * 2 + 2].try_into().unwrap(),
                    ));
                }
            }
            other => return Err(format!("unsupported embedding GGML type {other}")),
        }
        Ok(output)
    }

    pub fn tokenize(&self, text: &str, suppress_bos: bool) -> Result<Vec<u32>, String> {
        self.tokenizer.encode(text, suppress_bos)
    }

    pub fn embedding(&self, token: u32) -> Result<Vec<f32>, String> {
        if token as usize >= self.vocab {
            return Err(format!(
                "token id {token} exceeds vocabulary {}",
                self.vocab
            ));
        }
        let tensor = self.tensor("token_embd.weight")?;
        if tensor.dimensions.len() != 2 || tensor.dimensions[0] as usize != self.n_embd {
            return Err("token embedding shape does not match hidden size".into());
        }
        let row_bytes = tensor.payload_bytes / tensor.dimensions[1];
        let offset = tensor
            .file_offset
            .checked_add(row_bytes * token as u64)
            .ok_or("embedding row offset overflow")?;
        let row = self.file.bytes(offset, row_bytes as usize)?;
        self.decode_row(row, self.n_embd, tensor.ggml_type)
    }

    #[allow(clippy::needless_range_loop)]
    fn full_attention(
        &mut self,
        layer: usize,
        hidden: &[f32],
        position: usize,
    ) -> Result<Vec<f32>, String> {
        let prefix = format!("blk.{layer}");
        let q_full = self.matvec(self.tensor(&format!("{prefix}.attn_q.weight"))?, hidden)?;
        // GGML's Qcur_full is laid out per head as [Q(head), gate(head)],
        // not as one contiguous Q half followed by one gate half.
        let mut q = vec![0.0f32; self.n_heads * self.head_dim];
        let mut gate = vec![0.0f32; self.n_heads * self.head_dim];
        for head in 0..self.n_heads {
            let source = head * self.head_dim * 2;
            q[head * self.head_dim..(head + 1) * self.head_dim]
                .copy_from_slice(&q_full[source..source + self.head_dim]);
            gate[head * self.head_dim..(head + 1) * self.head_dim]
                .copy_from_slice(&q_full[source + self.head_dim..source + self.head_dim * 2]);
        }
        let mut k = self.matvec(self.tensor(&format!("{prefix}.attn_k.weight"))?, hidden)?;
        let v = self.matvec(self.tensor(&format!("{prefix}.attn_v.weight"))?, hidden)?;
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
        rope_partial(
            &mut q,
            position,
            self.n_heads,
            self.head_dim,
            self.rope_dim,
            self.rope_freq_base,
        );
        rope_partial(
            &mut k,
            position,
            self.n_kv_heads,
            self.head_dim,
            self.rope_dim,
            self.rope_freq_base,
        );
        self.caches[layer].append(&k, &v)?;
        let mut attended = vec![0.0f32; self.n_heads * self.head_dim];
        let scale = (self.head_dim as f32).sqrt().recip();
        for head in 0..self.n_heads {
            let query = &q[head * self.head_dim..(head + 1) * self.head_dim];
            let kv_head = head / (self.n_heads / self.n_kv_heads);
            let mut scores = Vec::with_capacity(self.caches[layer].positions);
            for cached_position in 0..self.caches[layer].positions {
                let key =
                    self.caches[layer].row(&self.caches[layer].keys, cached_position, kv_head);
                scores.push(
                    query
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
            let output = &mut attended[head * self.head_dim..(head + 1) * self.head_dim];
            for cached_position in 0..self.caches[layer].positions {
                let weight = scores[cached_position] / normalizer;
                let value =
                    self.caches[layer].row(&self.caches[layer].values, cached_position, kv_head);
                for lane in 0..self.head_dim {
                    output[lane] += weight * value[lane];
                }
            }
        }
        for (value, gate) in attended.iter_mut().zip(gate) {
            *value *= sigmoid(gate);
        }
        self.matvec(
            self.tensor(&format!("{prefix}.attn_output.weight"))?,
            &attended,
        )
    }

    #[allow(clippy::needless_range_loop)]
    fn recurrent_attention(&mut self, layer: usize, hidden: &[f32]) -> Result<Vec<f32>, String> {
        let prefix = format!("blk.{layer}");
        let qkv = self.matvec(self.tensor(&format!("{prefix}.attn_qkv.weight"))?, hidden)?;
        let z = self.matvec(self.tensor(&format!("{prefix}.attn_gate.weight"))?, hidden)?;
        let beta_raw = self.matvec(self.tensor(&format!("{prefix}.ssm_beta.weight"))?, hidden)?;
        let alpha_raw = self.matvec(self.tensor(&format!("{prefix}.ssm_alpha.weight"))?, hidden)?;
        let dt_bias = self.read_f32(self.tensor(&format!("{prefix}.ssm_dt.bias"))?)?;
        let a = self.read_f32(self.tensor(&format!("{prefix}.ssm_a"))?)?;
        let conv_tensor = self.tensor(&format!("{prefix}.ssm_conv1d.weight"))?;
        let conv_file_offset = conv_tensor.file_offset;
        let conv_dimensions = conv_tensor.dimensions.clone();
        let conv_channels = self.ssm_inner + 2 * self.ssm_groups * self.ssm_state;
        let history = self.conv_kernel - 1;
        if qkv.len() != conv_channels
            || conv_dimensions != vec![self.conv_kernel as u64, conv_channels as u64]
        {
            return Err(format!(
                "recurrent convolution geometry mismatch at layer {layer}"
            ));
        }
        let conv_offset = &mut self.recurrent[layer].conv;
        let mut x = vec![0.0f32; conv_channels];
        for channel in 0..conv_channels {
            let row_offset = conv_file_offset + (channel * self.conv_kernel * 4) as u64;
            let weights = self.file.bytes(row_offset, self.conv_kernel * 4)?;
            let mut sum = 0.0f32;
            for tap in 0..history {
                sum += f32::from_le_bytes(weights[tap * 4..tap * 4 + 4].try_into().unwrap())
                    * conv_offset[channel * history + tap];
            }
            sum += f32::from_le_bytes(weights[history * 4..history * 4 + 4].try_into().unwrap())
                * qkv[channel];
            x[channel] = silu(sum);
            for tap in 0..history.saturating_sub(1) {
                conv_offset[channel * history + tap] = conv_offset[channel * history + tap + 1];
            }
            if history > 0 {
                conv_offset[channel * history + history - 1] = qkv[channel];
            }
        }
        let key_dim = self.ssm_groups * self.ssm_state;
        let value_heads = self.ssm_rank;
        let head_value_dim = self.ssm_inner / value_heads;
        if key_dim != self.ssm_state * self.ssm_groups || head_value_dim != self.ssm_state {
            return Err("Qwen35MoE recurrent head geometry mismatch".into());
        }
        let mut q = x[..key_dim].to_vec();
        let mut k = x[key_dim..2 * key_dim].to_vec();
        let v = x[2 * key_dim..].to_vec();
        for head in 0..self.ssm_groups {
            l2_norm(
                &mut q[head * self.ssm_state..(head + 1) * self.ssm_state],
                self.rms_epsilon,
            );
            l2_norm(
                &mut k[head * self.ssm_state..(head + 1) * self.ssm_state],
                self.rms_epsilon,
            );
        }
        let mut q_heads = vec![0.0f32; self.ssm_inner];
        let mut k_heads = vec![0.0f32; self.ssm_inner];
        for head in 0..value_heads {
            let source = head % self.ssm_groups;
            q_heads[head * self.ssm_state..(head + 1) * self.ssm_state]
                .copy_from_slice(&q[source * self.ssm_state..(source + 1) * self.ssm_state]);
            k_heads[head * self.ssm_state..(head + 1) * self.ssm_state]
                .copy_from_slice(&k[source * self.ssm_state..(source + 1) * self.ssm_state]);
        }
        let mut output = vec![0.0f32; self.ssm_inner];
        let state_size = self.ssm_state * self.ssm_state;
        for head in 0..value_heads {
            let beta = sigmoid(beta_raw[head]);
            let gate = softplus(alpha_raw[head] + dt_bias[head]) * a[head];
            let decay = gate.exp();
            let state_base = head * state_size;
            for value in &mut self.recurrent[layer].state[state_base..state_base + state_size] {
                *value *= decay;
            }
            let k_head = &k_heads[head * self.ssm_state..(head + 1) * self.ssm_state];
            let v_head = &v[head * head_value_dim..(head + 1) * head_value_dim];
            // The GGML recurrent cache stores the state transposed:
            // state[row * S + input] = S[input, row].  Keep that layout so
            // the scalar recurrence matches ggml's dot/update order.
            let mut delta = vec![0.0f32; self.ssm_state];
            for row in 0..self.ssm_state {
                let mut state_key_dot = 0.0f32;
                for input_lane in 0..self.ssm_state {
                    state_key_dot += self.recurrent[layer].state
                        [state_base + row * self.ssm_state + input_lane]
                        * k_head[input_lane];
                }
                delta[row] = (v_head[row] - state_key_dot) * beta;
            }
            for row in 0..self.ssm_state {
                for input_lane in 0..self.ssm_state {
                    self.recurrent[layer].state[state_base + row * self.ssm_state + input_lane] +=
                        k_head[input_lane] * delta[row];
                }
            }
            let q_head = &q_heads[head * self.ssm_state..(head + 1) * self.ssm_state];
            for row in 0..self.ssm_state {
                for input_lane in 0..self.ssm_state {
                    output[head * head_value_dim + row] += self.recurrent[layer].state
                        [state_base + row * self.ssm_state + input_lane]
                        * q_head[input_lane]
                        / (self.ssm_state as f32).sqrt();
                }
            }
        }
        let norm = self.read_f32(self.tensor(&format!("{prefix}.ssm_norm.weight"))?)?;
        rms_norm_heads(
            &mut output,
            &norm,
            self.rms_epsilon,
            value_heads,
            self.ssm_state,
        )?;
        for (value, gate) in output.iter_mut().zip(z) {
            *value *= silu(gate);
        }
        self.matvec(self.tensor(&format!("{prefix}.ssm_out.weight"))?, &output)
    }

    fn compose_moe_route(
        &self,
        layer: usize,
        input: &[f32],
        experts: &[u32],
        weights: &[f32],
        gate_override: Option<(u32, &[f32])>,
    ) -> Result<Vec<f32>, String> {
        if experts.len() != weights.len() || experts.len() != self.experts_used {
            return Err("Qwen35MoE route geometry is invalid".into());
        }
        let prefix = format!("blk.{layer}");
        let mut routed = vec![0.0f32; self.n_embd];
        for (&expert, &weight) in experts.iter().zip(weights) {
            let gate =
                if gate_override.is_some_and(|(override_expert, _)| override_expert == expert) {
                    let values = gate_override.unwrap().1;
                    if values.len() != self.expert_ff {
                        return Err("Qwen35MoE gate override has invalid length".into());
                    }
                    values.to_vec()
                } else {
                    self.expert_matvec(
                        self.tensor(&format!("{prefix}.ffn_gate_exps.weight"))?,
                        expert as usize,
                        input,
                    )?
                };
            let up = self.expert_matvec(
                self.tensor(&format!("{prefix}.ffn_up_exps.weight"))?,
                expert as usize,
                input,
            )?;
            let mut activation = gate;
            for (value, up) in activation.iter_mut().zip(up) {
                *value = silu(*value) * up;
            }
            let down = self.expert_matvec(
                self.tensor(&format!("{prefix}.ffn_down_exps.weight"))?,
                expert as usize,
                &activation,
            )?;
            for (output, value) in routed.iter_mut().zip(down) {
                *output += weight * value;
            }
        }
        let shared_gate = self.matvec(
            self.tensor(&format!("{prefix}.ffn_gate_shexp.weight"))?,
            input,
        )?;
        let shared_up = self.matvec(
            self.tensor(&format!("{prefix}.ffn_up_shexp.weight"))?,
            input,
        )?;
        let mut shared_activation = shared_gate;
        for (value, up) in shared_activation.iter_mut().zip(shared_up) {
            *value = silu(*value) * up;
        }
        let shared_down = self.matvec(
            self.tensor(&format!("{prefix}.ffn_down_shexp.weight"))?,
            &shared_activation,
        )?;
        let shared_gate_tensor = self.tensor(&format!("{prefix}.ffn_gate_inp_shexp.weight"))?;
        let shared_gate_bytes = self.tensor_bytes(shared_gate_tensor)?;
        let shared_gate_weight = sigmoid(dot_f32_row(shared_gate_bytes, input)?);
        for (output, shared) in routed.iter_mut().zip(shared_down) {
            *output += shared_gate_weight * shared;
        }
        Ok(routed)
    }

    pub fn expert_projection(
        &self,
        layer: usize,
        expert: usize,
        projection: &str,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let tensor = self.tensor(&format!("blk.{layer}.ffn_{projection}_exps.weight"))?;
        self.expert_matvec(tensor, expert, input)
    }

    pub fn expert_projection_payload(
        &self,
        layer: usize,
        expert: usize,
        projection: &str,
    ) -> Result<Vec<u8>, String> {
        let tensor = self.tensor(&format!("blk.{layer}.ffn_{projection}_exps.weight"))?;
        if tensor.dimensions.len() != 3 || expert >= tensor.dimensions[2] as usize {
            return Err(format!(
                "expert payload geometry is invalid for {}",
                tensor.name
            ));
        }
        let input_elements = tensor.dimensions[0] as usize;
        let rows = tensor.dimensions[1] as usize;
        let row_bytes = row_bytes_for_type(tensor.ggml_type, input_elements)?;
        let expert_bytes = row_bytes
            .checked_mul(rows)
            .ok_or_else(|| format!("expert payload size overflows for {}", tensor.name))?;
        let offset = tensor
            .file_offset
            .checked_add((expert_bytes * expert) as u64)
            .ok_or_else(|| format!("expert payload offset overflows for {}", tensor.name))?;
        Ok(self.file.bytes(offset, expert_bytes)?.to_vec())
    }

    pub fn complete_layer_with_gate_override(
        &self,
        trace: &LayerTrace,
        expert: u32,
        gate: &[f32],
    ) -> Result<Vec<f32>, String> {
        if trace.route.layer >= self.n_layer || !trace.route.experts.contains(&expert) {
            return Err("gate override expert is not selected by the layer route".into());
        }
        let moe = self.compose_moe_route(
            trace.layer,
            &trace.ffn_input,
            &trace.route.experts,
            &trace.route.weights,
            Some((expert, gate)),
        )?;
        let mut output = trace.ffn_residual.clone();
        for (value, moe_value) in output.iter_mut().zip(moe) {
            *value += moe_value;
        }
        Ok(output)
    }

    pub fn n_layer(&self) -> usize {
        self.n_layer
    }
    pub fn experts(&self) -> usize {
        self.experts
    }
    pub fn experts_used(&self) -> usize {
        self.experts_used
    }
    pub fn tokenizer(&self) -> &SharedTokenizer {
        &self.tokenizer
    }
    pub fn phenotype(&self) -> &ModelPhenotype {
        &self.phenotype
    }

    pub fn execute_layer_cpu(
        &mut self,
        layer: usize,
        input: &[f32],
        position: usize,
    ) -> Result<LayerTrace, String> {
        if layer >= self.n_layer || input.len() != self.n_embd {
            return Err("Qwen35MoE layer input geometry is invalid".into());
        }
        let mut hidden = input.to_vec();
        let residual = hidden.clone();
        let prefix = format!("blk.{layer}");
        let attn_norm = self.read_f32(self.tensor(&format!("{prefix}.attn_norm.weight"))?)?;
        rms_norm(&mut hidden, &attn_norm, self.rms_epsilon)?;
        let attn = if (layer + 1) % self.full_attention_interval == 0 {
            self.full_attention(layer, &hidden, position)?
        } else {
            self.recurrent_attention(layer, &hidden)?
        };
        for (value, (left, right)) in hidden.iter_mut().zip(residual.iter().zip(attn)) {
            *value = *left + right;
        }
        let ffn_residual = hidden.clone();
        let post_norm =
            self.read_f32(self.tensor(&format!("{prefix}.post_attention_norm.weight"))?)?;
        rms_norm(&mut hidden, &post_norm, self.rms_epsilon)?;
        let ffn_input = hidden.clone();
        let (moe, route) = self.routed_moe(layer, &hidden)?;
        for (value, (left, right)) in hidden.iter_mut().zip(ffn_residual.iter().zip(&moe)) {
            *value = *left + *right;
        }
        Ok(LayerTrace {
            layer,
            input: input.to_vec(),
            ffn_residual,
            ffn_input,
            output: hidden,
            route,
        })
    }

    fn routed_moe(&self, layer: usize, input: &[f32]) -> Result<(Vec<f32>, RouteRecord), String> {
        let prefix = format!("blk.{layer}");
        let router = self.matvec(
            self.tensor(&format!("{prefix}.ffn_gate_inp.weight"))?,
            input,
        )?;
        let max_router = router.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps = router
            .iter()
            .map(|value| (*value - max_router).exp())
            .collect::<Vec<_>>();
        let denominator = exps.iter().sum::<f32>().max(ROUTER_EPS);
        let probabilities = exps
            .iter()
            .map(|value| *value / denominator)
            .collect::<Vec<_>>();
        let mut order = (0..self.experts).collect::<Vec<_>>();
        order.sort_by(|left, right| {
            probabilities[*right]
                .total_cmp(&probabilities[*left])
                .then_with(|| left.cmp(right))
        });
        let selected = order
            .into_iter()
            .take(self.experts_used)
            .collect::<Vec<_>>();
        let selected_total = selected
            .iter()
            .map(|expert| probabilities[*expert])
            .sum::<f32>()
            .max(ROUTER_EPS);
        let weights = selected
            .iter()
            .map(|expert| probabilities[*expert] / selected_total)
            .collect::<Vec<_>>();
        let selected_u32 = selected
            .iter()
            .map(|expert| *expert as u32)
            .collect::<Vec<_>>();
        let routed = self.compose_moe_route(layer, input, &selected_u32, &weights, None)?;
        Ok((
            routed,
            RouteRecord {
                layer,
                experts: selected.into_iter().map(|expert| expert as u32).collect(),
                weights,
            },
        ))
    }

    pub fn forward(
        &mut self,
        token: u32,
        position: usize,
    ) -> Result<(Vec<f32>, Vec<RouteRecord>), String> {
        let mut hidden = self.embedding(token)?;
        let mut routes = Vec::with_capacity(self.n_layer);
        for layer in 0..self.n_layer {
            let trace = self.execute_layer_cpu(layer, &hidden, position)?;
            hidden = trace.output;
            routes.push(trace.route);
        }
        let output_norm = self.read_f32(self.tensor("output_norm.weight")?)?;
        rms_norm(&mut hidden, &output_norm, self.rms_epsilon)?;
        let logits = self.matvec(self.tensor("output.weight")?, &hidden)?;
        Ok((logits, routes))
    }
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
    let mut model = NativeQwen35Moe::load(&args.model).map_err(io_error)?;
    let load_ns = load_started.elapsed().as_nanos() as u64;
    let prompt_tokens = model
        .tokenizer
        .encode(&args.prompt, args.no_bos)
        .map_err(io_error)?;
    let model_root = model
        .phenotype
        .sha256
        .clone()
        .ok_or("model hash is missing")?;
    let worker_count = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    let mut remora = MetabolismController::new(ControllerConfig {
        vram_capacity_mib: 1,
        ram_capacity_mib: model.phenotype.file_bytes / 1_048_576 + 1,
        protected_min_mib: 0,
        gpu_compute_budget: 1,
        slow_window: 256,
    });
    remora.set_basis(ClockBasis {
        model_root: model_root.clone(),
        graph_identity: "har.native.qwen35moe.hybrid.cpu.experimental.v1".into(),
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
    let mut route_trace = Vec::new();
    let prompt_started = Instant::now();
    for (position, token) in prompt_tokens.iter().copied().enumerate() {
        let (next_logits, routes) = model.forward(token, position).map_err(io_error)?;
        logits = next_logits;
        route_trace.push(routes);
    }
    let prompt_ns = prompt_started.elapsed().as_nanos() as u64;
    let prompt_hash = sha256_bytes(args.prompt.as_bytes());
    let mut generated_ids = Vec::with_capacity(args.max_new_tokens);
    let mut generated_text = String::new();
    let mut selected_logits = Vec::new();
    let mut generated_routes = Vec::new();
    let mut generated_forward_ns = Vec::new();
    let mut pending_compute_ns = prompt_ns;
    for index in 0..args.max_new_tokens {
        let (token, selected_logit) = argmax(&logits).map_err(io_error)?;
        generated_ids.push(token);
        generated_text.push_str(&model.tokenizer.decode_token(token).map_err(io_error)?);
        selected_logits.push(selected_logit);
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
        if model.tokenizer.eos_id() == Some(token) || index + 1 == args.max_new_tokens {
            break;
        }
        let started = Instant::now();
        let (next_logits, routes) = model
            .forward(token, prompt_tokens.len() + index)
            .map_err(io_error)?;
        logits = next_logits;
        generated_routes.push(routes);
        pending_compute_ns = started.elapsed().as_nanos() as u64;
        generated_forward_ns.push(pending_compute_ns);
    }
    let remora_final = remora.snapshot();
    let route_json = route_trace
        .iter()
        .chain(generated_routes.iter())
        .map(|step| {
            step.iter()
                .map(|route| {
                    json!({
                        "layer": route.layer,
                        "experts": route.experts,
                        "weights": route.weights,
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let output = json!({
        "schema": "har.native_qwen35moe_generation.v1",
        "status": "REAL_QWEN35MOE_NATIVE_PROMPT_TO_TOKEN_PASS",
        "classification": "REAL_GGUF_QWEN35MOE_NATIVE_RUST_CPU_GENERATION",
        "claims": {
            "real_gguf_weights": true,
            "native_tokenizer": true,
            "native_qwen35moe_forward": true,
            "recurrent_gated_delta_net": true,
            "full_attention_layers": true,
            "routed_top8_moe": true,
            "shared_expert": true,
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
            "experts": model.experts,
            "experts_used": model.experts_used,
            "expert_feed_forward_length": model.expert_ff,
            "shared_feed_forward_length": model.shared_ff,
            "ssm_inner_size": model.ssm_inner,
            "ssm_state_size": model.ssm_state,
            "ssm_group_count": model.ssm_groups,
            "ssm_time_step_rank": model.ssm_rank,
            "full_attention_interval": model.full_attention_interval,
            "rope_dimension": model.rope_dim,
            "rms_epsilon": model.rms_epsilon,
            "rope_freq_base": model.rope_freq_base
        },
        "input": {
            "prompt": args.prompt,
            "prompt_sha256": prompt_hash,
            "token_ids": prompt_tokens,
            "token_count": prompt_tokens.len(),
            "tokenizer_bos_id": model.tokenizer.bos_id(),
            "tokenizer_eos_id": model.tokenizer.eos_id(),
            "tokenizer_add_bos": model.tokenizer.add_bos()
        },
        "output": {
            "generated_token_ids": generated_ids,
            "generated_text": generated_text,
            "generated_token_count": generated_ids.len(),
            "selected_logit_f32": selected_logits,
            "logits_sha256": sha256_f32(&logits)
        },
        "routing": {
            "steps": route_json,
            "top_k": TOP_K,
            "selection": "softmax_all_experts_then_top_k_with_selected_weight_renormalization"
        },
        "remora": {
            "schema": har_metabolism::METABOLISM_SCHEMA,
            "initial_snapshot": remora_initial,
            "final_snapshot": remora_final,
            "telemetry_totals": remora_final.to_totals(),
            "safe_surplus": remora_safe_surplus,
            "control_basis": {
                "model_root": model_root,
                "graph_identity": "har.native.qwen35moe.hybrid.cpu.experimental.v1",
                "worker_set": format!("std-scoped-workers:{worker_count}")
            },
            "energy_scope": "UNKNOWN; no GPU energy counter was sampled"
        },
        "timing": {
            "timestamp_unix_s": parse_timestamp(),
            "model_load_and_map_ns": load_ns,
            "prompt_forward_ns": prompt_ns,
            "prompt_tokens_per_second": prompt_tokens.len() as f64 / (prompt_ns as f64 / 1_000_000_000.0).max(1.0e-12),
            "generated_forward_ns": generated_forward_ns,
            "generated_forward_count": generated_forward_ns.len()
        },
        "notes": [
            "This is a native Rust CPU path over real Qwen35MoE GGUF weights; no foreign inference backend is linked or called.",
            "Recurrent layers use the Qwen gated-delta-net autoregressive recurrence and preserve per-layer convolution/recurrent state.",
            "Full-attention layers use a text-mode partial RoPE approximation; multimodal/image position semantics are outside this text-only witness.",
            "The physical Q6_K stable-slot/GEMV artifacts are separate and are not silently attributed to this CPU forward.",
            "Energy remains UNKNOWN because no GPU energy counter was sampled.",
        ],
        "markers": [
            "HAR_NATIVE_QWEN35MOE_PROMPT_TO_TOKEN_PASS",
            "HAR_REAL_GGUF_WEIGHTS_CONSUMED",
            "HAR_NATIVE_ROUTER_TOP8_PASS",
            "HAR_NATIVE_GATED_DELTA_NET_PASS",
            "HAR_FULL_QWEN35MOE_VULKAN_LOOP_NOT_READY"
        ]
    });
    ensure_parent(&args.output)?;
    std::fs::write(&args.output, serde_json::to_vec_pretty(&output)?)?;
    println!("[HAR_NATIVE_QWEN35MOE_PROMPT_TO_TOKEN_PASS]");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_zero_block_decodes_to_zero() {
        let row = [0u8; Q8_0_BLOCK_BYTES];
        assert_eq!(q8_0_dot(&row, &[1.0; 32]).unwrap(), 0.0);
        let mut output = [1.0f32; 32];
        q8_0_decode_row(&row, &mut output).unwrap();
        assert!(output.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn top_k_tie_order_is_deterministic() {
        let mut order = vec![2usize, 0, 1];
        let probabilities = [1.0f32, 1.0, 1.0];
        order.sort_by(|left, right| {
            probabilities[*right]
                .total_cmp(&probabilities[*left])
                .then_with(|| left.cmp(right))
        });
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn softplus_is_finite_at_extremes() {
        assert!(softplus(-100.0).is_finite());
        assert!(softplus(100.0).is_finite());
    }
}
