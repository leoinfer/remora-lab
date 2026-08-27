//! Vulkan Q4_K GEMV [`BatchStepModel`] — the real-kernel seam for
//! `har-serve` (feature `vulkan`).
//!
//! Wraps `har-vulkan` around the existing `q4k_gemv.comp` shader
//! (one workgroup per row, 256 lanes, 144-byte Q4_K super-blocks): the
//! weights live in a device-local buffer, each sequence's activations are
//! staged host-visible, and one command buffer records N dispatches (one
//! per sequence) submitted together.  The recurrent part (embedding add,
//! hidden carry) stays on the CPU exactly as in the reference
//! [`crate::q4k::Q4KModel`], so the GPU model and the CPU oracle share
//! geometry and semantics.
//!
//! Honest caveat: the current shader is *per-sequence* — N dispatches read
//! the weight set N times, so this adapter validates scheduling semantics
//! and numerical correctness against real quantized weights, not the
//! bandwidth amortization (that needs the batched kernel, next milestone).
//!
//! Fail-closed: any GPU error panics (CORRECTNESS_POLICY — no silent
//! fallback).

use crate::adapter::{BatchStepModel, Hidden, Logits, StepOutcome};
use crate::q4k::{Q4KModel, Q4K_BLOCK_BYTES, Q4K_BLOCK_VALUES};
use ash::vk;
use har_vulkan::{Device, DeviceOptions, MemoryPreference, QueueKind};

pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub struct VulkanQ4KModel {
    device: Device,
    queue: har_vulkan::Queue,
    pipeline: har_vulkan::Pipeline,
    weights: har_vulkan::Buffer,
    inputs: Vec<har_vulkan::Buffer>,
    outputs: Vec<har_vulkan::Buffer>,
    cpu: Q4KModel,
}

impl VulkanQ4KModel {
    /// Open the device, build the `q4k_gemv` pipeline from `spirv`, stage
    /// `blocks` (rows × 144 bytes, one Q4_K super-block per row) into
    /// device-local memory, and allocate `max_batch` per-sequence
    /// input/output buffers.
    pub fn open(
        blocks: &[u8],
        vocab: usize,
        eos: u32,
        seed: u64,
        spirv: &[u32],
        max_batch: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        assert_eq!(blocks.len() % Q4K_BLOCK_BYTES, 0, "block-aligned weights");
        let rows = blocks.len() / Q4K_BLOCK_BYTES;
        assert_eq!(rows, vocab, "one row per vocab entry (shader contract)");

        let device = Device::open(DeviceOptions::default())?;
        let queue = device.create_queue(QueueKind::Compute)?;
        let pipeline = device.create_pipeline(
            spirv,
            &[(0, 1), (1, 1), (2, 1)],
            8,
            256,
            true,
            "q4k_gemv_wave32",
            None,
        )?;

        // Weights: staging (host-visible) -> device-local.
        let staging = device.create_buffer(
            blocks.len(),
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryPreference::HostVisible,
            "har-serve.q4k.staging",
        )?;
        staging.write(0, blocks)?;
        let weights = device.create_buffer(
            blocks.len(),
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryPreference::DeviceLocal,
            "har-serve.q4k.weights",
        )?;
        let mut copy = queue.allocate_command_buffer()?;
        copy.begin()?;
        copy.copy_buffer(&staging, &weights, 0, 0, blocks.len())?;
        copy.end()?;
        queue.submit_transfer(copy, &[], Vec::new())?.wait()?;

        let mut inputs = Vec::with_capacity(max_batch);
        let mut outputs = Vec::with_capacity(max_batch);
        for i in 0..max_batch {
            inputs.push(device.create_buffer(
                Q4K_BLOCK_VALUES * 4,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryPreference::HostVisible,
                &format!("har-serve.q4k.input.{i}"),
            )?);
            outputs.push(device.create_buffer(
                rows * 4,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryPreference::HostVisible,
                &format!("har-serve.q4k.output.{i}"),
            )?);
        }

        Ok(Self {
            device,
            queue,
            pipeline,
            weights,
            inputs,
            outputs,
            cpu: Q4KModel::from_blocks(blocks, vocab, eos, seed),
        })
    }

    pub fn device_name(&self) -> String {
        self.device.capabilities().name.clone()
    }

    /// The CPU reference model (same weights/geometry) — the differential
    /// oracle for tests and the smoke binary.
    pub fn reference(&self) -> &Q4KModel {
        &self.cpu
    }

    /// One batched forward: N dispatches of the q4k_gemv shader recorded
    /// into a single command buffer and submitted once.
    fn forward_gpu(&self, xs: &[Vec<f32>]) -> Vec<Logits> {
        let n = xs.len();
        assert!(n <= self.inputs.len(), "batch exceeds allocated buffers");
        let rows = self.cpu.rows as u32;

        for (i, x) in xs.iter().enumerate() {
            self.inputs[i]
                .write(0, &f32_bytes(x))
                .expect("input staging write");
        }

        let mut command = self
            .queue
            .allocate_command_buffer()
            .expect("command buffer alloc");
        command.begin().expect("command begin");
        command
            .bind_pipeline(&self.pipeline)
            .expect("bind pipeline");
        let mut sets = Vec::with_capacity(n);
        for i in 0..n {
            let mut set = self
                .queue
                .allocate_descriptor_set(&self.pipeline)
                .expect("descriptor alloc");
            set.update_storage(0, &self.weights, 0, self.cpu.rows * Q4K_BLOCK_BYTES)
                .expect("weights binding");
            set.update_storage(1, &self.inputs[i], 0, Q4K_BLOCK_VALUES * 4)
                .expect("input binding");
            set.update_storage(2, &self.outputs[i], 0, self.cpu.rows * 4)
                .expect("output binding");
            command
                .bind_descriptor_set(&set, &self.pipeline)
                .expect("bind set");
            command
                .push_constants(&self.pipeline, &u32_bytes(&[rows, 1]))
                .expect("push constants");
            command.dispatch(rows, 1, 1).expect("dispatch");
            sets.push(set);
        }
        command.end().expect("command end");
        self.queue
            .submit_compute(command, sets, &[], Vec::new())
            .expect("submit")
            .wait()
            .expect("ticket wait");

        (0..n)
            .map(|i| {
                let bytes = self.outputs[i]
                    .read(0, self.cpu.rows * 4)
                    .expect("output readback");
                bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect()
            })
            .collect()
    }
}

impl BatchStepModel for VulkanQ4KModel {
    fn batch_step(&self, inputs: &[(Hidden, u32)]) -> Vec<StepOutcome> {
        // Same recurrent semantics as Q4KModel: x = h + E[t], logits =
        // W·x, next hidden = x.  Only the matvec moves to the GPU.
        let d = Q4K_BLOCK_VALUES;
        let xs: Vec<Vec<f32>> = inputs
            .iter()
            .map(|(h, t)| {
                (0..d)
                    .map(|i| h[i] + self.cpu.embed[*t as usize * d + i])
                    .collect()
            })
            .collect();
        let logits = self.forward_gpu(&xs);
        xs.into_iter()
            .zip(logits)
            .map(|(x, l)| StepOutcome::plain(x, l))
            .collect()
    }

    fn initial_hidden(&self) -> Hidden {
        self.cpu.initial_hidden()
    }

    fn eos(&self) -> u32 {
        self.cpu.eos()
    }

    fn weight_bytes_per_row(&self) -> u64 {
        self.cpu.weight_bytes_per_row()
    }
}

/// Quantization formats supported by the batched kernels.  Both use a
/// 256-wide row: Q4_K = one 144-byte super-block, Q4_0 = eight 18-byte
/// blocks (the caller-selected GGUF-compatible format).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchedFormat {
    Q4K,
    Q40,
}

impl BatchedFormat {
    pub fn row_bytes(self) -> usize {
        match self {
            BatchedFormat::Q4K => 144,
            BatchedFormat::Q40 => 144,
        }
    }
    /// Bytes consumed by the GPU's word-aligned staging layout per row.
    /// Q4_0's packed 18-byte blocks are padded to five 32-bit words so the
    /// shader can cooperatively load each block without cross-word reads.
    pub fn storage_row_bytes(self) -> usize {
        match self {
            BatchedFormat::Q4K => 144,
            BatchedFormat::Q40 => 160,
        }
    }
    pub fn blocks_per_row(self) -> u32 {
        match self {
            BatchedFormat::Q4K => 1,
            BatchedFormat::Q40 => 8,
        }
    }
    pub fn shader_name(self) -> &'static str {
        match self {
            BatchedFormat::Q4K => "q4k_batched_gemv",
            BatchedFormat::Q40 => "q4_0_batched_gemv",
        }
    }
    fn gpu_storage(&self, blocks: &[u8]) -> Vec<u8> {
        if *self == BatchedFormat::Q4K {
            return blocks.to_vec();
        }
        let mut padded = Vec::with_capacity(blocks.len() / 18 * 20);
        for block in blocks.chunks_exact(18) {
            padded.extend_from_slice(block);
            padded.extend_from_slice(&[0, 0]);
        }
        assert_eq!(blocks.len() % 18, 0, "Q4_0 blocks must be complete");
        padded
    }
}

/// Batched Q4_K / Q4_0 GEMV model: the whole scheduler batch is served by
/// ONE shader dispatch, so the weight set streams from VRAM once per step
/// instead of once per sequence — the bandwidth amortization the serving
/// layer exists for.
///
/// Bindings follow the batched shaders: (0) weights rows×row_bytes,
/// (1) activations batch×256 f32, (2) output rows×batch f32 (row-major
/// over the batch).  Push constants: rows, blocks_per_row, batch.
pub struct VulkanBatchedModel {
    device: Device,
    queue: har_vulkan::Queue,
    pipeline: har_vulkan::Pipeline,
    weights: har_vulkan::Buffer,
    activations: har_vulkan::Buffer,
    output: har_vulkan::Buffer,
    rows: usize,
    max_batch: usize,
    embed: Vec<f32>,
    eos: u32,
    format: BatchedFormat,
    /// CPU oracle (same weights/geometry) for differential tests.
    reference: Box<dyn BatchStepModel>,
}

impl VulkanBatchedModel {
    pub fn open(
        format: BatchedFormat,
        blocks: &[u8],
        vocab: usize,
        eos: u32,
        seed: u64,
        spirv: &[u32],
        max_batch: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let row_bytes = format.row_bytes();
        assert_eq!(blocks.len() % row_bytes, 0, "row-aligned weights");
        let rows = blocks.len() / row_bytes;
        assert_eq!(rows, vocab, "one row per vocab entry");
        assert!(max_batch <= 8, "batched kernels bound the batch at 8");
        let gpu_blocks = format.gpu_storage(blocks);

        let device = Device::open(DeviceOptions::default())?;
        let queue = device.create_queue(QueueKind::Compute)?;
        let pipeline = device.create_pipeline(
            spirv,
            &[(0, 1), (1, 1), (2, 1)],
            12,
            256,
            true,
            format.shader_name(),
            None,
        )?;

        let staging = device.create_buffer(
            gpu_blocks.len(),
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryPreference::HostVisible,
            "har-serve.batched.staging",
        )?;
        staging.write(0, &gpu_blocks)?;
        let weights = device.create_buffer(
            gpu_blocks.len(),
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryPreference::DeviceLocal,
            "har-serve.batched.weights",
        )?;
        let mut copy = queue.allocate_command_buffer()?;
        copy.begin()?;
        copy.copy_buffer(&staging, &weights, 0, 0, gpu_blocks.len())?;
        copy.end()?;
        queue.submit_transfer(copy, &[], Vec::new())?.wait()?;

        let activations = device.create_buffer(
            max_batch * 256 * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryPreference::HostVisible,
            "har-serve.batched.activations",
        )?;
        let output = device.create_buffer(
            rows * max_batch * 4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryPreference::HostVisible,
            "har-serve.batched.output",
        )?;

        let reference: Box<dyn BatchStepModel> = match format {
            BatchedFormat::Q4K => {
                Box::new(crate::q4k::Q4KModel::from_blocks(blocks, vocab, eos, seed))
            }
            BatchedFormat::Q40 => {
                Box::new(crate::q40::Q40Model::from_blocks(blocks, vocab, eos, seed))
            }
        };

        Ok(Self {
            device,
            queue,
            pipeline,
            weights,
            activations,
            output,
            rows,
            max_batch,
            embed: crate::q4k::lcg_values(vocab * 256, seed),
            eos,
            format,
            reference,
        })
    }

    pub fn device_name(&self) -> String {
        self.device.capabilities().name.clone()
    }
    pub fn format(&self) -> BatchedFormat {
        self.format
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn reference(&self) -> &dyn BatchStepModel {
        self.reference.as_ref()
    }

    /// One dispatch for the whole batch.  Fail-closed on any GPU error.
    fn forward_gpu(&self, xs: &[Vec<f32>]) -> Vec<Logits> {
        let n = xs.len();
        assert!(n <= self.max_batch, "batch exceeds allocated buffers");
        for (i, x) in xs.iter().enumerate() {
            self.activations
                .write(i * 256 * 4, &f32_bytes(x))
                .expect("activation staging write");
        }
        let mut command = self
            .queue
            .allocate_command_buffer()
            .expect("command buffer alloc");
        command.begin().expect("command begin");
        command
            .bind_pipeline(&self.pipeline)
            .expect("bind pipeline");
        let mut set = self
            .queue
            .allocate_descriptor_set(&self.pipeline)
            .expect("descriptor alloc");
        set.update_storage(
            0,
            &self.weights,
            0,
            self.rows * self.format.storage_row_bytes(),
        )
        .expect("weights binding");
        set.update_storage(1, &self.activations, 0, n * 256 * 4)
            .expect("activations binding");
        set.update_storage(2, &self.output, 0, self.rows * n * 4)
            .expect("output binding");
        command
            .bind_descriptor_set(&set, &self.pipeline)
            .expect("bind set");
        command
            .push_constants(
                &self.pipeline,
                &u32_bytes(&[self.rows as u32, self.format.blocks_per_row(), n as u32]),
            )
            .expect("push constants");
        command.dispatch(self.rows as u32, 1, 1).expect("dispatch");
        command.end().expect("command end");
        self.queue
            .submit_compute(command, vec![set], &[], Vec::new())
            .expect("submit")
            .wait()
            .expect("ticket wait");

        // Output is row-major over the batch: out[row*batch + n].
        let bytes = self
            .output
            .read(0, self.rows * n * 4)
            .expect("output readback");
        let all: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        (0..n)
            .map(|seq| (0..self.rows).map(|r| all[r * n + seq]).collect())
            .collect()
    }
}

impl BatchStepModel for VulkanBatchedModel {
    fn batch_step(&self, inputs: &[(Hidden, u32)]) -> Vec<StepOutcome> {
        let xs: Vec<Vec<f32>> = inputs
            .iter()
            .map(|(h, t)| {
                (0..256)
                    .map(|i| h[i] + self.embed[*t as usize * 256 + i])
                    .collect()
            })
            .collect();
        let logits = self.forward_gpu(&xs);
        xs.into_iter()
            .zip(logits)
            .map(|(x, l)| StepOutcome::plain(x, l))
            .collect()
    }

    fn initial_hidden(&self) -> Hidden {
        vec![0.0f32; 256]
    }

    fn eos(&self) -> u32 {
        self.eos
    }

    fn weight_bytes_per_row(&self) -> u64 {
        (self.rows * self.format.storage_row_bytes()) as u64
    }
}
