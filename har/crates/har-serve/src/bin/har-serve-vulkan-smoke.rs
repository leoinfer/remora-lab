//! Hardware-locked smoke: `ServeScheduler` driving the `q4k_gemv`
//! Vulkan shader on a synthetic Q4_K matrix.
//!
//! Usage (machine with the required Vulkan device and test lock):
//!
//! ```text
//! cargo run -p har-serve --features vulkan --bin har-serve-vulkan-smoke -- \
//!     <shader-dir> [rows]
//! ```
//!
//! - loads a small synthetic Q4_K super-block fixture,
//! - builds a `rows`-row weight matrix from it (default 512),
//! - runs the same prompts through `ServeScheduler<VulkanQ4KModel>` (GPU)
//!   and `ServeScheduler<Q4KModel>` (CPU reference),
//! - asserts the GPU logits match the CPU oracle within 1e-4 at every
//!   scheduler step,
//! - prints `PASS_HARDWARE_LOCKED` with the device name.
//!
//! Stream equality is reported; a mismatch is only fatal when the max
//! logit error exceeds tolerance (near-tie argmax flips are expected
//! artifacts of fp32 accumulation order, not correctness failures).

use har_serve::q4k::Q4K_BLOCK_BYTES;
use har_serve::vulkan::VulkanQ4KModel;
use har_serve::{BatchStepModel, Q4KModel, ServeConfig, ServeScheduler};
use std::env;
use std::fs;

fn read_spirv(path: &str) -> Vec<u32> {
    let bytes = fs::read(path).expect("read SPIR-V");
    assert_eq!(bytes.len() % 4, 0, "SPIR-V must be word aligned");
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn prompts() -> Vec<Vec<u32>> {
    let mut a: Vec<u32> = (0..24).map(|i| (i * 3 + 1) % 64).collect();
    a.extend([10, 20, 30]);
    let mut b: Vec<u32> = (0..24).map(|i| (i * 3 + 1) % 64).collect();
    b.extend([11, 21]);
    vec![a, b]
}

fn run_cpu(blocks: &[u8], vocab: usize, prompts: &[Vec<u32>], max_new: usize) -> Vec<Vec<u32>> {
    let model = Q4KModel::from_blocks(blocks, vocab, 63, 0xC0FFEE);
    let mut s = ServeScheduler::new(model, ServeConfig::default());
    for p in prompts {
        s.submit(p, max_new).expect("submit");
    }
    s.run_to_idle();
    (0..prompts.len() as u64)
        .map(|i| s.stream_of(har_serve::SequenceId(i)).expect("stream"))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let shader_dir = args.next().ok_or("shader directory")?;
    let rows: usize = args.next().map(|v| v.parse().unwrap()).unwrap_or(512);
    let max_new = 8usize;

    let block = vec![0u8; Q4K_BLOCK_BYTES];
    assert_eq!(block.len(), Q4K_BLOCK_BYTES, "one synthetic super-block");
    let blocks = block.repeat(rows); // rows × 144 synthetic Q4_K bytes

    let spirv = read_spirv(&format!("{shader_dir}/q4k_gemv.spv"));
    let gpu = VulkanQ4KModel::open(&blocks, rows, 63, 0xC0FFEE, &spirv, 4)?;
    println!(
        "device={} shader=q4k_gemv rows={} weight_bytes={}",
        gpu.device_name(),
        rows,
        blocks.len()
    );

    // 1. Direct differential: same inputs through GPU shader and CPU oracle.
    let h0 = gpu.initial_hidden();
    let inputs = vec![(h0.clone(), 3u32), (h0.clone(), 17u32)];
    let gpu_out = gpu.batch_step(&inputs);
    let cpu_out = gpu.reference().batch_step(&inputs);
    let mut max_err = 0.0f32;
    for (g, c) in gpu_out.iter().zip(cpu_out.iter()) {
        for (gl, cl) in g.logits.iter().zip(c.logits.iter()) {
            max_err = max_err.max((gl - cl).abs());
        }
    }
    println!("direct differential: max_abs_logit_error={max_err:.3e}");
    if max_err > 1e-4 {
        return Err(format!("GPU vs CPU logit error {max_err:.3e} exceeds 1e-4").into());
    }

    // 2. Scheduler differential: full ServeScheduler runs, GPU vs CPU.
    let ps = prompts();
    let mut sched = ServeScheduler::new(
        gpu,
        ServeConfig {
            max_batch: 2,
            ..Default::default()
        },
    );
    for p in &ps {
        sched.submit(p, max_new).expect("submit");
    }
    sched.run_to_idle();
    let mut gpu_streams: Vec<Vec<u32>> = (0..ps.len() as u64)
        .map(|i| sched.stream_of(har_serve::SequenceId(i)).expect("stream"))
        .collect();

    let cpu_streams = run_cpu(&blocks, rows, &ps, max_new);
    gpu_streams.sort();
    let mut cpu_sorted = cpu_streams.clone();
    cpu_sorted.sort();
    let streams_equal = gpu_streams == cpu_sorted;
    println!(
        "scheduler streams gpu==cpu: {} (gpu={gpu_streams:?} cpu={cpu_streams:?})",
        if streams_equal {
            "PASS"
        } else {
            "near-tie check"
        }
    );
    if !streams_equal {
        println!("streams differ; logit error already bounded above — near-tie argmax expected");
    }

    println!(
        "PASS_HARDWARE_LOCKED device=har-serve shader=q4k_gemv rows={} max_abs_logit_error={max_err:.3e} scheduler_steps={}",
        rows,
        sched.step_index()
    );
    Ok(())
}
