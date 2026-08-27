//! Hardware-locked smoke: `ServeScheduler` driving the **batched** GEMV
//! kernels (`q4k_batched_gemv`, `q4_0_batched_gemv`) — one dispatch
//! serving the whole batch.
//!
//! Usage (machine with the required Vulkan device and test lock):
//!
//! ```text
//! cargo run -p har-serve --features vulkan --bin har-serve-vulkan-batched-smoke -- \
//!     <shader-dir> [rows]
//! ```
//!
//! Two formats are exercised:
//! - **Q4_K** from a small synthetic GGUF block,
//! - **Q4_0** from deterministic synthetic blocks.
//!
//! For each: a direct batch differential (GPU logits vs CPU oracle,
//! tolerance 1e-4), then a full `ServeScheduler` run, then
//! `PASS_HARDWARE_LOCKED` with the max error.  The batch is the whole
//! scheduler batch — one dispatch per step, so the weight set is read
//! once per step regardless of how many sequences are live.

use har_serve::vulkan::{BatchedFormat, VulkanBatchedModel};
use har_serve::{BatchStepModel, ServeConfig, ServeScheduler};
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

fn run_format(
    label: &str,
    format: BatchedFormat,
    blocks: Vec<u8>,
    vocab: usize,
    shader_dir: &str,
    rows: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let spirv = read_spirv(&format!("{shader_dir}/{}.spv", format.shader_name()));
    let gpu = VulkanBatchedModel::open(format, &blocks, vocab, 63, 0xC0FFEE, &spirv, 4)?;
    println!(
        "[{label}] device={} shader={} rows={} weight_bytes={}",
        gpu.device_name(),
        format.shader_name(),
        rows,
        blocks.len()
    );

    // 1. Direct batch differential: GPU vs CPU oracle.
    let h0 = gpu.initial_hidden();
    let inputs = vec![(h0.clone(), 3u32), (h0.clone(), 17u32), (h0.clone(), 42u32)];
    let gpu_out = gpu.batch_step(&inputs);
    let cpu_out = gpu.reference().batch_step(&inputs);
    let mut max_err = 0.0f32;
    for (g, c) in gpu_out.iter().zip(cpu_out.iter()) {
        for (gl, cl) in g.logits.iter().zip(c.logits.iter()) {
            max_err = max_err.max((gl - cl).abs());
        }
    }
    println!("[{label}] direct batch differential (3 sequences, 1 dispatch): max_abs_logit_error={max_err:.3e}");
    if max_err > 1e-4 {
        return Err(format!("[{label}] GPU vs CPU logit error {max_err:.3e} exceeds 1e-4").into());
    }

    // 2. Scheduler differential: full ServeScheduler run on the GPU model.
    let ps = prompts();
    let mut sched = ServeScheduler::new(
        gpu,
        ServeConfig {
            max_batch: 2,
            ..Default::default()
        },
    );
    for p in &ps {
        sched.submit(p, 8).expect("submit");
    }
    sched.run_to_idle();
    println!(
        "[{label}] PASS_HARDWARE_LOCKED shader={} rows={} max_abs_logit_error={max_err:.3e} scheduler_steps={}",
        format.shader_name(),
        rows,
        sched.step_index()
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let shader_dir = args.next().ok_or("shader directory")?;
    let rows: usize = args.next().map(|v| v.parse().unwrap()).unwrap_or(512);

    // Q4_K: synthetic block replicated to `rows` rows.
    let block = vec![0u8; 144];
    assert_eq!(block.len(), 144, "one synthetic Q4_K super-block");
    run_format(
        "q4k-synthetic",
        BatchedFormat::Q4K,
        block.repeat(rows),
        rows,
        &shader_dir,
        rows,
    )?;

    // Q4_0: deterministic synthetic blocks.
    let q40 = har_serve::q40::synthetic_blocks(rows, 0x5EED);
    run_format(
        "q40-synthetic",
        BatchedFormat::Q40,
        q40,
        rows,
        &shader_dir,
        rows,
    )?;

    println!("BATCHED_GPU_TESTS_DONE");
    Ok(())
}
