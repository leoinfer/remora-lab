//! Hardware-locked smoke: `flash_decode_q8` — paged-KV flash-decode
//! attention (GQA, q8_0 KV cache, online softmax) vs the CPU reference.
//!
//! Usage (machine with the required Vulkan device and test lock):
//!
//! ```text
//! cargo run -p har-serve --features vulkan --bin har-serve-vulkan-attn-smoke -- \
//!     <shader-dir> [seq-len] [q-heads] [kv-heads]
//! ```
//!
//! Generates deterministic synthetic q8_0 KV + queries, runs the shader
//! (one workgroup per query head), compares every output against the
//! CPU reference within 1e-3, and prints `PASS_HARDWARE_LOCKED`.

use ash::vk;
use har_serve::attention::{
    flash_decode_reference, q8_0_quantize, row_to_words, FlashDecodeParams, HEAD_DIM, KV_ROW_BYTES,
};
use har_serve::vulkan::f32_bytes;
use har_vulkan::{Device, DeviceOptions, MemoryPreference, QueueKind};
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

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn lcg(count: usize, seed: u64) -> Vec<f32> {
    har_serve::q4k::lcg_values(count, seed)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let shader_dir = args.next().ok_or("shader directory")?;
    let seq_len: usize = args.next().map(|v| v.parse().unwrap()).unwrap_or(1024);
    let q_heads: usize = args.next().map(|v| v.parse().unwrap()).unwrap_or(8);
    let kv_heads: usize = args.next().map(|v| v.parse().unwrap()).unwrap_or(4);
    assert_eq!(
        q_heads % kv_heads,
        0,
        "GQA requires q_heads % kv_heads == 0"
    );
    let params = FlashDecodeParams {
        q_heads,
        kv_heads,
        seq_len,
        chunk: 1024,
    };

    // Deterministic synthetic KV (q8_0) and queries.
    let mut k_words = Vec::new();
    let mut v_words = Vec::new();
    for h in 0..kv_heads {
        for p in 0..seq_len {
            let k = lcg(HEAD_DIM, 1000 + (h * seq_len + p) as u64 * 31);
            let v = lcg(HEAD_DIM, 2000 + (h * seq_len + p) as u64 * 17);
            k_words.extend(row_to_words(&q8_0_quantize(&k)));
            v_words.extend(row_to_words(&q8_0_quantize(&v)));
        }
    }
    let query = lcg(q_heads * HEAD_DIM, 3000);

    // CPU reference.
    let expected = flash_decode_reference(&params, &k_words, &v_words, &query);
    let row_words = KV_ROW_BYTES / 4;

    // GPU: stage K/V/q, dispatch one WG per query head.
    let device = Device::open(DeviceOptions::default())?;
    let queue = device.create_queue(QueueKind::Compute)?;
    let spirv = read_spirv(&format!("{shader_dir}/flash_decode_q8.spv"));
    let pipeline = device.create_pipeline(
        &spirv,
        &[(0, 1), (1, 1), (2, 1), (3, 1)],
        12,
        256,
        true,
        "flash_decode_q8_wave32",
        None,
    )?;

    let k_bytes = k_words.len() * 4;
    let v_bytes = v_words.len() * 4;
    let q_bytes = query.len() * 4;
    let o_bytes = q_heads * HEAD_DIM * 4;

    let host_k = device.create_buffer(
        k_bytes,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryPreference::HostVisible,
        "attn.k.host",
    )?;
    host_k.write(0, &u32_bytes(&k_words))?;
    let host_v = device.create_buffer(
        v_bytes,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryPreference::HostVisible,
        "attn.v.host",
    )?;
    host_v.write(0, &u32_bytes(&v_words))?;

    let k_buf = device.create_buffer(
        k_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryPreference::DeviceLocal,
        "attn.k",
    )?;
    let v_buf = device.create_buffer(
        v_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryPreference::DeviceLocal,
        "attn.v",
    )?;
    let mut copy = queue.allocate_command_buffer()?;
    copy.begin()?;
    copy.copy_buffer(&host_k, &k_buf, 0, 0, k_bytes)?;
    copy.copy_buffer(&host_v, &v_buf, 0, 0, v_bytes)?;
    copy.end()?;
    queue.submit_transfer(copy, &[], Vec::new())?.wait()?;

    let q_buf = device.create_buffer(
        q_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryPreference::HostVisible,
        "attn.q",
    )?;
    q_buf.write(0, &f32_bytes(&query))?;
    let o_buf = device.create_buffer(
        o_bytes,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryPreference::HostVisible,
        "attn.o",
    )?;

    let mut set = queue.allocate_descriptor_set(&pipeline)?;
    set.update_storage(0, &k_buf, 0, k_bytes)?;
    set.update_storage(1, &v_buf, 0, v_bytes)?;
    set.update_storage(2, &q_buf, 0, q_bytes)?;
    set.update_storage(3, &o_buf, 0, o_bytes)?;

    let mut command = queue.allocate_command_buffer()?;
    command.begin()?;
    command.bind_pipeline(&pipeline)?;
    command.bind_descriptor_set(&set, &pipeline)?;
    command.push_constants(
        &pipeline,
        &u32_bytes(&[q_heads as u32, kv_heads as u32, seq_len as u32]),
    )?;
    command.dispatch(q_heads as u32, 1, 1)?;
    command.end()?;
    queue
        .submit_compute(command, vec![set], &[], Vec::new())?
        .wait()?;

    let out_bytes = o_buf.read(0, o_bytes)?;
    let actual: Vec<f32> = out_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    let mut max_err = 0.0f32;
    for (a, e) in actual.iter().zip(expected.iter()) {
        max_err = max_err.max((a - e).abs());
    }
    println!(
        "device={} shader=flash_decode_q8 q_heads={} kv_heads={} seq_len={} kv_bytes={} max_abs_error={max_err:.3e}",
        device.capabilities().name,
        q_heads,
        kv_heads,
        seq_len,
        k_bytes + v_bytes
    );
    if max_err > 1e-3 {
        return Err(format!("flash decode error {max_err:.3e} exceeds 1e-3").into());
    }
    let _ = row_words;
    println!("PASS_HARDWARE_LOCKED shader=flash_decode_q8 max_abs_error={max_err:.3e}");
    Ok(())
}
