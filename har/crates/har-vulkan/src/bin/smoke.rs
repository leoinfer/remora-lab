use ash::vk;
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
fn bytes_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn push_u32(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shader_dir = env::args()
        .nth(1)
        .unwrap_or_else(|| "har/shaders".to_owned());
    let device = Device::open(DeviceOptions::default())?;
    println!("{}", device.capabilities().to_json());
    let queue = device.create_queue(QueueKind::Compute)?;
    let slots = device.create_expert_slot_table(2, 4096)?;
    assert_eq!(slots.storage_buffer().size(), 8192);
    assert!(slots
        .storage_buffer()
        .usage()
        .contains(vk::BufferUsageFlags::TRANSFER_DST));
    assert!(slots
        .storage_buffer()
        .usage()
        .contains(vk::BufferUsageFlags::TRANSFER_SRC));

    let sampler_spv = read_spirv(&format!("{shader_dir}/greedy_argmax.spv"));
    let pipeline_cache = device.create_pipeline_cache(&[])?;
    let sampler = device.create_pipeline(
        &sampler_spv,
        &[(0, 1), (1, 1)],
        16,
        256,
        true,
        "greedy_argmax_wave32",
        Some(&pipeline_cache),
    )?;
    assert!(!pipeline_cache.data()?.is_empty());
    let vocabulary = 1024usize;
    let mut logits = vec![-1.0f32; vocabulary];
    logits[777] = 9.0;
    let logits_buffer = device.create_buffer(
        logits.len() * 4,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryPreference::HostVisible,
        "rust.smoke.logits",
    )?;
    logits_buffer.write(0, &bytes_f32(&logits))?;
    let output = device.create_buffer(
        4,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryPreference::HostVisible,
        "rust.smoke.sample.output",
    )?;
    let mut descriptors = queue.allocate_descriptor_set(&sampler)?;
    descriptors.update_storage(0, &logits_buffer, 0, vocabulary * 4)?;
    descriptors.update_storage(1, &output, 0, 4)?;
    let query = device.create_timestamp_query(2)?;
    let mut command = queue.allocate_command_buffer()?;
    command.begin()?;
    command.reset_query(&query)?;
    command.write_timestamp(&query, 0, vk::PipelineStageFlags::COMPUTE_SHADER)?;
    command.bind_pipeline(&sampler)?;
    command.bind_descriptor_set(&descriptors, &sampler)?;
    command.push_constants(&sampler, &push_u32(&[vocabulary as u32, 0, 0, 0]))?;
    command.dispatch(1, 1, 1)?;
    command.write_timestamp(&query, 1, vk::PipelineStageFlags::COMPUTE_SHADER)?;
    command.end()?;
    let ticket = queue.submit_compute(command, vec![descriptors], &[], vec![query.clone()])?;
    ticket.wait()?;
    let selected = u32::from_le_bytes(output.read(0, 4)?.try_into().unwrap());
    assert_eq!(selected, 777);

    let q4_spv = read_spirv(&format!("{shader_dir}/q4k_gemv.spv"));
    let q4 = device.create_pipeline(
        &q4_spv,
        &[(0, 1), (1, 1), (2, 1)],
        8,
        256,
        true,
        "q4_k_gemv_wave32",
        None,
    )?;
    let mut block = vec![0u8; 144];
    block[1] = 0x3c; // d=1.0 in binary16
    for byte in &mut block[4..16] {
        *byte = 1;
    }
    for byte in &mut block[16..] {
        *byte = 0x11;
    }
    let weights_host = device.create_buffer(
        144,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryPreference::HostVisible,
        "rust.smoke.q4.staging",
    )?;
    weights_host.write(0, &block)?;
    let weights = device.create_buffer(
        144,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryPreference::DeviceLocal,
        "rust.smoke.q4.weights",
    )?;
    let mut copy = queue.allocate_command_buffer()?;
    copy.begin()?;
    copy.copy_buffer(&weights_host, &weights, 0, 0, 144)?;
    copy.end()?;
    let transfer = queue.submit_transfer(copy, &[], Vec::new())?;
    transfer.wait()?;
    let input_values = vec![1.0f32; 256];
    let input = device.create_buffer(
        256 * 4,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryPreference::HostVisible,
        "rust.smoke.q4.input",
    )?;
    input.write(0, &bytes_f32(&input_values))?;
    let q4_output = device.create_buffer(
        4,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryPreference::HostVisible,
        "rust.smoke.q4.output",
    )?;
    let mut q4_descriptors = queue.allocate_descriptor_set(&q4)?;
    q4_descriptors.update_storage(0, &weights, 0, 144)?;
    q4_descriptors.update_storage(1, &input, 0, 256 * 4)?;
    q4_descriptors.update_storage(2, &q4_output, 0, 4)?;
    let q4_query = device.create_timestamp_query(2)?;
    let mut q4_command = queue.allocate_command_buffer()?;
    q4_command.begin()?;
    q4_command.reset_query(&q4_query)?;
    q4_command.write_timestamp(&q4_query, 0, vk::PipelineStageFlags::COMPUTE_SHADER)?;
    q4_command.bind_pipeline(&q4)?;
    q4_command.bind_descriptor_set(&q4_descriptors, &q4)?;
    q4_command.push_constants(&q4, &push_u32(&[1, 1]))?;
    q4_command.dispatch(1, 1, 1)?;
    q4_command.write_timestamp(&q4_query, 1, vk::PipelineStageFlags::COMPUTE_SHADER)?;
    q4_command.end()?;
    let q4_ticket = queue.submit_compute(
        q4_command,
        vec![q4_descriptors],
        &[],
        vec![q4_query.clone()],
    )?;
    q4_ticket.wait()?;
    let q4_value = f32::from_le_bytes(q4_output.read(0, 4)?.try_into().unwrap());
    assert!((q4_value - 256.0).abs() < 1e-4, "q4 value {q4_value}");
    println!(
        "PASS_RUST_VULKAN device={} sampler_token={} q4_value={} sampler_gpu_ns={} q4_gpu_ns={}",
        device.capabilities().name,
        selected,
        q4_value,
        query.elapsed_ns()?,
        q4_query.elapsed_ns()?
    );
    Ok(())
}
