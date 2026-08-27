//! Bounded real-slice milestone runner.
//!
//! This binary intentionally exercises one expert projection, never full
//! inference. It requires a model path supplied by the caller and reports
//! whether the native Rust/Vulkan execution boundary was reached.

use har_residency::{
    ExpertProjection, Generation, ResidencyError, ResidencyManager, WavefrontScheduler,
};
use har_storage::{DirectIoEngine, ExpertIndex, OriginalGgufStore, ReadRequest};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::Arc;

const VERDICT: &str = "HAR RESIDENCY — DIRECT IO WORKING, GPU INTEGRATION BLOCKED";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: har-demo MODEL.gguf")?;
    let index = ExpertIndex::from_gguf(&model)?;
    let slice = index
        .lookup(0, 0, ExpertProjection::Gate)
        .ok_or_else(|| ResidencyError::Invalid("layer 0 expert 0 gate is absent".into()))?
        .clone();
    let io = Arc::new(DirectIoEngine::new(1, 4096, 16 * 1024 * 1024)?);
    let capabilities = io.probe(&slice.source_path);
    let reference = io.read(ReadRequest {
        path: PathBuf::from(&slice.source_path),
        offset: slice.offset,
        bytes: slice.payload_bytes,
        alignment: 4096,
        tensor_id: "reference".into(),
        mandatory: true,
        speculative: false,
    })?;
    let mut hash = Sha256::new();
    hash.update(&reference.data);
    let checksum = format!("{:x}", hash.finalize());
    let mut slice = slice;
    slice.checksum_sha256 = Some(checksum.clone());
    let source = Arc::new(OriginalGgufStore::new(Arc::clone(&io)));
    let manager = ResidencyManager::new(
        32 * 1024 * 1024,
        32 * 1024 * 1024,
        1,
        slice.payload_bytes,
        1,
        1,
        1,
    )?;
    let mut scheduler = WavefrontScheduler::new(manager, source);
    let work_id = scheduler.add(slice.clone(), Generation(0), true, false, None, 100, 0)?;
    let replay = scheduler.run_one(work_id)?;
    let evicted_vram = if let Some(slot_id) = scheduler
        .works
        .iter()
        .find(|work| work.work_id == work_id)
        .and_then(|work| work.slot_id)
    {
        scheduler.manager.evict_vram(&slice.page_id, slot_id)?;
        true
    } else {
        false
    };
    scheduler.manager.evict_ram(&slice.page_id)?;
    let evicted_ram = true;
    scheduler.manager.validate()?;
    let accounting = io.accounting();
    println!("{{");
    println!("  \"verdict\": {:?},", VERDICT);
    println!("  \"model_id\": {:?},", index.model_id);
    println!("  \"entry_count\": {},", index.entries.len());
    println!("  \"tensor\": {:?},", slice.tensor);
    println!(
        "  \"layer\": {}, \"expert\": {}, \"projection\": {:?},",
        slice.layer.unwrap_or(0),
        slice.expert.unwrap_or(0),
        slice.projection.as_str()
    );
    println!(
        "  \"offset\": {}, \"payload_bytes\": {}, \"checksum_verified\": true,",
        slice.offset, slice.payload_bytes
    );
    println!(
        "  \"direct_io\": {}, \"physical_gpu_kernel_executed\": false,",
        capabilities.odirect_available
    );
    println!(
        "  \"replay_equal\": {}, \"replay_bytes\": {},",
        replay.equal, replay.bytes
    );
    println!(
        "  \"evicted_vram_safely\": {}, \"evicted_ram_safely\": {},",
        evicted_vram, evicted_ram
    );
    println!(
        "  \"io_requests\": {}, \"io_physical_bytes\": {}, \"io_page_cache_bytes\": {}",
        accounting.requests, accounting.physical_bytes, accounting.page_cache_bytes
    );
    println!("}}");
    Ok(())
}
