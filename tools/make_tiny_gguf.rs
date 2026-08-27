//! Create a tiny deterministic Q4_0 GGUF fixture for local runtime tracing.
//!
//! The generated file is a synthetic test input, not a model release. It is
//! written to the caller's path and is never part of the repository.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: make-tiny-gguf OUTPUT")?;
    let rows = 16usize;
    let columns = 256usize;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());

    let key = b"general.alignment";
    bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
    bytes.extend_from_slice(key);
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&32u32.to_le_bytes());

    let tensor = b"token_embd.weight";
    bytes.extend_from_slice(&(tensor.len() as u64).to_le_bytes());
    bytes.extend_from_slice(tensor);
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&(columns as u64).to_le_bytes());
    bytes.extend_from_slice(&(rows as u64).to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    while bytes.len() % 32 != 0 {
        bytes.push(0);
    }

    // Q4_0: two-byte half scale plus 16 packed nibbles per 32 values.
    for _ in 0..rows * (columns / 32) {
        bytes.extend_from_slice(&[0x00, 0x3c]);
        bytes.extend((0..16u8).map(|value| value.wrapping_mul(3)));
    }
    fs::write(&path, bytes)?;
    println!("wrote synthetic GGUF fixture to {}", path.display());
    Ok(())
}
