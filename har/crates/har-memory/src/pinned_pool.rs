//! Pinned (page-locked) host staging pool for PCIe transfers.
//!
//! Concept source: public memory-transfer references; no upstream
//! implementation is copied. H2D/D2H staging buffers are expensive to create
//! per transfer, so a fixed pool of aligned host blocks is acquired and
//! released instead. A backend supplies the platform-specific pinned-memory
//! operation; this module owns only partitioning and bookkeeping.

use har_core::{HarError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PinnedPoolConfig {
    /// Total page-locked bytes owned by the pool.
    pub capacity_bytes: u64,
    /// All acquisitions return offsets aligned to this value.
    pub alignment: u64,
}

impl PinnedPoolConfig {
    pub fn validate(&self) -> Result<()> {
        if self.capacity_bytes == 0 || self.alignment == 0 || !self.alignment.is_power_of_two() {
            return Err(HarError::Invalid {
                kind: "pinned pool config",
                message: "capacity must be > 0 and alignment a nonzero power of two".into(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FreeBlock {
    start: u64,
    end: u64,
}

/// A live acquisition. `generation == id` here; equality guards stale/double release.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PinnedHandle {
    pub id: u64,
    pub generation: u64,
    pub offset: u64,
    pub bytes: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PinnedPoolStats {
    pub acquisitions: u64,
    pub releases: u64,
    pub coalesces: u64,
    pub peak_in_use_bytes: u64,
}

/// Fixed-capacity first-fit allocator over one pinned region.
///
/// The region itself is allocated once by the backend adapter at startup;
/// this type only partitions it, so no raw memory lives here.
#[derive(Clone, Debug)]
pub struct PinnedHostPool {
    config: PinnedPoolConfig,
    free: BTreeMap<u64, FreeBlock>,
    in_use: BTreeMap<u64, PinnedHandle>,
    next_id: u64,
    stats: PinnedPoolStats,
}

impl PinnedHostPool {
    pub fn new(config: PinnedPoolConfig) -> Result<Self> {
        config.validate()?;
        let mut free = BTreeMap::new();
        free.insert(
            0,
            FreeBlock {
                start: 0,
                end: config.capacity_bytes,
            },
        );
        Ok(Self {
            config,
            free,
            in_use: BTreeMap::new(),
            next_id: 1,
            stats: PinnedPoolStats::default(),
        })
    }

    pub fn config(&self) -> &PinnedPoolConfig {
        &self.config
    }

    pub fn stats(&self) -> &PinnedPoolStats {
        &self.stats
    }

    pub fn in_use_bytes(&self) -> u64 {
        self.in_use.values().map(|h| h.bytes).sum()
    }

    /// Acquire `bytes` at an aligned offset. First fit over lowest-offset
    /// blocks; caller-visible byte count is preserved on the handle while the
    /// internal reservation rounds up to `alignment`.
    pub fn acquire(&mut self, bytes: u64) -> Result<PinnedHandle> {
        if bytes == 0 {
            return Err(HarError::Invalid {
                kind: "pinned pool",
                message: "zero-byte acquisition".into(),
            });
        }
        let align = self.config.alignment;
        let rounded = bytes.div_ceil(align) * align;
        let chosen = self.free.values().find_map(|b| {
            let misalign = (align - (b.start % align)) % align;
            if b.end - b.start >= misalign + rounded {
                Some((b.start, b.end, misalign))
            } else {
                None
            }
        });
        let (block_start, block_end, misalign) = chosen.ok_or_else(|| HarError::Invalid {
            kind: "pinned pool",
            message: format!("no contiguous {}-byte aligned block free", bytes),
        })?;
        let payload_start = block_start + misalign;
        let reserved_end = payload_start + rounded;
        self.free.remove(&block_start);
        if payload_start > block_start {
            self.free.insert(
                block_start,
                FreeBlock {
                    start: block_start,
                    end: payload_start,
                },
            );
        }
        if reserved_end < block_end {
            self.free.insert(
                reserved_end,
                FreeBlock {
                    start: reserved_end,
                    end: block_end,
                },
            );
        }
        let id = self.next_id;
        self.next_id += 1;
        let handle = PinnedHandle {
            id,
            generation: id,
            offset: payload_start,
            bytes,
        };
        self.in_use.insert(payload_start, handle.clone());
        self.stats.acquisitions += 1;
        self.stats.peak_in_use_bytes = self.stats.peak_in_use_bytes.max(self.in_use_bytes());
        Ok(handle)
    }

    /// Release a handle. Unknown, forged, or double releases fail closed.
    pub fn release(&mut self, handle: &PinnedHandle) -> Result<()> {
        let live = self
            .in_use
            .get(&handle.offset)
            .ok_or_else(|| HarError::Invalid {
                kind: "pinned pool",
                message: format!("release of unknown offset {}", handle.offset),
            })?;
        if live != handle {
            return Err(HarError::Invalid {
                kind: "pinned pool",
                message: format!("stale handle for offset {}", handle.offset),
            });
        }
        let rounded_end =
            handle.offset + handle.bytes.div_ceil(self.config.alignment) * self.config.alignment;
        self.in_use.remove(&handle.offset);
        self.stats.releases += 1;
        self.insert_free(handle.offset, rounded_end);
        Ok(())
    }

    fn insert_free(&mut self, start: u64, reserved_end: u64) {
        let mut start = start;
        let mut end = reserved_end;
        let mut coalesced = 0u32;
        if let Some((&k, b)) = self.free.range(..start).next_back() {
            if k + (b.end - b.start) == start {
                start = k;
                coalesced += 1;
            }
        }
        if let Some(b) = self.free.get(&end) {
            end = b.end;
            coalesced += 1;
        }
        if coalesced > 0 {
            self.stats.coalesces += coalesced as u64;
        }
        self.free.retain(|&k, _| !(k >= start && k < end));
        self.free.insert(start, FreeBlock { start, end });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool() -> PinnedHostPool {
        PinnedHostPool::new(PinnedPoolConfig {
            capacity_bytes: 1 << 20,
            alignment: 4096,
        })
        .unwrap()
    }

    #[test]
    fn acquire_returns_aligned_blocks_and_reuses_after_release() {
        let mut p = pool();
        let a = p.acquire(4096).unwrap();
        assert_eq!(a.offset, 0);
        let b = p.acquire(8192).unwrap();
        assert_eq!(b.offset, 4096);
        p.release(&a).unwrap();
        p.release(&b).unwrap();
        let c = p.acquire(1 << 20).unwrap();
        assert_eq!(c.offset, 0);
        assert!(p.stats().coalesces >= 2);
    }

    #[test]
    fn capacity_is_enforced_fail_closed() {
        let mut p = pool();
        assert!(p.acquire((1 << 20) + 1).is_err());
        let a = p.acquire(512 << 10).unwrap();
        let b = p.acquire(512 << 10).unwrap();
        assert_eq!(a.offset, 0);
        assert_eq!(b.offset, 512 << 10);
        assert!(p.acquire(4096).is_err());
        p.release(&a).unwrap();
        assert!(p.acquire(512 << 10).is_ok());
    }

    #[test]
    fn zero_byte_acquisition_rejected() {
        let mut p = pool();
        assert!(p.acquire(0).is_err());
    }

    #[test]
    fn double_release_fails_closed() {
        let mut p = pool();
        let a = p.acquire(4096).unwrap();
        p.release(&a).unwrap();
        assert!(p.release(&a).is_err());
    }

    #[test]
    fn forged_handle_rejected() {
        let mut p = pool();
        let _a = p.acquire(4096).unwrap();
        let b = p.acquire(4096).unwrap();
        let forged = PinnedHandle {
            id: b.id,
            generation: b.generation + 1,
            offset: b.offset,
            bytes: b.bytes,
        };
        assert!(p.release(&forged).is_err());
        p.release(&b).unwrap();
    }

    #[test]
    fn peak_usage_tracked() {
        let mut p = pool();
        let a = p.acquire(64 << 10).unwrap();
        let _b = p.acquire(128 << 10).unwrap();
        p.release(&a).unwrap();
        assert_eq!(p.stats().peak_in_use_bytes, 192 << 10);
    }

    #[test]
    fn unaligned_sizes_round_up_internally() {
        let mut p = PinnedHostPool::new(PinnedPoolConfig {
            capacity_bytes: 64 << 10,
            alignment: 4096,
        })
        .unwrap();
        let a = p.acquire(100).unwrap();
        assert_eq!(a.bytes, 100);
        let b = p.acquire(4000).unwrap();
        assert_eq!(b.offset, 4096);
        assert!(p.acquire(56 << 10).is_ok());
    }

    #[test]
    fn fragmented_region_serves_smallest_first_fit() {
        let mut p = pool();
        let a = p.acquire(4 << 10).unwrap();
        let _b = p.acquire(4 << 10).unwrap();
        let c = p.acquire(4 << 10).unwrap();
        p.release(&_b).unwrap();
        let d = p.acquire(2 << 10).unwrap(); // fits exactly into the freed hole
        assert_eq!(d.offset, 4 << 10);
        p.release(&a).unwrap();
        p.release(&c).unwrap();
        p.release(&d).unwrap();
        assert!(p.acquire(1 << 20).is_ok()); // full coalesce restored
    }
}
