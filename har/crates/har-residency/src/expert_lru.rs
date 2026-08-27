//! Timestamp-LRU global expert cache — a deterministic baseline policy.
//!
//! Concept source: public expert-cache and residency literature; no upstream
//! implementation is copied. This module provides a deterministic baseline
//! with hit/miss/eviction telemetry so a future policy can be compared on
//! identical access streams.

use crate::types::{PageId, ResidencyError, Result};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Entry {
    bytes: u64,
    last_used_ns: u64,
    pins: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LruCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub insertions: u64,
    pub evictions: u64,
    pub rejected_oversize: u64,
}

/// Global (cross-layer) expert page cache bounded by resident bytes.
///
/// Deterministic: time is supplied by callers (`now_ns`), never sampled from
/// the wall clock, so replays of an access stream are bit-identical.
#[derive(Debug)]
pub struct LruExpertCache {
    capacity_bytes: u64,
    entries: HashMap<PageId, Entry>,
    resident_bytes: u64,
    stats: LruCacheStats,
}

impl LruExpertCache {
    pub fn new(capacity_bytes: u64) -> Result<Self> {
        if capacity_bytes == 0 {
            return Err(ResidencyError::Invalid(
                "expert cache capacity must be > 0".into(),
            ));
        }
        Ok(Self {
            capacity_bytes,
            entries: HashMap::new(),
            resident_bytes: 0,
            stats: LruCacheStats::default(),
        })
    }

    pub fn capacity_bytes(&self) -> u64 {
        self.capacity_bytes
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn stats(&self) -> &LruCacheStats {
        &self.stats
    }

    pub fn contains(&self, page: &PageId) -> bool {
        self.entries.contains_key(page)
    }

    /// Record a use. Hit bumps recency; miss leaves the stream unchanged.
    pub fn touch(&mut self, page: &PageId, now_ns: u64) -> bool {
        match self.entries.get_mut(page) {
            Some(entry) => {
                entry.last_used_ns = now_ns;
                self.stats.hits += 1;
                true
            }
            None => {
                self.stats.misses += 1;
                false
            }
        }
    }

    /// Insert (or refresh) a page of `bytes`, evicting LRU-unpinned victims
    /// until it fits. An entry larger than the whole cache is rejected.
    pub fn insert(&mut self, page: PageId, bytes: u64, now_ns: u64) -> Result<()> {
        if bytes > self.capacity_bytes {
            self.stats.rejected_oversize += 1;
            return Err(ResidencyError::Invalid(format!(
                "page {} bytes {} exceed cache capacity {}",
                page.ordinal, bytes, self.capacity_bytes
            )));
        }
        if let Some(existing) = self.entries.get_mut(&page) {
            // Refresh recency/size; keep pin count.
            let pins = existing.pins;
            self.resident_bytes = self.resident_bytes - existing.bytes + bytes;
            *existing = Entry {
                bytes,
                last_used_ns: now_ns,
                pins,
            };
            self.stats.insertions += 1;
            self.evict_until_fit();
            return Ok(());
        }
        self.make_room(bytes)?;
        self.entries.insert(
            page.clone(),
            Entry {
                bytes,
                last_used_ns: now_ns,
                pins: 0,
            },
        );
        self.resident_bytes += bytes;
        self.stats.insertions += 1;
        Ok(())
    }

    /// Pin prevents eviction; repeated pins stack and must be balanced.
    pub fn pin(&mut self, page: &PageId, now_ns: u64) -> Result<()> {
        let entry = self.entries.get_mut(page).ok_or_else(|| {
            ResidencyError::Invalid("cannot pin a page that is not resident".into())
        })?;
        entry.last_used_ns = now_ns;
        entry.pins = entry.pins.saturating_add(1);
        Ok(())
    }

    pub fn unpin(&mut self, page: &PageId) -> Result<()> {
        let entry = self.entries.get_mut(page).ok_or_else(|| {
            ResidencyError::Invalid("cannot unpin a page that is not resident".into())
        })?;
        if entry.pins == 0 {
            return Err(ResidencyError::Invalid("unpin below zero".into()));
        }
        entry.pins -= 1;
        Ok(())
    }

    fn make_room(&mut self, incoming_bytes: u64) -> Result<()> {
        while self.resident_bytes + incoming_bytes > self.capacity_bytes {
            let victim = self
                .entries
                .iter()
                .filter(|(_, e)| e.pins == 0)
                .min_by_key(|(_, e)| (e.last_used_ns, e.bytes))
                .map(|(k, _)| k.clone());
            let victim = victim.ok_or_else(|| {
                ResidencyError::Invalid("cache full of pinned pages; cannot evict".into())
            })?;
            if let Some(entry) = self.entries.remove(&victim) {
                self.resident_bytes -= entry.bytes;
                self.stats.evictions += 1;
            }
        }
        Ok(())
    }

    fn evict_until_fit(&mut self) {
        // Used after resize; oversized-but-legal inserts still need room.
        while self.resident_bytes > self.capacity_bytes {
            let victim = self
                .entries
                .iter()
                .filter(|(_, e)| e.pins == 0)
                .min_by_key(|(_, e)| (e.last_used_ns, e.bytes))
                .map(|(k, _)| k.clone());
            match victim {
                Some(v) => {
                    if let Some(entry) = self.entries.remove(&v) {
                        self.resident_bytes -= entry.bytes;
                        self.stats.evictions += 1;
                    }
                }
                None => break,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ModelRoot, PageKind};

    fn page(n: u64) -> PageId {
        PageId {
            model_root: ModelRoot::new("test-model"),
            kind: PageKind::Weights,
            ordinal: n,
        }
    }

    #[test]
    fn capacity_must_be_nonzero() {
        assert!(LruExpertCache::new(0).is_err());
    }

    #[test]
    fn touch_distinguishes_hit_from_miss() {
        let mut c = LruExpertCache::new(1024).unwrap();
        c.insert(page(1), 256, 10).unwrap();
        assert!(c.touch(&page(1), 20));
        assert!(!c.touch(&page(9), 30));
        assert_eq!(c.stats().hits, 1);
        assert_eq!(c.stats().misses, 1);
    }

    #[test]
    fn evicts_least_recently_used_first() {
        let mut c = LruExpertCache::new(512).unwrap();
        for n in 0..2 {
            c.insert(page(n), 256, n).unwrap();
        }
        c.touch(&page(0), 100); // refresh page 0 recency (hit)
        c.touch(&page(0), 100);
        c.insert(page(2), 256, 200).unwrap(); // must evict page 1 (older recency)
        assert!(c.contains(&page(0)));
        assert!(!c.contains(&page(1)));
        assert!(c.contains(&page(2)));
        assert_eq!(c.stats().evictions, 1);
    }

    #[test]
    fn oversize_page_rejected_fail_closed() {
        let mut c = LruExpertCache::new(256).unwrap();
        assert!(c.insert(page(1), 512, 0).is_err());
        assert_eq!(c.stats().rejected_oversize, 1);
        assert!(c.is_empty());
    }

    #[test]
    fn pinned_pages_survive_eviction() {
        let mut c = LruExpertCache::new(512).unwrap();
        c.insert(page(0), 256, 0).unwrap();
        c.insert(page(1), 256, 1).unwrap();
        c.pin(&page(0), 2).unwrap();
        assert!(c.insert(page(2), 256, 3).is_ok()); // evicts page 1 only
        assert!(c.contains(&page(0)));
        assert!(!c.contains(&page(1)));
        // Only page 0 is pinned; page 2 is a legal victim.
        assert!(c.insert(page(3), 256, 4).is_ok());
        assert!(c.contains(&page(0)));
        assert!(!c.contains(&page(2)));
        assert_eq!(c.stats().evictions, 2);
        // All-unpinnable state: only the pinned page remains, no victims.
        let mut full = LruExpertCache::new(512).unwrap();
        full.insert(page(7), 512, 0).unwrap();
        full.pin(&page(7), 1).unwrap();
        assert!(full.insert(page(8), 128, 2).is_err());
    }

    #[test]
    fn pin_unpin_must_balance() {
        let mut c = LruExpertCache::new(512).unwrap();
        c.insert(page(0), 128, 0).unwrap();
        assert!(c.unpin(&page(0)).is_err());
        c.pin(&page(0), 1).unwrap();
        c.pin(&page(0), 2).unwrap();
        c.unpin(&page(0)).unwrap();
        c.unpin(&page(0)).unwrap();
        assert!(c.unpin(&page(0)).is_err());
    }

    #[test]
    fn reinsert_resizes_and_refreshes() {
        let mut c = LruExpertCache::new(512).unwrap();
        c.insert(page(0), 256, 0).unwrap();
        c.insert(page(0), 384, 5).unwrap(); // same page, bigger payload
        assert_eq!(c.resident_bytes(), 384);
        assert_eq!(c.len(), 1);
        assert_eq!(c.stats().insertions, 2);
        c.insert(page(1), 128, 6).unwrap();
        assert_eq!(c.resident_bytes(), 512);
        assert_eq!(c.stats().evictions, 0);
    }
}
