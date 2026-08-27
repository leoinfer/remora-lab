//! Generic page-store boundary shared by weights and KV.
//!
//! The store only answers identity and bytes.  It does not decide leases,
//! generations, eviction, or GPU ownership; those remain in `ResidencyManager`.

use crate::types::{PageId, Result, StorageSlice};
use std::collections::BTreeMap;
use std::sync::RwLock;

pub trait PageStore: Send + Sync {
    fn lookup(&self, page_id: &PageId) -> Result<StorageSlice>;
    fn read(&self, slice: &StorageSlice) -> Result<Vec<u8>>;
}

#[derive(Debug, Default)]
pub struct InMemoryPageStore {
    slices: RwLock<BTreeMap<PageId, (StorageSlice, Vec<u8>)>>,
}

impl InMemoryPageStore {
    pub fn insert(&self, slice: StorageSlice, bytes: Vec<u8>) -> Result<()> {
        if bytes.len() as u64 != slice.payload_bytes {
            return Err(crate::types::ResidencyError::Invalid(
                "page fixture size does not match StorageSlice".into(),
            ));
        }
        self.slices
            .write()
            .map_err(|_| crate::types::ResidencyError::Invalid("page store lock poisoned".into()))?
            .insert(slice.page_id.clone(), (slice, bytes));
        Ok(())
    }
}

impl PageStore for InMemoryPageStore {
    fn lookup(&self, page_id: &PageId) -> Result<StorageSlice> {
        self.slices
            .read()
            .map_err(|_| crate::types::ResidencyError::Invalid("page store lock poisoned".into()))?
            .get(page_id)
            .map(|entry| entry.0.clone())
            .ok_or_else(|| crate::types::ResidencyError::Invalid("page not found".into()))
    }

    fn read(&self, slice: &StorageSlice) -> Result<Vec<u8>> {
        self.slices
            .read()
            .map_err(|_| crate::types::ResidencyError::Invalid("page store lock poisoned".into()))?
            .get(&slice.page_id)
            .filter(|entry| {
                entry.0.offset == slice.offset && entry.0.payload_bytes == slice.payload_bytes
            })
            .map(|entry| entry.1.clone())
            .ok_or_else(|| {
                crate::types::ResidencyError::Invalid(
                    "page slice does not match store identity".into(),
                )
            })
    }
}
