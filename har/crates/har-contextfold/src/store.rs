use har_kv::{KVPageId, PhysicalPageLocation, StableDigest};
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StoreError {
    MissingPage,
    EmptyPayload,
    StoreRejected(String),
    Io(String),
}
impl fmt::Display for StoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingPage => formatter.write_str("page is absent from store"),
            Self::EmptyPayload => formatter.write_str("empty page payload"),
            Self::StoreRejected(value) => formatter.write_str(value),
            Self::Io(value) => write!(formatter, "store I/O: {value}"),
        }
    }
}
impl std::error::Error for StoreError {}

/// residency layer owns the physical implementation. This trait is the only storage
/// dependency of ContextFold: logical IDs and representation semantics remain
/// in policy layer's crate.
pub trait KVPageStore {
    fn store_id(&self) -> &str;
    fn put(
        &mut self,
        page_id: &KVPageId,
        payload: Vec<u8>,
        generation: u64,
    ) -> Result<PhysicalPageLocation, StoreError>;
    fn get(
        &self,
        page_id: &KVPageId,
        location: &PhysicalPageLocation,
    ) -> Result<Vec<u8>, StoreError>;
    fn remove(
        &mut self,
        page_id: &KVPageId,
        location: &PhysicalPageLocation,
    ) -> Result<(), StoreError>;
}

#[derive(Clone, Debug, Default)]
pub struct MemoryPageStore {
    id: String,
    next_offset: u64,
    entries: BTreeMap<String, (PhysicalPageLocation, Vec<u8>)>,
}

impl MemoryPageStore {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            next_offset: 0,
            entries: BTreeMap::new(),
        }
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl KVPageStore for MemoryPageStore {
    fn store_id(&self) -> &str {
        &self.id
    }
    fn put(
        &mut self,
        page_id: &KVPageId,
        payload: Vec<u8>,
        generation: u64,
    ) -> Result<PhysicalPageLocation, StoreError> {
        if payload.is_empty() {
            return Err(StoreError::EmptyPayload);
        }
        let location = PhysicalPageLocation {
            store_id: self.id.clone(),
            offset: self.next_offset,
            length: payload.len() as u64,
            generation,
        };
        self.next_offset = self.next_offset.saturating_add(payload.len() as u64);
        self.entries
            .insert(page_id.key(), (location.clone(), payload));
        Ok(location)
    }
    fn get(
        &self,
        page_id: &KVPageId,
        location: &PhysicalPageLocation,
    ) -> Result<Vec<u8>, StoreError> {
        let (stored, payload) = self
            .entries
            .get(&page_id.key())
            .ok_or(StoreError::MissingPage)?;
        if stored != location {
            return Err(StoreError::StoreRejected(
                "physical location mismatch".to_string(),
            ));
        }
        Ok(payload.clone())
    }
    fn remove(
        &mut self,
        page_id: &KVPageId,
        location: &PhysicalPageLocation,
    ) -> Result<(), StoreError> {
        let (stored, _) = self
            .entries
            .get(&page_id.key())
            .ok_or(StoreError::MissingPage)?;
        if stored != location {
            return Err(StoreError::StoreRejected(
                "physical location mismatch".to_string(),
            ));
        }
        self.entries.remove(&page_id.key());
        Ok(())
    }
}

/// Small file-backed reference store used by the real-page orchestration
/// example. It is a test/reference implementation of residency layer's trait, not a
/// claim about the production NVMe/VRAM transfer path.
#[derive(Clone, Debug)]
pub struct FilePageStore {
    root: PathBuf,
    id: String,
    entries: BTreeMap<String, PhysicalPageLocation>,
}

impl FilePageStore {
    pub fn new(root: impl AsRef<Path>, id: impl Into<String>) -> Result<Self, StoreError> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root).map_err(|error| StoreError::Io(error.to_string()))?;
        Ok(Self {
            root,
            id: id.into(),
            entries: BTreeMap::new(),
        })
    }

    fn path_for(&self, page_id: &KVPageId) -> PathBuf {
        self.root
            .join(format!("{}.page", StableDigest::from_text(&page_id.key())))
    }
}

impl KVPageStore for FilePageStore {
    fn store_id(&self) -> &str {
        &self.id
    }
    fn put(
        &mut self,
        page_id: &KVPageId,
        payload: Vec<u8>,
        generation: u64,
    ) -> Result<PhysicalPageLocation, StoreError> {
        if payload.is_empty() {
            return Err(StoreError::EmptyPayload);
        }
        let path = self.path_for(page_id);
        fs::write(&path, &payload).map_err(|error| StoreError::Io(error.to_string()))?;
        let location = PhysicalPageLocation {
            store_id: self.id.clone(),
            offset: 0,
            length: payload.len() as u64,
            generation,
        };
        self.entries.insert(page_id.key(), location.clone());
        Ok(location)
    }
    fn get(
        &self,
        page_id: &KVPageId,
        location: &PhysicalPageLocation,
    ) -> Result<Vec<u8>, StoreError> {
        if location.store_id != self.id {
            return Err(StoreError::StoreRejected(
                "store identity mismatch".to_string(),
            ));
        }
        if self.entries.get(&page_id.key()) != Some(location) {
            return Err(StoreError::StoreRejected(
                "stale physical location".to_string(),
            ));
        }
        let path = self.path_for(page_id);
        let bytes = fs::read(path).map_err(|_| StoreError::MissingPage)?;
        if bytes.len() as u64 != location.length {
            return Err(StoreError::StoreRejected(
                "stored length mismatch".to_string(),
            ));
        }
        Ok(bytes)
    }
    fn remove(
        &mut self,
        page_id: &KVPageId,
        location: &PhysicalPageLocation,
    ) -> Result<(), StoreError> {
        if location.store_id != self.id {
            return Err(StoreError::StoreRejected(
                "store identity mismatch".to_string(),
            ));
        }
        if self.entries.get(&page_id.key()) != Some(location) {
            return Err(StoreError::StoreRejected(
                "stale physical location".to_string(),
            ));
        }
        fs::remove_file(self.path_for(page_id)).map_err(|_| StoreError::MissingPage)?;
        self.entries.remove(&page_id.key());
        Ok(())
    }
}
