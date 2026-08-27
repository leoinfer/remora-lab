//! Bounded positioned I/O.
//!
//! All unsafe code for aligned Linux O_DIRECT buffers lives in the small
//! `unix_direct_io` module below.  The scheduler sees only `ReadResult` and a
//! cancellation-safe handle; it never touches file descriptors or pointers.

use har_residency::{ResidencyError, Result};
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Condvar, Mutex,
};
use std::thread::{self, JoinHandle};
use std::time::Instant;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectIoCapabilities {
    pub path: PathBuf,
    pub odirect_available: bool,
    pub alignment: u64,
    pub filesystem_id: Option<u64>,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReadRequest {
    pub path: PathBuf,
    pub offset: u64,
    pub bytes: u64,
    pub alignment: u64,
    pub tensor_id: String,
    pub mandatory: bool,
    pub speculative: bool,
}

impl ReadRequest {
    pub fn validate(&self) -> Result<()> {
        if self.path.as_os_str().is_empty() || self.alignment == 0 {
            return Err(ResidencyError::Invalid(
                "I/O path/alignment is empty".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReadResult {
    pub data: Vec<u8>,
    pub direct_io: bool,
    pub aligned_offset: u64,
    pub aligned_bytes: u64,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
    pub page_cache_bytes: u64,
    pub bounce_bytes: u64,
    pub first_byte_ns: u128,
    pub completed_ns: u128,
    pub queue_wait_ns: u128,
}

impl ReadResult {
    pub fn alignment_penalty_bytes(&self) -> u64 {
        self.aligned_bytes.saturating_sub(self.logical_bytes)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct IoAccounting {
    pub requests: u64,
    pub completed: u64,
    pub cancelled: u64,
    pub failed: u64,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
    pub page_cache_bytes: u64,
    pub direct_io_bytes: u64,
    pub bounce_bytes: u64,
    pub alignment_penalty_bytes: u64,
    pub first_byte_ns: Vec<u128>,
    pub latency_ns: Vec<u128>,
}

#[derive(Debug)]
struct QueueState {
    in_flight: usize,
    accounting: IoAccounting,
}

#[derive(Debug)]
struct Inner {
    queue_depth: usize,
    alignment: u64,
    bounce_buffer_bytes: u64,
    state: Mutex<QueueState>,
    cv: Condvar,
    closed: AtomicBool,
}

#[derive(Debug)]
pub struct ReadHandle {
    cancel: Arc<AtomicBool>,
    join: Option<JoinHandle<Result<ReadResult>>>,
}

impl ReadHandle {
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    pub fn join(mut self) -> Result<ReadResult> {
        self.join
            .take()
            .expect("read handle already joined")
            .join()
            .map_err(|_| ResidencyError::Io("I/O worker panicked".into()))?
    }
}

#[derive(Clone, Debug)]
pub struct DirectIoEngine {
    inner: Arc<Inner>,
}

impl DirectIoEngine {
    pub fn new(queue_depth: usize, alignment: u64, bounce_buffer_bytes: u64) -> Result<Self> {
        if queue_depth == 0
            || alignment == 0
            || bounce_buffer_bytes < alignment
            || bounce_buffer_bytes % alignment != 0
        {
            return Err(ResidencyError::Invalid(
                "invalid direct-I/O queue/alignment/bounce configuration".into(),
            ));
        }
        Ok(Self {
            inner: Arc::new(Inner {
                queue_depth,
                alignment,
                bounce_buffer_bytes,
                state: Mutex::new(QueueState {
                    in_flight: 0,
                    accounting: IoAccounting::default(),
                }),
                cv: Condvar::new(),
                closed: AtomicBool::new(false),
            }),
        })
    }

    pub fn alignment(&self) -> u64 {
        self.inner.alignment
    }
    pub fn queue_depth(&self) -> usize {
        self.inner.queue_depth
    }
    pub fn bounce_buffer_bytes(&self) -> u64 {
        self.inner.bounce_buffer_bytes
    }

    pub fn probe(&self, path: impl AsRef<Path>) -> DirectIoCapabilities {
        let path = path.as_ref().to_path_buf();
        let filesystem_id = std::fs::metadata(&path)
            .ok()
            .and_then(|_| std::fs::File::open(&path).ok())
            .and_then(|file| file.metadata().ok())
            .map(|metadata| metadata.len());
        #[cfg(target_os = "linux")]
        {
            match OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECT)
                .open(&path)
            {
                Ok(file) => {
                    drop(file);
                    DirectIoCapabilities {
                        path,
                        odirect_available: true,
                        alignment: self.alignment(),
                        filesystem_id,
                        reason: "O_DIRECT open succeeded".into(),
                    }
                }
                Err(error) => DirectIoCapabilities {
                    path,
                    odirect_available: false,
                    alignment: self.alignment(),
                    filesystem_id,
                    reason: format!("O_DIRECT unavailable: {error}"),
                },
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            DirectIoCapabilities {
                path,
                odirect_available: false,
                alignment: self.alignment(),
                filesystem_id,
                reason: "Linux O_DIRECT path is unavailable on this target".into(),
            }
        }
    }

    pub fn submit(&self, request: ReadRequest) -> Result<ReadHandle> {
        request.validate()?;
        if self.inner.closed.load(Ordering::Acquire) {
            return Err(ResidencyError::Invalid(
                "direct-I/O engine is closed".into(),
            ));
        }
        let permit_start = Instant::now();
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| ResidencyError::Io("I/O queue mutex poisoned".into()))?;
        while state.in_flight >= self.inner.queue_depth {
            state = self
                .inner
                .cv
                .wait(state)
                .map_err(|_| ResidencyError::Io("I/O queue condition poisoned".into()))?;
        }
        state.in_flight += 1;
        state.accounting.requests += 1;
        let queue_wait_ns = permit_start.elapsed().as_nanos();
        drop(state);
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_worker = Arc::clone(&cancel);
        let inner = Arc::clone(&self.inner);
        let join = thread::spawn(move || {
            let result = read_one(&request, &inner, &cancel_worker, queue_wait_ns);
            let mut state = inner
                .state
                .lock()
                .map_err(|_| ResidencyError::Io("I/O queue mutex poisoned".into()))?;
            state.in_flight = state.in_flight.saturating_sub(1);
            match &result {
                Ok(value) => {
                    state.accounting.completed += 1;
                    state.accounting.logical_bytes += value.logical_bytes;
                    state.accounting.physical_bytes += value.physical_bytes;
                    state.accounting.page_cache_bytes += value.page_cache_bytes;
                    if value.direct_io {
                        state.accounting.direct_io_bytes += value.physical_bytes;
                    }
                    state.accounting.bounce_bytes += value.bounce_bytes;
                    state.accounting.alignment_penalty_bytes += value.alignment_penalty_bytes();
                    state
                        .accounting
                        .first_byte_ns
                        .push(value.first_byte_ns.saturating_sub(value.queue_wait_ns));
                    state
                        .accounting
                        .latency_ns
                        .push(value.completed_ns.saturating_sub(value.queue_wait_ns));
                }
                Err(ResidencyError::Cancelled) => state.accounting.cancelled += 1,
                Err(_) => state.accounting.failed += 1,
            }
            inner.cv.notify_one();
            result
        });
        Ok(ReadHandle {
            cancel,
            join: Some(join),
        })
    }

    pub fn read(&self, request: ReadRequest) -> Result<ReadResult> {
        self.submit(request)?.join()
    }

    pub fn accounting(&self) -> IoAccounting {
        self.inner
            .state
            .lock()
            .map(|state| state.accounting.clone())
            .unwrap_or_default()
    }

    pub fn close(&self) {
        self.inner.closed.store(true, Ordering::Release);
        self.inner.cv.notify_all();
    }
}

fn align_down(value: u64, alignment: u64) -> u64 {
    value / alignment * alignment
}
fn align_up(value: u64, alignment: u64) -> u64 {
    value.saturating_add(alignment - 1) / alignment * alignment
}

fn read_one(
    request: &ReadRequest,
    inner: &Inner,
    cancel: &AtomicBool,
    queue_wait_ns: u128,
) -> Result<ReadResult> {
    if request.bytes == 0 {
        return Ok(ReadResult {
            data: Vec::new(),
            direct_io: false,
            aligned_offset: request.offset,
            aligned_bytes: 0,
            logical_bytes: 0,
            physical_bytes: 0,
            page_cache_bytes: 0,
            bounce_bytes: 0,
            first_byte_ns: queue_wait_ns,
            completed_ns: queue_wait_ns,
            queue_wait_ns,
        });
    }
    let capability = DirectIoEngine {
        inner: Arc::new(Inner {
            queue_depth: 1,
            alignment: request.alignment,
            bounce_buffer_bytes: inner.bounce_buffer_bytes,
            state: Mutex::new(QueueState {
                in_flight: 0,
                accounting: IoAccounting::default(),
            }),
            cv: Condvar::new(),
            closed: AtomicBool::new(false),
        }),
    }
    .probe(&request.path);
    let direct = capability.odirect_available;
    let aligned_offset = if direct {
        align_down(request.offset, request.alignment)
    } else {
        request.offset
    };
    let aligned_end = if direct {
        align_up(
            request.offset.saturating_add(request.bytes),
            request.alignment,
        )
    } else {
        request.offset.saturating_add(request.bytes)
    };
    let aligned_bytes = aligned_end.saturating_sub(aligned_offset);
    let mut output = vec![0u8; request.bytes as usize];
    let mut copied = 0usize;
    let mut physical_bytes = 0u64;
    let mut bounce_bytes = 0u64;
    let first_start = Instant::now();
    let file = if direct {
        open_direct(&request.path)?
    } else {
        File::open(&request.path).map_err(|error| ResidencyError::Io(error.to_string()))?
    };
    let chunk_limit = if direct {
        inner.bounce_buffer_bytes
    } else {
        request.bytes.max(1 << 20)
    };
    let mut current = aligned_offset;
    while current < aligned_end {
        if cancel.load(Ordering::Acquire) {
            return Err(ResidencyError::Cancelled);
        }
        let chunk = (aligned_end - current).min(chunk_limit);
        let chunk = if direct {
            align_up(chunk, request.alignment)
        } else {
            chunk
        };
        let got = if direct {
            let count = unix_direct_io::read_aligned(&file, current, chunk as usize)?;
            physical_bytes += count as u64;
            bounce_bytes += chunk;
            count
        } else {
            let mut buffer = vec![0u8; chunk as usize];
            let count = file
                .read_at(&mut buffer, current)
                .map_err(|error| ResidencyError::Io(error.to_string()))?;
            let start = current.max(request.offset);
            let end = (current + count as u64).min(request.offset + request.bytes);
            if end > start {
                let src = (start - current) as usize;
                output[(start - request.offset) as usize..(end - request.offset) as usize]
                    .copy_from_slice(&buffer[src..src + (end - start) as usize]);
                copied += (end - start) as usize;
            }
            physical_bytes += count as u64;
            if count == 0 {
                break;
            }
            current += chunk;
            continue;
        };
        let start = current.max(request.offset);
        let end = (current + got as u64).min(request.offset + request.bytes);
        if end > start {
            let src = (start - current) as usize;
            output[(start - request.offset) as usize..(end - request.offset) as usize]
                .copy_from_slice(&unix_direct_io::last_buffer()[src..src + (end - start) as usize]);
            copied += (end - start) as usize;
        }
        current += chunk;
    }
    if copied != request.bytes as usize {
        return Err(ResidencyError::Io(format!(
            "short positioned read: {copied} of {}",
            request.bytes
        )));
    }
    let first_byte_ns = first_start.elapsed().as_nanos();
    let completed_ns = first_start.elapsed().as_nanos();
    Ok(ReadResult {
        data: output,
        direct_io: direct,
        aligned_offset,
        aligned_bytes,
        logical_bytes: request.bytes,
        physical_bytes,
        page_cache_bytes: if direct { 0 } else { request.bytes },
        bounce_bytes,
        first_byte_ns,
        completed_ns,
        queue_wait_ns,
    })
}

#[cfg(target_os = "linux")]
fn open_direct(path: &Path) -> Result<File> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .map_err(|error| ResidencyError::Io(format!("O_DIRECT open failed: {error}")))
}

#[cfg(not(target_os = "linux"))]
fn open_direct(_path: &Path) -> Result<File> {
    Err(ResidencyError::Unsupported("O_DIRECT is Linux-only".into()))
}

/// The only unsafe boundary in this crate.  `read_aligned` allocates an
/// alignment-sized C buffer, calls `pread`, copies the initialized prefix into
/// a thread-local Vec, then frees it on every path.  The caller supplies an
/// aligned offset and an alignment-multiple length.
#[cfg(target_os = "linux")]
mod unix_direct_io {
    use super::*;
    use std::cell::RefCell;
    thread_local! { static LAST: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) }; }

    pub fn read_aligned(file: &File, offset: u64, bytes: usize) -> Result<usize> {
        if bytes == 0 {
            LAST.with(|last| last.borrow_mut().clear());
            return Ok(0);
        }
        let alignment = 4096usize;
        let mut pointer: *mut libc::c_void = std::ptr::null_mut();
        // SAFETY: `posix_memalign` receives a valid out-pointer, a power-of-two
        // alignment and a non-zero size.  Ownership is released exactly once.
        let allocation = unsafe { libc::posix_memalign(&mut pointer, alignment, bytes) };
        if allocation != 0 || pointer.is_null() {
            return Err(ResidencyError::Io(format!(
                "posix_memalign failed: {allocation}"
            )));
        }
        // SAFETY: pointer is the allocation returned above and bytes is its
        // complete initialized capacity for the duration of `pread`.
        let read_count =
            unsafe { libc::pread(file.as_raw_fd(), pointer, bytes, offset as libc::off_t) };
        if read_count < 0 {
            let error = std::io::Error::last_os_error();
            // SAFETY: pointer came from posix_memalign and has not been freed.
            unsafe {
                libc::free(pointer);
            }
            return Err(ResidencyError::Io(error.to_string()));
        }
        // SAFETY: pread initialized exactly read_count bytes, which are copied
        // before the allocation is freed.
        let bytes_copy = unsafe {
            std::slice::from_raw_parts(pointer.cast::<u8>(), read_count as usize).to_vec()
        };
        // SAFETY: pointer is still the unique allocation owner.
        unsafe {
            libc::free(pointer);
        }
        LAST.with(|last| *last.borrow_mut() = bytes_copy);
        Ok(read_count as usize)
    }

    pub fn last_buffer() -> Vec<u8> {
        LAST.with(|last| last.borrow().clone())
    }
}

#[cfg(not(target_os = "linux"))]
mod unix_direct_io {
    use super::*;
    pub fn read_aligned(_file: &File, _offset: u64, _bytes: usize) -> Result<usize> {
        Err(ResidencyError::Unsupported(
            "direct I/O is Linux-only".into(),
        ))
    }
    pub fn last_buffer() -> Vec<u8> {
        Vec::new()
    }
}

#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
#[cfg(target_os = "linux")]
use std::os::unix::fs::OpenOptionsExt;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn unaligned_read_is_exact_and_accounted() {
        let path = std::env::temp_dir().join(format!("har-direct-{}.bin", std::process::id()));
        let mut file = File::create(&path).unwrap();
        file.write_all(&(0u8..=255).cycle().take(16384).collect::<Vec<_>>())
            .unwrap();
        drop(file);
        let engine = DirectIoEngine::new(1, 4096, 4096).unwrap();
        let result = engine
            .read(ReadRequest {
                path: path.clone(),
                offset: 3,
                bytes: 101,
                alignment: 4096,
                tensor_id: "fixture".into(),
                mandatory: true,
                speculative: false,
            })
            .unwrap();
        assert_eq!(
            result.data,
            (3u8..=255).chain(0u8..).take(101).collect::<Vec<_>>()
        );
        assert_eq!(result.logical_bytes, 101);
        assert!(result.aligned_bytes >= result.logical_bytes);
        if result.direct_io {
            assert_eq!(result.page_cache_bytes, 0);
        } else {
            assert_eq!(result.page_cache_bytes, 101);
        }
        std::fs::remove_file(path).unwrap();
    }
}
